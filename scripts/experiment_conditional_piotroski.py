#!/usr/bin/env python3
"""
ROADMAP item: test the Piotroski F-score AS THE PAPER SPECIFIES IT.

The 2026-08-11 screen ranked `piotroski_f` across the whole universe and got a
net Sharpe of -0.33. That is not a refutation of Piotroski (2000): the paper
applies the F-score **within the high book-to-market quintile**, as a tool for
separating the recovering value stocks from the value traps. Standalone it is
being asked to do a job it was never specified for, which the registry entry
already warned about.

This script runs the conditional version and three controls. All four are
reported — reading only the best of four is the multiple-testing trap this repo
has an ADR about (0003).

Variants:
    A  piotroski_in_value        F-score within the value quintile, long/short
    B  piotroski_in_value_long   the same, long-only (the paper is mostly a
                                 long-side result)
    C  book_to_market            plain value long/short — baseline for A
    D  piotroski_standalone      reproduces the screen's -0.33 as a control
    E  book_to_market long       plain value LONG-ONLY — the correct baseline for
                                 B. A long-only book's Sharpe is dominated by
                                 market beta, so comparing a long-only variant to
                                 a long/short baseline measures mostly beta.
    F  value_quintile_flat       long-only, every name in the value quintile
                                 equally (F-score ignored) — isolates what the
                                 F-score adds on top of simply being in the
                                 quintile

Implementation note: rather than reimplementing the pipeline, the conditioning is
expressed as a **derived factor** — `piotroski_f` masked to NaN outside that
date's value quintile — and fed to the shared cross-section runner, so costs,
T+1 lag, delisting handling and universe filtering are identical to every other
result in the repo.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/experiment_conditional_piotroski.py
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.factor_store import FactorStore
from core.data.universe_filters import build_universe_filter
from core.strategies.factor_runner import run_factor_cross_section_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("experiment_conditional_piotroski")

FACTORS_DIR = ROOT / "data" / "factors"
PRICES_PATH = FACTORS_DIR / "prices.parquet"
QUALITY_DIR = ROOT / "data" / "quality"

START = pd.Timestamp("2000-01-03", tz="America/New_York")
END = pd.Timestamp("2026-08-07", tz="America/New_York")
TRADING_DAYS = 252

# Piotroski's own screen is the top BM quintile.
VALUE_QUANTILE = 0.80
# F-score is an integer 0-9, so ties are everywhere. Wide tiers within the
# already-narrow value quintile keep each leg populated; narrow tiers would
# mostly select on the tie-breaking order rather than on the score.
TIER_PCT = 0.30
MIN_VALUE_NAMES = 30


def annualized_sharpe(net_returns: pd.Series) -> float:
    """Annualized Sharpe of a daily net-return series (0 risk-free)."""
    daily = net_returns.dropna()
    if len(daily) < TRADING_DAYS or daily.std() == 0:
        return float("nan")
    return float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS))


def t_statistic(net_returns: pd.Series) -> float:
    """t-stat of the mean daily net return."""
    daily = net_returns.dropna()
    if len(daily) < TRADING_DAYS or daily.std() == 0:
        return float("nan")
    return float(daily.mean() / daily.std() * np.sqrt(len(daily)))


def max_drawdown(net_returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the compounded series."""
    curve = (1 + net_returns.fillna(0)).cumprod()
    return float((curve / curve.cummax()).min() - 1)


def decade_sharpes(net_returns: pd.Series) -> dict[str, float | None]:
    """Sharpe per decade — exposes factors that only worked in one regime."""
    out: dict[str, float | None] = {}
    for label, low, high in (
        ("2000s", "2000", "2009"),
        ("2010s", "2010", "2019"),
        ("2020s", "2020", "2026"),
    ):
        chunk = net_returns.loc[low:high]
        out[label] = round(annualized_sharpe(chunk), 2) if len(chunk) > TRADING_DAYS else None
    return out


def mask_to_value_quintile(
    piotroski: pd.DataFrame,
    book_to_market: pd.DataFrame,
    quantile: float = VALUE_QUANTILE,
    min_names: int = MIN_VALUE_NAMES,
) -> pd.DataFrame:
    """
    Blank the F-score outside each date's high book-to-market quintile.

    The breakpoint is computed **within each date** from that date's own
    cross-section — never over the full sample, which would rank a 2003 name
    against a distribution that did not exist until 2020.

    Args:
        piotroski: ``(date, symbol)`` frame with ``piotroski_f``.
        book_to_market: ``(date, symbol)`` frame with ``book_to_market``.
        quantile: Cutoff for "value"; 0.80 = top quintile.
        min_names: Dates with fewer valued names than this are dropped entirely
            rather than traded on a handful of stocks.

    Returns:
        One-column frame ``piotroski_in_value`` with NaN outside the quintile.
    """
    combined = piotroski.join(book_to_market, how="inner").dropna()
    if combined.empty:
        return pd.DataFrame(columns=["piotroski_in_value"])

    grouped = combined.groupby(level="date")["book_to_market"]
    breakpoint_by_date = grouped.transform(lambda values: values.quantile(quantile))
    names_by_date = grouped.transform("count")

    in_value = (combined["book_to_market"] >= breakpoint_by_date) & (names_by_date >= min_names)
    masked = combined["piotroski_f"].where(in_value)
    result = masked.dropna().to_frame("piotroski_in_value")

    logger.info(
        "Value-conditioned rows: %s of %s (%.1f%%), %s dates",
        f"{len(result):,}",
        f"{len(combined):,}",
        100 * len(result) / max(len(combined), 1),
        f"{result.index.get_level_values('date').nunique():,}",
    )
    return result


def run_variant(
    label: str,
    factors: pd.DataFrame,
    factor_col: str,
    prices: pd.DataFrame,
    universe_filter,
    long_only: bool = False,
    tier_pct: float = 0.20,
) -> dict:
    """Run one variant through the shared cross-section runner and score it."""
    net = run_factor_cross_section_backtest(
        factors,
        prices,
        factor_col=factor_col,
        start=START,
        end=END,
        top_pct=tier_pct,
        bottom_pct=tier_pct,
        long_only=long_only,
        universe_filter=universe_filter,
    )
    result = {
        "variant": label,
        "factor": factor_col,
        "long_only": long_only,
        "tier_pct": tier_pct,
        "sharpe_net": round(annualized_sharpe(net), 3),
        "ann_return_net": round(float(net.mean() * TRADING_DAYS), 4),
        "t_stat": round(t_statistic(net), 2),
        "max_drawdown": round(max_drawdown(net), 3),
        "by_decade": decade_sharpes(net),
        "n_days": int(net.dropna().shape[0]),
    }
    logger.info(
        "%-28s sharpe=%6.2f  t=%5.2f  ann=%6.1f%%  maxDD=%5.0f%%  decades=%s",
        label,
        result["sharpe_net"],
        result["t_stat"],
        result["ann_return_net"] * 100,
        result["max_drawdown"] * 100,
        result["by_decade"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional Piotroski experiment")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(PRICES_PATH)
    # Scope the store to the fundamentals panel: the price panels are large and
    # this experiment needs none of their columns.
    store = FactorStore(FACTORS_DIR, panel_priority=("factors_fundamental.parquet",))
    piotroski = store.load_factor("piotroski_f")
    book_to_market = store.load_factor("book_to_market")

    # Shells/SPACs are catastrophic for a value screen specifically: a
    # pre-merger trust with tiny book equity and a $10 price looks like an
    # extreme on whichever side the sign falls.
    universe_filter = build_universe_filter(
        prices.columns, exclude_non_operating=True, index_name="sp500"
    )

    conditioned = mask_to_value_quintile(piotroski, book_to_market)
    results = [
        run_variant(
            "A piotroski_in_value L/S",
            conditioned,
            "piotroski_in_value",
            prices,
            universe_filter,
            tier_pct=TIER_PCT,
        ),
        run_variant(
            "B piotroski_in_value long",
            conditioned,
            "piotroski_in_value",
            prices,
            universe_filter,
            long_only=True,
            tier_pct=TIER_PCT,
        ),
        run_variant(
            "C book_to_market (baseline)",
            book_to_market,
            "book_to_market",
            prices,
            universe_filter,
        ),
        run_variant(
            "D piotroski standalone",
            piotroski,
            "piotroski_f",
            prices,
            universe_filter,
            tier_pct=TIER_PCT,
        ),
        run_variant(
            "E book_to_market long",
            book_to_market,
            "book_to_market",
            prices,
            universe_filter,
            long_only=True,
        ),
        # A constant inside the value quintile ranks every member equally, so the
        # top tier is an arbitrary slice of the quintile — i.e. "hold value
        # stocks" with no F-score information at all. If B does not beat F, the
        # F-score contributed nothing and the result is just the value quintile.
        run_variant(
            "F value quintile flat long",
            conditioned.assign(value_quintile_flat=1.0)[["value_quintile_flat"]],
            "value_quintile_flat",
            prices,
            universe_filter,
            long_only=True,
            tier_pct=TIER_PCT,
        ),
    ]

    by_letter = {r["variant"][0]: r for r in results}
    verdict = {
        "long_short": {
            "conditional_A": by_letter["A"]["sharpe_net"],
            "plain_value_C": by_letter["C"]["sharpe_net"],
            "conditioning_helps": bool(by_letter["A"]["sharpe_net"] > by_letter["C"]["sharpe_net"]),
        },
        "long_only": {
            "conditional_B": by_letter["B"]["sharpe_net"],
            "plain_value_E": by_letter["E"]["sharpe_net"],
            "quintile_flat_F": by_letter["F"]["sharpe_net"],
            "conditioning_helps_vs_plain_value": bool(
                by_letter["B"]["sharpe_net"] > by_letter["E"]["sharpe_net"]
            ),
            "fscore_adds_over_just_being_in_the_quintile": bool(
                by_letter["B"]["sharpe_net"] > by_letter["F"]["sharpe_net"]
            ),
        },
        "note": (
            "Compare long-only to long-only and long/short to long/short. A "
            "long-only book's Sharpe is dominated by market beta, so B beating the "
            "long/short baseline C would prove nothing. The question the F-score "
            "must answer is whether B beats F — holding the SAME value quintile "
            "with the score ignored. Six variants were tested; the best of six is "
            "inflated by selection (ADR 0003)."
        ),
    }

    payload = {
        "run_date": str(date.today()),
        "period": f"{START.date()} -> {END.date()}",
        "methodology": (
            f"long/short (or long-only) top/bottom {TIER_PCT:.0%} tiers, monthly rebalance, "
            "10bps one-way, S&P 500 point-in-time membership, shells/SPACs excluded, "
            f"value quintile = top {1 - VALUE_QUANTILE:.0%} book_to_market computed per date"
        ),
        "results": results,
        "verdict": verdict,
    }

    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.out or QUALITY_DIR / f"experiment_conditional_piotroski_{date.today():%Y%m%d}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", out_path)
    logger.info(
        "VERDICT long/short: conditional %.2f vs plain value %.2f",
        by_letter["A"]["sharpe_net"],
        by_letter["C"]["sharpe_net"],
    )
    logger.info(
        "VERDICT long-only: conditional %.2f vs plain value %.2f vs quintile-flat %.2f "
        "-- the F-score adds value only if it beats quintile-flat",
        by_letter["B"]["sharpe_net"],
        by_letter["E"]["sharpe_net"],
        by_letter["F"]["sharpe_net"],
    )


if __name__ == "__main__":
    main()
