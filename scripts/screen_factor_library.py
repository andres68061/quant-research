#!/usr/bin/env python3
"""
Systematic screen of the new factor library — all factors, one harness, one report.

This is the first *backtest* of the ~30 factor columns added in the 2026-08 data
expansion. Until now they were computed and coverage-checked only; every
``expected_sharpe_range`` in the registry is a literature prior, not our evidence.

Methodology (deliberately boring):

- Every factor goes through the SAME pipeline: ``run_factor_cross_section_backtest``
  long/short (top/bottom 20%), monthly rebalance, 10 bps one-way costs, S&P 500
  point-in-time membership filter, T+1 execution. No per-factor tuning — tuning
  before the first look is how you manufacture a winner.
- Only the literature-oriented side of each factor is tested (``neg_accruals``,
  not both signs): testing both signs doubles N and guarantees one looks good.
- **Multiple testing**: with N factors, the best Sharpe is partly luck by
  construction. Each factor's t-stat is compared against a Šidák-corrected
  threshold at family-wise alpha = 0.05 (ADR 0003), and the full cross-section is
  reported — never just the winner.
- Sub-period Sharpes (per decade) expose factors that only worked pre-2010.

Outputs:
    docs/research/factor_screen_<date>.md   full table, pass/fail, methodology
    data/quality/factor_screen_<date>.json  machine-readable results

Per the strategy-experiment-log skill, the outcome is then recorded in
FAILED_STRATEGIES_LOG.md (negatives, i.e. most of it) and ROADMAP.md (survivors).

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/screen_factor_library.py
    /opt/anaconda3/envs/quant/bin/python scripts/screen_factor_library.py --factors accruals,roa
"""

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.universe_filters import build_universe_filter
from core.metrics.performance import (
    calculate_hit_rate,
    calculate_profit_factor,
    calculate_win_loss_ratio,
)
from core.research.caveats import SURFACE_FACTOR_SCREEN, as_dicts, caveats_for_surface
from core.strategies.factor_runner import run_factor_cross_section_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("screen_factor_library")

FACTORS_DIR = ROOT / "data" / "factors"
RESEARCH_DIR = ROOT / "docs" / "research"
QUALITY_DIR = ROOT / "data" / "quality"

# (factor_column, source_file, family). Oriented side only — the direction the
# published anomaly says to go long.
SCREEN_FACTORS: tuple[tuple[str, str, str], ...] = (
    # Fundamental — quality / profitability
    ("gross_profitability", "factors_fundamental.parquet", "profitability"),
    ("operating_profitability", "factors_fundamental.parquet", "profitability"),
    ("roa", "factors_fundamental.parquet", "profitability"),
    ("cfo_to_assets", "factors_fundamental.parquet", "profitability"),
    ("gross_margin", "factors_fundamental.parquet", "profitability"),
    ("piotroski_f", "factors_fundamental.parquet", "quality-score"),
    ("altman_z", "factors_fundamental.parquet", "quality-score"),
    # Fundamental — earnings quality / investment / issuance
    ("neg_accruals", "factors_fundamental.parquet", "earnings-quality"),
    ("neg_asset_growth", "factors_fundamental.parquet", "investment"),
    ("neg_net_operating_assets", "factors_fundamental.parquet", "earnings-quality"),
    ("neg_capex_intensity", "factors_fundamental.parquet", "investment"),
    ("neg_inventory_growth", "factors_fundamental.parquet", "investment"),
    ("neg_net_share_issuance", "factors_fundamental.parquet", "issuance"),
    ("net_payout_yield", "factors_fundamental.parquet", "issuance"),
    # Fundamental — value
    ("book_to_market", "factors_fundamental.parquet", "value"),
    ("earnings_yield", "factors_fundamental.parquet", "value"),
    ("cash_flow_to_price", "factors_fundamental.parquet", "value"),
    ("sales_to_price", "factors_fundamental.parquet", "value"),
    ("fcf_yield", "factors_fundamental.parquet", "value"),
    ("ebitda_to_ev", "factors_fundamental.parquet", "value"),
    ("rd_to_market", "factors_fundamental.parquet", "value"),
    ("rd_intensity", "factors_fundamental.parquet", "value"),
    # Microstructure
    ("amihud_illiquidity", "factors_microstructure.parquet", "liquidity"),
    ("overnight_intraday_gap", "factors_microstructure.parquet", "microstructure"),
    ("close_location_21d", "factors_microstructure.parquet", "microstructure"),
    ("range_to_close_vol", "factors_microstructure.parquet", "microstructure"),
    ("corwin_schultz_spread", "factors_microstructure.parquet", "liquidity"),
    # Event (calendar-diluted version; the event-time study is separate)
    ("sue_price_scaled", "factors_earnings_surprise.parquet", "earnings-surprise"),
)

START = pd.Timestamp("2000-01-03", tz="America/New_York")
END = pd.Timestamp("2026-08-07", tz="America/New_York")
FAMILY_WISE_ALPHA = 0.05
TRADING_DAYS = 252


def annualized_sharpe(net_returns: pd.Series) -> float:
    daily = net_returns.dropna()
    if len(daily) < TRADING_DAYS or daily.std() == 0:
        return float("nan")
    return float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS))


def t_stat(net_returns: pd.Series) -> float:
    daily = net_returns.dropna()
    if len(daily) < TRADING_DAYS or daily.std() == 0:
        return float("nan")
    return float(daily.mean() / daily.std() * np.sqrt(len(daily)))


def decade_sharpes(net_returns: pd.Series) -> dict[str, float]:
    out = {}
    for label, lo, hi in (
        ("2000s", "2000", "2009"),
        ("2010s", "2010", "2019"),
        ("2020s", "2020", "2026"),
    ):
        chunk = net_returns.loc[lo:hi]
        out[label] = round(annualized_sharpe(chunk), 2) if len(chunk) > TRADING_DAYS else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen the factor library")
    parser.add_argument("--factors", type=str, default=None, help="Comma-separated subset")
    parser.add_argument(
        "--universe",
        type=str,
        default="sp500",
        choices=["sp500", "all"],
        help="sp500 = point-in-time index members only; all = the full panel "
        "(shells/SPACs always excluded — see the non-operating-vehicles caveat)",
    )
    args = parser.parse_args()

    wanted = set(args.factors.split(",")) if args.factors else None
    to_screen = [f for f in SCREEN_FACTORS if wanted is None or f[0] in wanted]

    prices = pd.read_parquet(FACTORS_DIR / "prices.parquet")
    # Shells/SPACs are excluded in BOTH modes: a pre-merger trust account has no
    # factor exposure to rank, and its artificial low volatility lands it in the
    # traded tiers of vol-sensitive factors.
    universe_filter = build_universe_filter(
        prices.columns,
        exclude_non_operating=True,
        index_name="sp500" if args.universe == "sp500" else None,
    )
    logger.info(
        "Screening %d factors on the %s universe, %s -> %s",
        len(to_screen),
        args.universe,
        START.date(),
        END.date(),
    )

    panels: dict[str, pd.DataFrame] = {}
    results: list[dict] = []
    for i, (factor_col, source, family) in enumerate(to_screen, 1):
        if source not in panels:
            panels[source] = pd.read_parquet(FACTORS_DIR / source)
        factors = panels[source]
        if factor_col not in factors.columns:
            logger.warning("%s missing from %s; skipping", factor_col, source)
            continue
        started = time.monotonic()
        try:
            net = run_factor_cross_section_backtest(
                factors[[factor_col]].astype("float64"),
                prices,
                factor_col=factor_col,
                start=START,
                end=END,
                universe_filter=universe_filter,
            )
        except Exception:
            logger.exception("FAILED %s", factor_col)
            results.append({"factor": factor_col, "family": family, "status": "error"})
            continue

        results.append(
            {
                "factor": factor_col,
                "family": family,
                "status": "ok",
                "sharpe_net": round(annualized_sharpe(net), 3),
                "ann_return_net": round(float(net.mean() * TRADING_DAYS), 4),
                "t_stat": round(t_stat(net), 2),
                "max_drawdown": round(
                    float((1 + net).cumprod().div((1 + net).cumprod().cummax()).min() - 1), 3
                ),
                # Sharpe alone cannot distinguish "right often, small" from
                # "rarely right, huge" — and those are different products.
                "hit_rate": round(calculate_hit_rate(net), 3),
                "win_loss_ratio": round(calculate_win_loss_ratio(net), 2),
                "profit_factor": round(calculate_profit_factor(net), 2),
                "by_decade": decade_sharpes(net),
                "n_days": int(net.dropna().shape[0]),
            }
        )
        logger.info(
            "[%d/%d] %-26s sharpe=%.2f t=%.2f (%.0fs)",
            i,
            len(to_screen),
            factor_col,
            results[-1]["sharpe_net"],
            results[-1]["t_stat"],
            time.monotonic() - started,
        )

    ok = [r for r in results if r["status"] == "ok" and np.isfinite(r.get("t_stat", np.nan))]
    n_tests = len(ok)
    per_test_alpha = 1 - (1 - FAMILY_WISE_ALPHA) ** (1 / n_tests) if n_tests else np.nan
    threshold = float(stats.norm.ppf(1 - per_test_alpha / 2)) if n_tests else np.nan
    for r in ok:
        r["sidak_pass"] = bool(abs(r["t_stat"]) >= threshold and r["t_stat"] > 0)

    survivors = [r for r in ok if r.get("sidak_pass")]
    payload = {
        "run_date": str(date.today()),
        "period": f"{START.date()} -> {END.date()}",
        "universe": args.universe,
        "methodology": (
            f"long/short top-bottom 20%, monthly rebalance, 10bps one-way costs, "
            f"{'S&P 500 point-in-time membership' if args.universe == 'sp500' else 'full panel'}, "
            "shells/SPACs excluded, T+1 execution, min 20 stocks"
        ),
        "n_tests": n_tests,
        "family_wise_alpha": FAMILY_WISE_ALPHA,
        "sidak_t_threshold": round(threshold, 2),
        "survivors": [r["factor"] for r in survivors],
        "caveats": as_dicts(caveats_for_surface(SURFACE_FACTOR_SCREEN)),
        "results": sorted(ok, key=lambda r: -(r["sharpe_net"] or -9)),
    }

    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    (QUALITY_DIR / f"factor_screen_{stamp}.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"# Factor library screen — {date.today()}",
        "",
        "## What this document is (read first)",
        "",
        "A **factor** is a rule for ranking stocks — 'cheapest by book-to-market',",
        "'most profitable', 'least indebted'. A **factor backtest** asks the only",
        "question that matters: if you had ranked every stock by this rule, bought",
        "the top 20% and short-sold the bottom 20%, rebalancing monthly, would you",
        "have made money after trading costs?",
        "",
        "Each row below is one factor put through that exact test. **Net Sharpe** is",
        "return per unit of risk after costs (1.0 is very good, 0.5 respectable,",
        "0 means the rule had no predictive value, negative means it predicted",
        "backwards). **t** measures whether the result could plausibly be luck.",
        "",
        "### Why the bar is set so high",
        "",
        f"We tested {n_tests} factors at once. If you test enough rules, the best one",
        "looks impressive purely by chance — flip 28 coins 20 times each and the",
        "luckiest coin looks special. The Šidák correction raises the significance",
        f"bar in proportion to how many things were tested: here |t| ≥ {threshold:.2f}",
        "rather than the usual 2.0. Every factor is published below, winners and",
        "losers, because reading only the winner reintroduces exactly the bias the",
        "correction removes.",
        "",
        "### How to read the result",
        "",
        "**A factor that fails is not necessarily worthless** — it means this rule,",
        "on this universe, over this period, after these costs, did not clear the",
        "bar. Most published anomalies decay after publication and most were",
        "documented on small caps, so a large-cap null is the expected outcome, not",
        "a surprise. The decade columns matter: a factor that worked only in the",
        "2000s is a decayed factor; one that is flat across all three decades is a",
        "candidate worth more work.",
        "",
        "## Setup",
        "",
        f"{payload['methodology']}.",
        f"Period {payload['period']}. **{n_tests} tests**, family-wise alpha "
        f"{FAMILY_WISE_ALPHA}, Šidák per-test threshold |t| ≥ {threshold:.2f}.",
        "",
        "## Results",
        "",
        "| Factor | Family | Net Sharpe | Ann ret | t | Hit | W/L | PF | 2000s | 2010s | 2020s | MaxDD | Šidák |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload["results"]:
        by_decade = r["by_decade"]
        lines.append(
            f"| {r['factor']} | {r['family']} | {r['sharpe_net']:.2f} | "
            f"{r['ann_return_net'] * 100:.1f}% | {r['t_stat']:.2f} | "
            f"{r.get('hit_rate', 0) * 100:.0f}% | {r.get('win_loss_ratio', 0):.2f} | "
            f"{r.get('profit_factor', 0):.2f} | "
            f"{by_decade.get('2000s')} | {by_decade.get('2010s')} | {by_decade.get('2020s')} | "
            f"{r['max_drawdown'] * 100:.0f}% | {'**PASS**' if r.get('sidak_pass') else 'fail'} |"
        )
    lines += [
        "",
        f"Survivors: {', '.join(payload['survivors']) or 'none'}.",
        "",
        "## Disclosed caveats",
        "",
        "From the shared registry (`core/research/caveats.py`) — the same "
        "disclosures shown on every research surface:",
        "",
    ]
    for caveat in caveats_for_surface(SURFACE_FACTOR_SCREEN):
        lines.append(f"- **[{caveat.severity}] {caveat.title}** — {caveat.detail}")
    (RESEARCH_DIR / f"factor_screen_{stamp}.md").write_text("\n".join(lines) + "\n")
    logger.info(
        "DONE: %d tested, %d Šidák survivors: %s",
        n_tests,
        len(survivors),
        payload["survivors"],
    )


if __name__ == "__main__":
    main()
