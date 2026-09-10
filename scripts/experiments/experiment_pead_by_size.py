#!/usr/bin/env python3
"""
ROADMAP item: does post-earnings drift live in small caps, as the literature says?

The 2026-08-11 PEAD study ran on the 774-name S&P panel and found a Q5-Q1 spread
of +0.60% at day 60, t=1.5 — not significant. The registered caveat
(``pead-large-cap-universe``) says that is the expected outcome: Bernard & Thomas
and everyone after them find the drift concentrated in small, low-coverage names,
which an S&P-scale universe excludes by construction.

The ADR-0013 cutover makes the real test possible: the earnings panel now covers
5,061 symbols and 267,860 announcements, most of them outside the S&P 500.

This script runs the same event study **three times**, splitting events by the
announcing company's market cap at the time of the announcement (terciles
computed within each calendar quarter, so a 2003 small cap is judged against 2003
peers). If the literature is right, the drift spread should be largest in the
small tercile and near zero in the large one.

Shells/SPACs are excluded: a pre-merger trust has no earnings to be surprised by.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/experiments/experiment_pead_by_size.py
    /opt/anaconda3/envs/quant/bin/python scripts/experiments/experiment_pead_by_size.py --horizon 60
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtest.event_study import extract_events_from_surprise_panel, run_event_study
from core.data.universe.filters import load_non_operating_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("experiment_pead_by_size")

FACTORS_DIR = ROOT / "data" / "factors"
SURPRISE_PANEL = FACTORS_DIR / "factors_earnings_surprise.parquet"
PRICES_PATH = FACTORS_DIR / "prices.parquet"
MARKET_CAPS_PATH = ROOT / "data" / "market_caps" / "historical_market_caps.parquet"
QUALITY_DIR = ROOT / "data" / "quality"

SIZE_LABELS = ("small", "mid", "large")
MIN_EVENTS_PER_BUCKET = 500


def attach_size_tercile(events: pd.DataFrame, market_cap: pd.Series) -> pd.DataFrame:
    """
    Label each announcement with its size tercile, ranked within its own quarter.

    Ranking within the quarter matters: absolute market caps grew by an order of
    magnitude over the sample, so a full-sample breakpoint would classify most
    pre-2005 events as "small" purely through inflation.

    Args:
        events: ``symbol``, ``event_date``, ``signal`` rows.
        market_cap: ``(date, symbol)`` market caps.

    Returns:
        ``events`` plus a ``size`` column (``small``/``mid``/``large``); events
        with no market cap on their announcement date are dropped.
    """
    keys = pd.MultiIndex.from_arrays(
        [events["event_date"], events["symbol"]], names=["date", "symbol"]
    )
    caps = market_cap.reindex(keys).to_numpy()
    labeled = events.assign(market_cap=caps).dropna(subset=["market_cap"])
    if labeled.empty:
        return labeled.assign(size=pd.Series(dtype="object"))

    quarter = labeled["event_date"].dt.tz_localize(None).dt.to_period("Q")
    tercile = labeled.groupby(quarter)["market_cap"].transform(
        lambda values: (
            pd.qcut(values.rank(method="first"), 3, labels=list(SIZE_LABELS))
            if len(values) >= 3
            else pd.Series(np.nan, index=values.index)
        )
    )
    labeled = labeled.assign(size=tercile).dropna(subset=["size"])
    logger.info(
        "Sized %s of %s events: %s",
        f"{len(labeled):,}",
        f"{len(events):,}",
        labeled["size"].value_counts().to_dict(),
    )
    return labeled


def summarize(result: dict, label: str, n_events: int) -> dict:
    """Extract the reportable numbers from one event-study run."""
    car = result["car_paths"]
    spread = result["spread_path"]
    summary = {
        "bucket": label,
        "n_events": n_events,
        "event_counts_by_quantile": result["event_counts"],
        "spread_final_pct": round(float(spread.iloc[-1]) * 100, 3),
        "spread_day20_pct": round(float(spread.loc[20]) * 100, 3) if 20 in spread.index else None,
        "spread_t_stat": round(float(result["spread_t_stat"]), 2),
        "top_quantile_final_pct": round(float(car.iloc[-1, -1]) * 100, 3),
        "bottom_quantile_final_pct": round(float(car.iloc[-1, 0]) * 100, 3),
    }
    logger.info(
        "%-8s n=%7s  spread@20d=%6s%%  spread@end=%6.2f%%  t=%5.2f",
        label,
        f"{n_events:,}",
        summary["spread_day20_pct"],
        summary["spread_final_pct"],
        summary["spread_t_stat"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PEAD by market-cap tercile")
    parser.add_argument("--horizon", type=int, default=60, help="Drift window in trading days")
    parser.add_argument("--quantiles", type=int, default=5, help="Surprise buckets")
    parser.add_argument(
        "--min-price",
        type=float,
        default=1.0,
        help="Price floor for return eligibility; the conventional screen against "
        "sub-dollar stocks whose returns are bid-ask bounce",
    )
    args = parser.parse_args()

    panel = pd.read_parquet(SURPRISE_PANEL, columns=["sue_price_scaled", "days_since_earnings"])
    prices = pd.read_parquet(PRICES_PATH)

    excluded = load_non_operating_symbols()
    events = extract_events_from_surprise_panel(panel)
    events = events[~events["symbol"].isin(excluded)]
    logger.info(
        "%s announcements across %s symbols after excluding %s non-operating names",
        f"{len(events):,}",
        f"{events['symbol'].nunique():,}",
        f"{len(excluded):,}",
    )

    caps = pd.read_parquet(MARKET_CAPS_PATH)["market_cap"]
    caps.index = caps.index.set_names(["date", "symbol"])
    cap_dates = caps.index.get_level_values("date")
    if cap_dates.tz is None and prices.index.tz is not None:
        caps.index = pd.MultiIndex.from_arrays(
            [cap_dates.tz_localize(prices.index.tz), caps.index.get_level_values("symbol")],
            names=["date", "symbol"],
        )

    sized = attach_size_tercile(events, caps)

    results = [
        summarize(
            run_event_study(
                events,
                prices,
                horizon_days=args.horizon,
                n_quantiles=args.quantiles,
                min_price=args.min_price,
            ),
            "all",
            len(events),
        )
    ]
    for label in SIZE_LABELS:
        bucket = sized[sized["size"] == label]
        if len(bucket) < MIN_EVENTS_PER_BUCKET:
            logger.warning("Skipping %s: only %d events", label, len(bucket))
            continue
        results.append(
            summarize(
                run_event_study(
                    bucket,
                    prices,
                    horizon_days=args.horizon,
                    n_quantiles=args.quantiles,
                    min_price=args.min_price,
                ),
                label,
                len(bucket),
            )
        )

    by_bucket = {r["bucket"]: r for r in results}
    small = by_bucket.get("small")
    large = by_bucket.get("large")
    verdict = {
        "literature_prediction": "drift strongest in small caps, weakest in large",
        "small_spread_pct": small["spread_final_pct"] if small else None,
        "large_spread_pct": large["spread_final_pct"] if large else None,
        "small_t_stat": small["spread_t_stat"] if small else None,
        "matches_literature": (
            bool(small["spread_final_pct"] > large["spread_final_pct"]) if small and large else None
        ),
        "small_is_significant": bool(small and abs(small["spread_t_stat"]) >= 2),
        "note": (
            "Gross of costs. Small-cap PEAD is the hardest version to actually trade: "
            "the names with the biggest drift also carry the widest spreads and the "
            "least borrow. A significant spread here is a research finding, not a "
            "strategy — the cost model has to come next."
        ),
    }

    payload = {
        "run_date": str(date.today()),
        "horizon_days": args.horizon,
        "n_quantiles": args.quantiles,
        "universe": (
            "full canonical panel, non-operating vehicles excluded, "
            f"returns require price >= ${args.min_price:.2f} and |ret| <= 300% (bad-print bound)"
        ),
        "size_breakpoints": "market-cap terciles computed within each calendar quarter",
        "results": results,
        "verdict": verdict,
    }
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QUALITY_DIR / f"experiment_pead_by_size_{date.today():%Y%m%d}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", out_path)
    if small and large:
        logger.info(
            "VERDICT: small %.2f%% (t=%.2f) vs large %.2f%% (t=%.2f) — %s literature",
            small["spread_final_pct"],
            small["spread_t_stat"],
            large["spread_final_pct"],
            large["spread_t_stat"],
            "MATCHES" if verdict["matches_literature"] else "CONTRADICTS",
        )


if __name__ == "__main__":
    main()
