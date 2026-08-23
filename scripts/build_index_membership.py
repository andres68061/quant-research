#!/usr/bin/env python3
"""
Build the canonical index-membership labeling table (ADR 0013 condition).

After the panel cutover, "is in the S&P 500" stops being implied by presence in
the price panel and becomes an explicit label:

    data/universe/index_membership.parquet
        symbol      ticker
        index_name  "sp500" (only index tracked today; column exists so more can join)
        valid_from  first date of a continuous membership interval (inclusive)
        valid_to    last date of the interval (inclusive); NaT = still a member

Source: the S&P historical membership CSV via ``core.data.sp500_constituents``
(daily membership sets), compressed into intervals. A symbol that left and
rejoined has multiple rows.

Backtests keep using ``sp500_universe_filter`` (same underlying data); this
table is the queryable/joinable artifact for research, labeling, and the UI.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build_index_membership.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.sp500_constituents import SP500Constituents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_index_membership")

OUT_PATH = ROOT / "data" / "universe" / "index_membership.parquet"


def build_intervals(daily_members: pd.DataFrame) -> pd.DataFrame:
    """
    Compress a date x symbol membership indicator into continuous intervals.

    Args:
        daily_members: Boolean DataFrame, index = membership snapshot dates
            (ascending), columns = symbols; True where the symbol is a member.

    Returns:
        Long DataFrame with ``symbol``, ``index_name``, ``valid_from``,
        ``valid_to`` (NaT while still a member as of the last snapshot).
    """
    records: list[dict] = []
    last_snapshot = daily_members.index[-1]
    for symbol in daily_members.columns:
        member = daily_members[symbol]
        if not member.any():
            continue
        # An interval starts where membership turns on, ends where it turns off.
        changed = member.ne(member.shift(fill_value=False))
        starts = member.index[changed & member]
        ends = member.index[changed & ~member]
        for i, start in enumerate(starts):
            end = ends[i] if i < len(ends) else None
            records.append(
                {
                    "symbol": symbol,
                    "index_name": "sp500",
                    "valid_from": start,
                    # valid_to inclusive = day before the first non-member snapshot.
                    "valid_to": (end - pd.Timedelta(days=1)) if end is not None else pd.NaT,
                }
            )
        # Guard: membership true through the final snapshot stays open (NaT).
        _ = last_snapshot
    return pd.DataFrame(records).sort_values(["symbol", "valid_from"]).reset_index(drop=True)


def main() -> None:
    constituents = SP500Constituents()
    snapshots = constituents.get_constituents_series()
    all_symbols = sorted(constituents.get_ticker_universe())

    indicator = pd.DataFrame(False, index=pd.DatetimeIndex(snapshots.index), columns=all_symbols)
    for date, row in snapshots.iterrows():
        present = [s for s in row["tickers"] if s in indicator.columns]
        indicator.loc[date, present] = True

    intervals = build_intervals(indicator)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_parquet(OUT_PATH, index=False)

    open_now = intervals["valid_to"].isna().sum()
    logger.info(
        "Wrote %s: %d intervals, %d symbols, %d current members, span %s -> %s",
        OUT_PATH,
        len(intervals),
        intervals["symbol"].nunique(),
        open_now,
        intervals["valid_from"].min().date(),
        indicator.index[-1].date(),
    )


if __name__ == "__main__":
    main()
