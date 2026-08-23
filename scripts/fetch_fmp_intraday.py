#!/usr/bin/env python3
"""
Download FMP intraday bars into the raw layer, checkpointed per symbol-year.

Layout:

    data/raw/fmp/intraday/{interval}/{SYMBOL}/{YEAR}.parquet

**Why per-year files.** A single symbol's 1-minute history costs ~85 calls per
year. Writing one file per symbol would mean losing an hour of work to one
interruption; per-symbol-year checkpoints cap the loss at a few minutes and make
the run trivially resumable.

**Cost.** Read this before starting a wide pull — intraday is the most expensive
thing on the plan:

    interval  calls/symbol/year   500 symbols x 10 years
    1min      ~85                 ~425,000  (~14 h)
    5min      ~45                 ~225,000  (~7.5 h)
    1hour     ~5                  ~25,000   (~50 min)
    4hour     ~3                  ~15,000   (~30 min)

Default scope is deliberately narrow: ``1hour`` for the current S&P 500. Widen it
on purpose, not by accident.

Bars are stored exactly as the vendor returns them: **split-adjusted as of the
fetch date, never dividend-adjusted**. Read via
``core.data.fmp.intraday.load_intraday_bars``, which detects and repairs splits
that occurred after the fetch (a blanket re-adjustment would double-adjust).

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_intraday.py \
        --interval 1hour --years 10
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_intraday.py \
        --interval 1min --symbols AAPL,MSFT,SPY --years 5
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_intraday.py --estimate-only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.constituents import fetch_current_sp500
from core.data.fmp.intraday import (
    INTRADAY_INTERVALS,
    fetch_intraday_history,
    generate_intraday_chunks,
)
from core.data.fmp.storage import load_universe_symbols, safe_filename, write_atomic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_fmp_intraday")

RAW_INTRADAY_DIR = ROOT / "data" / "raw" / "fmp" / "intraday"
DEFAULT_INTERVAL = "1hour"
DEFAULT_YEARS = 10
_CALLS_PER_MINUTE = 500.0


def default_universe() -> list[str]:
    """Current S&P 500 members — a defensible default scope for a costly pull."""
    return sorted(fetch_current_sp500())


def estimate_calls(
    symbols: list[str], interval: str, start: pd.Timestamp, end: pd.Timestamp
) -> int:
    """Count the chunks a full run would request, for a cost warning before starting."""
    spec = INTRADAY_INTERVALS[interval]
    per_symbol = len(generate_intraday_chunks(start, end, spec.chunk_days))
    return per_symbol * len(symbols)


def fetch_all(
    symbols: list[str],
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    refresh: bool,
) -> None:
    """
    Fetch intraday bars year by year, skipping symbol-years already on disk.

    Args:
        symbols: Tickers to fetch.
        interval: Key into :data:`core.data.fmp.intraday.INTRADAY_INTERVALS`.
        start: First date (inclusive).
        end: Last date (inclusive).
        refresh: Refetch even when the symbol-year file exists.
    """
    interval_dir = RAW_INTRADAY_DIR / interval
    years = list(range(start.year, end.year + 1))
    counts = {"ok": 0, "skipped": 0, "empty": 0, "failed": 0}
    started = time.monotonic()
    total_units = len(symbols) * len(years)
    unit = 0

    for symbol in symbols:
        symbol_dir = interval_dir / safe_filename(symbol)
        for year in years:
            unit += 1
            out_path = symbol_dir / f"{year}.parquet"
            if out_path.exists() and not refresh:
                counts["skipped"] += 1
                continue

            year_start = max(start, pd.Timestamp(f"{year}-01-01"))
            year_end = min(end, pd.Timestamp(f"{year}-12-31"))
            try:
                bars = fetch_intraday_history(symbol, interval, year_start, year_end)
            except Exception:
                logger.exception("FAILED %s %s %d", symbol, interval, year)
                counts["failed"] += 1
                continue

            symbol_dir.mkdir(parents=True, exist_ok=True)
            write_atomic(bars, out_path)
            counts["empty" if bars.empty else "ok"] += 1

        attempted = counts["ok"] + counts["empty"] + counts["failed"]
        if attempted and unit % 50 < len(years):
            rate = attempted / max(time.monotonic() - started, 1e-9)
            remaining = (total_units - unit) / rate / 60.0 if rate > 0 else float("nan")
            logger.info(
                "%s [%d/%d symbol-years] %s (~%.0f min left)",
                interval,
                unit,
                total_units,
                counts,
                remaining,
            )

    logger.info("%s DONE %s", interval, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FMP intraday bars")
    parser.add_argument(
        "--interval", type=str, default=DEFAULT_INTERVAL, choices=sorted(INTRADAY_INTERVALS)
    )
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated subset")
    parser.add_argument("--universe-file", type=Path, default=None, help="Universe parquet")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="Years of history")
    parser.add_argument("--refresh", action="store_true", help="Refetch existing symbol-years")
    parser.add_argument("--estimate-only", action="store_true", help="Print cost and exit")
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")
    args = parser.parse_args()

    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(years=args.years)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.universe_file:
        symbols = load_universe_symbols(args.universe_file)
    else:
        symbols = default_universe()

    calls = estimate_calls(symbols, args.interval, start, end)
    hours = calls / _CALLS_PER_MINUTE / 60.0
    logger.info(
        "Scope: %d symbols x %s x %d years = ~%s calls (~%.1f h at %.0f calls/min)",
        len(symbols),
        args.interval,
        args.years,
        f"{calls:,}",
        hours,
        _CALLS_PER_MINUTE,
    )
    if args.estimate_only:
        return

    if hours > 2.0 and not args.yes:
        raise SystemExit(
            f"This run would take ~{hours:.1f} hours. Re-run with --yes to confirm, "
            "or narrow --symbols / --years / --interval."
        )

    fetch_all(symbols, args.interval, start, end, refresh=args.refresh)


if __name__ == "__main__":
    main()
