#!/usr/bin/env python3
"""
Download FMP historical market caps into the raw layer and build the panel.

Raw:  data/raw/fmp/market_caps/{SYMBOL}.parquet
Out:  data/market_caps/historical_market_caps.parquet
      (MultiIndex date, symbol — replaces the yfinance shares×price approximation)

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_market_caps.py
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_market_caps.py --symbols AAPL,MSFT
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_market_caps.py --build-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.market_caps import (
    DEFAULT_START,
    build_market_cap_panel,
    fetch_historical_market_cap,
)
from core.data.fmp.storage import (
    load_fetch_windows,
    load_universe_symbols,
    safe_filename,
    write_atomic,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_fmp_market_caps")

RAW_DIR = ROOT / "data" / "raw" / "fmp" / "market_caps"
PRICES_PANEL = ROOT / "data" / "factors" / "prices.parquet"
PANEL_OUT = ROOT / "data" / "market_caps" / "historical_market_caps.parquet"

_PROGRESS_EVERY = 50


def load_universe() -> list[str]:
    columns = pd.read_parquet(PRICES_PANEL).columns
    return sorted(c for c in columns if not c.startswith("^"))


def fetch_all(
    symbols: list[str],
    refresh: bool,
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> None:
    """
    Fetch each symbol's market-cap history; skip symbols already on disk.

    Args:
        symbols: Tickers to fetch.
        refresh: Refetch even when the raw file exists.
        windows: Optional ``{symbol: (start, end)}`` so short-lived names cost
            fewer chunked calls.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    windows = windows or {}
    end = pd.Timestamp.now().normalize()
    n_done = n_skipped = n_empty = n_failed = 0
    started = time.monotonic()

    for i, symbol in enumerate(symbols, 1):
        out_path = RAW_DIR / f"{safe_filename(symbol)}.parquet"
        if out_path.exists() and not refresh:
            n_skipped += 1
            continue
        symbol_start, symbol_end = windows.get(symbol, (DEFAULT_START, end))
        try:
            history = fetch_historical_market_cap(symbol, symbol_start, symbol_end)
        except Exception:
            logger.exception("FAILED %s", symbol)
            n_failed += 1
            continue
        write_atomic(history, out_path)
        if history.empty:
            n_empty += 1
        else:
            n_done += 1
        if i % _PROGRESS_EVERY == 0 or i == len(symbols):
            attempted = n_done + n_empty + n_failed
            rate = attempted / max(time.monotonic() - started, 1e-9)
            remaining = (len(symbols) - i) / rate / 60.0 if rate > 0 else float("nan")
            logger.info(
                "[%d/%d] ok=%d skipped=%d empty=%d failed=%d (~%.0f min left)",
                i,
                len(symbols),
                n_done,
                n_skipped,
                n_empty,
                n_failed,
                remaining,
            )
    logger.info("DONE ok=%d skipped=%d empty=%d failed=%d", n_done, n_skipped, n_empty, n_failed)


def write_panel() -> None:
    panel = build_market_cap_panel(RAW_DIR)
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    # Keep a legacy `ticker` level name alias for any consumer that still expects it:
    # write with symbol; load_market_cap renames ticker→symbol if needed.
    panel.to_parquet(PANEL_OUT)
    logger.info("Wrote %s: %s", PANEL_OUT, panel.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FMP historical market caps")
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--universe-file", type=Path, default=None, help="Universe parquet")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--build-only", action="store_true", help="Skip fetch; rebuild panel only")
    args = parser.parse_args()

    if not args.build_only:
        if args.symbols:
            symbols = args.symbols.split(",")
        elif args.universe_file:
            symbols = load_universe_symbols(args.universe_file)
        else:
            symbols = load_universe()
        windows = load_fetch_windows(args.universe_file, default_start=DEFAULT_START)
        logger.info("Fetching %d symbols (%d with narrowed windows)", len(symbols), len(windows))
        fetch_all(symbols, refresh=args.refresh, windows=windows)
    write_panel()


if __name__ == "__main__":
    main()
