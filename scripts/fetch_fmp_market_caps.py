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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_fmp_market_caps")

RAW_DIR = ROOT / "data" / "raw" / "fmp" / "market_caps"
PRICES_PANEL = ROOT / "data" / "factors" / "prices.parquet"
PANEL_OUT = ROOT / "data" / "market_caps" / "historical_market_caps.parquet"


def load_universe() -> list[str]:
    columns = pd.read_parquet(PRICES_PANEL).columns
    return sorted(c for c in columns if not c.startswith("^"))


def fetch_all(symbols: list[str], refresh: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now().normalize()
    n_done = n_skipped = n_empty = n_failed = 0
    for i, symbol in enumerate(symbols, 1):
        out_path = RAW_DIR / f"{symbol}.parquet"
        if out_path.exists() and not refresh:
            n_skipped += 1
            continue
        try:
            history = fetch_historical_market_cap(symbol, DEFAULT_START, end)
        except Exception:
            logger.exception("FAILED %s", symbol)
            n_failed += 1
            continue
        history.to_parquet(out_path)
        if history.empty:
            n_empty += 1
        else:
            n_done += 1
        if i % 25 == 0 or i == len(symbols):
            logger.info(
                "[%d/%d] ok=%d skipped=%d empty=%d failed=%d (last=%s)",
                i,
                len(symbols),
                n_done,
                n_skipped,
                n_empty,
                n_failed,
                symbol,
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
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--build-only", action="store_true", help="Skip fetch; rebuild panel only")
    args = parser.parse_args()

    if not args.build_only:
        symbols = args.symbols.split(",") if args.symbols else load_universe()
        fetch_all(symbols, refresh=args.refresh)
    write_panel()


if __name__ == "__main__":
    main()
