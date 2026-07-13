#!/usr/bin/env python3
"""
Fetch a small SEC EDGAR filings sample for FMP acceptedDate cross-checks.

Writes ``data/raw/sec/filings_sample.parquet``.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_sec_filings_sample.py
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_sec_filings_sample.py --symbols AAPL,MSFT,JPM
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.sec import build_filings_sample, fetch_company_tickers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_sec_filings_sample")

OUT_PATH = ROOT / "data" / "raw" / "sec" / "filings_sample.parquet"
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ", "PG"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SEC EDGAR filings sample")
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated tickers",
    )
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    logger.info("Loading SEC company ticker map...")
    mapping = fetch_company_tickers()
    logger.info("Mapped %d tickers", len(mapping))

    # Gentle pacing for SEC fair-access (≤10 req/s); one pause between symbols.
    frames = []
    for i, symbol in enumerate(symbols):
        if i:
            time.sleep(0.2)
        part = build_filings_sample(
            [symbol],
            ticker_to_cik=mapping,
            limit_per_symbol=12,
        )
        if not part.empty:
            frames.append(part)

    if not frames:
        logger.error("No filings retrieved")
        sys.exit(1)

    import pandas as pd

    out = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d filing rows → %s", len(out), OUT_PATH)


if __name__ == "__main__":
    main()
