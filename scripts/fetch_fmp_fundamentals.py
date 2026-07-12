#!/usr/bin/env python3
"""
Download quarterly financial statements from FMP into the raw layer.

Raw layout (one file per symbol per statement, vendor fields verbatim):

    data/raw/fmp/fundamentals/income_statement/{SYMBOL}.parquet
    data/raw/fmp/fundamentals/balance_sheet/{SYMBOL}.parquet
    data/raw/fmp/fundamentals/cash_flow/{SYMBOL}.parquet

Resumable like fetch_fmp_prices.py: existing files are skipped unless
--refresh. Empty files are written for symbols without coverage so reruns
don't re-probe. Universe = columns of the price panel.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_fundamentals.py
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_fundamentals.py --symbols AAPL,GE
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.fundamentals import STATEMENT_ENDPOINTS, fetch_quarterly_statement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_fmp_fundamentals")

RAW_FUNDAMENTALS_DIR = ROOT / "data" / "raw" / "fmp" / "fundamentals"
PRICES_PANEL = ROOT / "data" / "factors" / "prices.parquet"


def load_universe() -> list[str]:
    """Stock symbols from the price panel (indexes like ^GSPC have no statements)."""
    panel_columns = pd.read_parquet(PRICES_PANEL).columns
    return sorted(c for c in panel_columns if not c.startswith("^"))


def fetch_all(symbols: list[str], refresh: bool) -> None:
    n_done = n_skipped = n_empty = n_failed = 0
    for i, symbol in enumerate(symbols, 1):
        for statement in STATEMENT_ENDPOINTS:
            out_path = RAW_FUNDAMENTALS_DIR / statement / f"{symbol}.parquet"
            if out_path.exists() and not refresh:
                n_skipped += 1
                continue
            try:
                statements = fetch_quarterly_statement(symbol, statement)
            except Exception:
                logger.exception("FAILED %s/%s", symbol, statement)
                n_failed += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            statements.to_parquet(out_path)
            if statements.empty:
                n_empty += 1
            else:
                n_done += 1
        if i % 25 == 0 or i == len(symbols):
            logger.info(
                "[%d/%d] files ok=%d skipped=%d empty=%d failed=%d (last: %s)",
                i,
                len(symbols),
                n_done,
                n_skipped,
                n_empty,
                n_failed,
                symbol,
            )
    logger.info("DONE ok=%d skipped=%d empty=%d failed=%d", n_done, n_skipped, n_empty, n_failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FMP quarterly statements")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated subset")
    parser.add_argument("--refresh", action="store_true", help="Refetch even if file exists")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else load_universe()
    fetch_all(symbols, refresh=args.refresh)


if __name__ == "__main__":
    main()
