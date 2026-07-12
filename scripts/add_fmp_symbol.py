#!/usr/bin/env python3
"""
Add one or more FMP symbols to the canonical equity price panel.

Fetches dividend-adjusted history into ``data/raw/fmp/prices/{SYMBOL}.parquet``,
merges the column into ``data/factors/prices.parquet``, and optionally rebuilds
price factors.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/add_fmp_symbol.py --symbols BNY
    /opt/anaconda3/envs/quant/bin/python scripts/add_fmp_symbol.py --symbols BNY --rebuild-factors
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

from core.data.factors.build_factors import build_price_factors, merge_market_cap
from core.data.factors.io import write_parquet
from core.data.fmp.panel import add_symbol_to_fmp_panel
from core.data.sector_classification import add_or_update_sectors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("add_fmp_symbol")

PRICES_PATH = ROOT / "data" / "factors" / "prices.parquet"
FACTORS_PRICE_PATH = ROOT / "data" / "factors" / "factors_price.parquet"
FACTORS_ALL_PATH = ROOT / "data" / "factors" / "factors_all.parquet"
MCAP_PATH = ROOT / "data" / "market_caps" / "historical_market_caps.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Add FMP symbols to the equity price panel")
    parser.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g. BNY")
    parser.add_argument("--rebuild-factors", action="store_true")
    parser.add_argument("--update-sectors", action="store_true", default=True)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not PRICES_PATH.exists():
        raise SystemExit(f"Missing panel: {PRICES_PATH}")

    panel = pd.read_parquet(PRICES_PATH)
    for symbol in symbols:
        panel = add_symbol_to_fmp_panel(panel, symbol)
        if args.update_sectors:
            add_or_update_sectors([symbol], force_refresh=True)

    write_parquet(panel, PRICES_PATH)
    logger.info("Wrote panel %s shape=%s", PRICES_PATH, panel.shape)

    if args.rebuild_factors:
        factors_price = build_price_factors(panel, market_symbol="^GSPC")
        write_parquet(factors_price, FACTORS_PRICE_PATH)
        factors_all = merge_market_cap(factors_price, MCAP_PATH)
        write_parquet(factors_all, FACTORS_ALL_PATH)
        logger.info("Rebuilt factors_price %s and factors_all %s", factors_price.shape, factors_all.shape)


if __name__ == "__main__":
    main()
