#!/usr/bin/env python3
"""
Build the point-in-time fundamentals panel and fundamental factors.

Reads raw FMP statements (data/raw/fmp/fundamentals/), aligns every metric on
its publication date (acceptedDate; visible the NEXT trading day), and writes:

    data/factors/fundamentals.parquet         PIT metric panel (date, symbol)
    data/factors/factors_fundamental.parquet  book_to_market, earnings_yield,
                                              roe, asset_growth

Market caps come from data/market_caps/historical_market_caps.parquet.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build_fundamentals_panel.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.factors.fundamentals import (
    build_pit_fundamentals_panel,
    compute_fundamental_factors,
    extract_pit_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_fundamentals_panel")

RAW_DIR = ROOT / "data" / "raw" / "fmp" / "fundamentals"
PRICES_PATH = ROOT / "data" / "factors" / "prices.parquet"
MARKET_CAPS_PATH = ROOT / "data" / "market_caps" / "historical_market_caps.parquet"
PIT_PANEL_OUT = ROOT / "data" / "factors" / "fundamentals.parquet"
FACTORS_OUT = ROOT / "data" / "factors" / "factors_fundamental.parquet"


def main() -> None:
    trading_index = pd.read_parquet(PRICES_PATH).index

    income_dir = RAW_DIR / "income_statement"
    balance_dir = RAW_DIR / "balance_sheet"
    symbols = sorted(p.stem for p in income_dir.glob("*.parquet"))
    logger.info("Extracting PIT metrics for %d symbols", len(symbols))

    per_symbol_metrics = {}
    n_empty = 0
    for symbol in symbols:
        balance_path = balance_dir / f"{symbol}.parquet"
        if not balance_path.exists():
            continue
        income_statements = pd.read_parquet(income_dir / f"{symbol}.parquet")
        balance_sheets = pd.read_parquet(balance_path)
        metrics = extract_pit_metrics(income_statements, balance_sheets)
        if metrics.empty:
            n_empty += 1
            continue
        per_symbol_metrics[symbol] = metrics

    logger.info(
        "Building PIT panel (%d symbols with data, %d empty)", len(per_symbol_metrics), n_empty
    )
    pit_panel = build_pit_fundamentals_panel(per_symbol_metrics, trading_index)
    pit_panel.to_parquet(PIT_PANEL_OUT)
    logger.info("Wrote %s: %s", PIT_PANEL_OUT.name, pit_panel.shape)

    market_caps = pd.read_parquet(MARKET_CAPS_PATH)
    market_cap = market_caps["market_cap"]
    market_cap.index = market_cap.index.set_names(["date", "symbol"])
    if market_cap.index.get_level_values("date").tz is None and trading_index.tz is not None:
        dates = market_cap.index.get_level_values("date").tz_localize(trading_index.tz)
        market_cap.index = pd.MultiIndex.from_arrays(
            [dates, market_cap.index.get_level_values("symbol")], names=["date", "symbol"]
        )

    fundamental_factors = compute_fundamental_factors(pit_panel, market_cap)
    fundamental_factors.to_parquet(FACTORS_OUT)
    coverage = fundamental_factors.notna().mean()
    logger.info(
        "Wrote %s: %s; non-NaN coverage: %s",
        FACTORS_OUT.name,
        fundamental_factors.shape,
        coverage.round(3).to_dict(),
    )


if __name__ == "__main__":
    main()
