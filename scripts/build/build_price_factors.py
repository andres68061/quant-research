#!/usr/bin/env python3
"""
Rebuild the price-derived factor panel from the canonical price panel, in batches.

Replaces the price-factor portion of ``backfill_all.py`` (which also refetches
prices and is scoped to the S&P universe — wrong on both counts after the ADR
0013 cutover).

**Why batched.** ``build_price_factors`` builds one Series per (symbol, factor)
and concatenates them into a single wide frame before stacking. At 8,910 symbols
x 7 factors x 10,483 dates that intermediate is ~5 GB and the process is killed —
the same failure that stopped the first expanded fundamentals build. Batching
caps the intermediate at one batch and appends each result to the parquet.

**Why dropna.** The stacked long panel is (dates x symbols) rows regardless of
whether a symbol traded — 93M rows at full universe, of which only ~27M carry
data. Rows that are entirely NaN describe nothing; dropping them cuts the
artifact ~3.5x with no information loss.

Outputs:
    data/factors/factors_price.parquet   momentum, reversal, 52w, vol, beta
    data/factors/factors_all.parquet     the above + log_market_cap

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build/build_price_factors.py
    /opt/anaconda3/envs/quant/bin/python scripts/build/build_price_factors.py --batch-size 500
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.factors.build_factors import build_price_factors, load_market_cap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_price_factors")

FACTORS_DIR = ROOT / "data" / "factors"
PRICES_PATH = FACTORS_DIR / "prices.parquet"
MARKET_CAPS_PATH = ROOT / "data" / "market_caps" / "historical_market_caps.parquet"
FACTORS_PRICE_OUT = FACTORS_DIR / "factors_price.parquet"
FACTORS_ALL_OUT = FACTORS_DIR / "factors_all.parquet"

MARKET_SYMBOL = "^GSPC"
DEFAULT_BATCH_SIZE = 600
PANEL_DTYPE = "float32"


def build_batched(
    prices: pd.DataFrame,
    market_cap: pd.DataFrame | None,
    batch_size: int,
) -> tuple[int, int]:
    """
    Build both factor artifacts batch by batch, appending to parquet as we go.

    Args:
        prices: Canonical wide close panel.
        market_cap: Long (date, symbol) market caps, or None to skip
            ``log_market_cap`` (``factors_all`` then mirrors ``factors_price``).
        batch_size: Symbols per batch.

    Returns:
        ``(rows_written, symbols_processed)``.
    """
    symbols = [s for s in prices.columns if s != MARKET_SYMBOL]
    market_close = prices[MARKET_SYMBOL] if MARKET_SYMBOL in prices.columns else None
    if market_close is None:
        logger.warning("%s absent from the panel; beta_60d will be omitted", MARKET_SYMBOL)

    price_writer: pq.ParquetWriter | None = None
    all_writer: pq.ParquetWriter | None = None
    rows_written = 0

    try:
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            columns = batch + ([MARKET_SYMBOL] if market_close is not None else [])
            batch_factors = build_price_factors(prices[columns], market_symbol=MARKET_SYMBOL)

            # Drop the market symbol's own rows and all-NaN (never-traded) rows.
            batch_factors = batch_factors[
                batch_factors.index.get_level_values("symbol") != MARKET_SYMBOL
            ].dropna(how="all")
            if batch_factors.empty:
                continue
            batch_factors = batch_factors.astype(PANEL_DTYPE)

            batch_all = batch_factors
            if market_cap is not None:
                aligned = market_cap["log_market_cap"].reindex(batch_factors.index)
                batch_all = batch_factors.assign(log_market_cap=aligned.astype(PANEL_DTYPE))

            price_table = pa.Table.from_pandas(batch_factors)
            all_table = pa.Table.from_pandas(batch_all)
            if price_writer is None:
                price_writer = pq.ParquetWriter(FACTORS_PRICE_OUT, price_table.schema)
                all_writer = pq.ParquetWriter(FACTORS_ALL_OUT, all_table.schema)
            price_writer.write_table(price_table)
            all_writer.write_table(all_table)

            rows_written += len(batch_factors)
            logger.info(
                "[%d/%d symbols] +%s rows (total %s)",
                min(start + batch_size, len(symbols)),
                len(symbols),
                f"{len(batch_factors):,}",
                f"{rows_written:,}",
            )
    finally:
        if price_writer is not None:
            price_writer.close()
        if all_writer is not None:
            all_writer.close()

    return rows_written, len(symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild price factors from the canonical panel")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Canonical panel: %d dates x %d symbols", prices.shape[0], prices.shape[1])

    market_cap = None
    if MARKET_CAPS_PATH.exists():
        caps = load_market_cap(MARKET_CAPS_PATH)
        if caps is not None:
            caps.index = caps.index.set_names(["date", "symbol"])
            dates = caps.index.get_level_values("date")
            if dates.tz is None and prices.index.tz is not None:
                caps.index = pd.MultiIndex.from_arrays(
                    [dates.tz_localize(prices.index.tz), caps.index.get_level_values("symbol")],
                    names=["date", "symbol"],
                )
            market_cap = caps
            logger.info("Market caps: %s rows", f"{len(caps):,}")
    else:
        logger.warning("No market caps at %s; log_market_cap omitted", MARKET_CAPS_PATH)

    rows, symbols = build_batched(prices, market_cap, args.batch_size)
    logger.info(
        "Wrote %s and %s: %s rows across %d symbols",
        FACTORS_PRICE_OUT.name,
        FACTORS_ALL_OUT.name,
        f"{rows:,}",
        symbols,
    )


if __name__ == "__main__":
    main()
