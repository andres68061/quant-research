#!/usr/bin/env python3
"""
Build the earnings-surprise (PEAD) panel and the point-in-time vendor-metric panel.

Two artifacts, both derived from datasets already downloaded — no network access:

    data/factors/factors_earnings_surprise.parquet   SUE + days_since_earnings
    data/factors/factors_vendor_metrics.parquet      ~42 vendor ratios, PIT-dated

**Why these are separate from the fundamentals panel.** They have different
visibility lifetimes. A balance-sheet item stays valid until the next filing (273
trading days). An earnings surprise is an *event* signal: the drift it predicts
runs about one quarter, so it is held 60 trading days and then dies rather than
forward-filling into the next announcement.

The vendor-metric panel is the payoff from the filing-date join in
``core.data.factors.vendor_metrics`` — those datasets are stamped with fiscal
period ends and are unusable until joined to real publication dates (ADR 0010).

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build_event_factors.py
    /opt/anaconda3/envs/quant/bin/python scripts/build_event_factors.py --symbols AAPL,MSFT
    /opt/anaconda3/envs/quant/bin/python scripts/build_event_factors.py --only earnings
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.artifacts import streamed_artifact
from core.data.factors.earnings_surprise import (
    DRIFT_WINDOW_TRADING_DAYS,
    add_days_since_announcement,
    compute_announcement_surprises,
)
from core.data.factors.fundamentals import (
    MAX_STALENESS_TRADING_DAYS,
    build_pit_fundamentals_panel,
)
from core.data.factors.vendor_metrics import (
    VENDOR_DATASET_FIELDS,
    build_symbol_vendor_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_event_factors")

RAW_FMP_DIR = ROOT / "data" / "raw" / "fmp"
PRICES_PATH = ROOT / "data" / "factors" / "prices.parquet"
INCOME_DIR = RAW_FMP_DIR / "fundamentals" / "income_statement"
EARNINGS_OUT = ROOT / "data" / "factors" / "factors_earnings_surprise.parquet"
VENDOR_OUT = ROOT / "data" / "factors" / "factors_vendor_metrics.parquet"

PANEL_DTYPE = "float32"
DEFAULT_BATCH_SIZE = 600


def _read_if_exists(path: Path) -> pd.DataFrame:
    """Read a raw parquet, returning an empty frame when the symbol has no file."""
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def build_earnings_panel(symbols: list[str], prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build the dailyized earnings-surprise panel.

    Args:
        symbols: Symbols to process.
        prices: Wide close panel; supplies both the per-symbol price used to scale
            the surprise and the target trading calendar.

    Returns:
        MultiIndex (date, symbol) panel of surprise factors plus
        ``days_since_earnings``.
    """
    earnings_dir = RAW_FMP_DIR / "earnings"
    per_symbol: dict[str, pd.DataFrame] = {}
    n_missing = 0

    for symbol in symbols:
        earnings = _read_if_exists(earnings_dir / f"{symbol}.parquet")
        if earnings.empty or symbol not in prices.columns:
            n_missing += 1
            continue
        surprises = compute_announcement_surprises(earnings, prices[symbol])
        if not surprises.empty:
            per_symbol[symbol] = surprises.astype(PANEL_DTYPE)

    logger.info("Earnings surprises for %d symbols (%d without data)", len(per_symbol), n_missing)
    if not per_symbol:
        return pd.DataFrame()

    # Drift window, not the fundamentals staleness cap: the signal is an event.
    panel = build_pit_fundamentals_panel(
        per_symbol, prices.index, max_staleness_days=DRIFT_WINDOW_TRADING_DAYS
    )
    return add_days_since_announcement(panel)


def build_vendor_panel(symbols: list[str], trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build the dailyized, publication-dated vendor-metric panel.

    Args:
        symbols: Symbols to process.
        trading_index: Target trading calendar.

    Returns:
        MultiIndex (date, symbol) panel with :data:`VENDOR_METRIC_COLUMNS`.
    """
    per_symbol: dict[str, pd.DataFrame] = {}
    n_undated = 0

    for symbol in symbols:
        statements = _read_if_exists(INCOME_DIR / f"{symbol}.parquet")
        if statements.empty:
            continue
        vendor_frames = {
            dataset: _read_if_exists(RAW_FMP_DIR / dataset / f"{symbol}.parquet")
            for dataset in VENDOR_DATASET_FIELDS
        }
        if all(frame.empty for frame in vendor_frames.values()):
            continue

        metrics = build_symbol_vendor_metrics(vendor_frames, statements)
        if metrics.empty:
            n_undated += 1
            continue
        per_symbol[symbol] = metrics.astype(PANEL_DTYPE)

    logger.info(
        "Vendor metrics for %d symbols (%d had rows but no matching filing dates)",
        len(per_symbol),
        n_undated,
    )
    if not per_symbol:
        return pd.DataFrame()

    return build_pit_fundamentals_panel(
        per_symbol, trading_index, max_staleness_days=MAX_STALENESS_TRADING_DAYS
    )


def write_in_batches(
    symbols: list[str],
    prices: pd.DataFrame,
    builder: "Callable[[list[str], pd.DataFrame], pd.DataFrame]",
    out_path: Path,
    batch_size: int,
    label: str,
) -> int:
    """
    Build a dailyized panel batch by batch, appending each batch to parquet.

    Dailyizing reindexes every symbol onto the full trading calendar before the
    all-NaN rows are dropped, so a single-pass build at full universe allocates
    (symbols x dates x columns) — ~15 GB for the 42-column vendor panel at 8,900
    symbols. Batching caps the intermediate at one batch.

    Args:
        symbols: Universe to build.
        prices: Canonical price panel (calendar + per-symbol closes).
        builder: ``(symbols, prices) -> dailyized panel`` for one batch.
        out_path: Destination parquet.
        batch_size: Symbols per batch.
        label: Name used in progress logs.

    Returns:
        Total rows written.
    """
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            panel = builder(batch, prices)
            if panel.empty:
                continue
            table = pa.Table.from_pandas(panel)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            total += len(panel)
            logger.info(
                "%s [%d/%d symbols] +%s rows (total %s)",
                label,
                min(start + batch_size, len(symbols)),
                len(symbols),
                f"{len(panel):,}",
                f"{total:,}",
            )
    finally:
        if writer is not None:
            writer.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build earnings-surprise and vendor-metric panels")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated subset")
    parser.add_argument(
        "--only", type=str, default=None, choices=["earnings", "vendor"], help="Build just one"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    prices = pd.read_parquet(PRICES_PATH)
    # A --symbols run is a partial build. streamed_artifact redirects it to a
    # scratch path so it cannot replace the full panel, which is exactly how a
    # three-symbol smoke test once destroyed 23M rows.
    subset = args.symbols is not None
    symbols = (
        args.symbols.split(",")
        if args.symbols
        else sorted(path.stem for path in INCOME_DIR.glob("*.parquet"))
    )
    logger.info("Processing %d symbols (batch size %d)", len(symbols), args.batch_size)

    if args.only != "vendor":
        with streamed_artifact(EARNINGS_OUT, subset=subset, label="earnings") as temp_path:
            rows = write_in_batches(
                symbols, prices, build_earnings_panel, temp_path, args.batch_size, "earnings"
            )
        if rows == 0:
            logger.warning("No earnings surprises built; is data/raw/fmp/earnings populated?")

    if args.only != "earnings":
        with streamed_artifact(VENDOR_OUT, subset=subset, label="vendor") as temp_path:
            rows = write_in_batches(
                symbols,
                prices,
                lambda batch, panel: build_vendor_panel(batch, panel.index),
                temp_path,
                args.batch_size,
                "vendor",
            )
        if rows == 0:
            logger.warning("No vendor metrics built; are key_metrics/ratios populated?")


if __name__ == "__main__":
    main()
