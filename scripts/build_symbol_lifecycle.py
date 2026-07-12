#!/usr/bin/env python3
"""
Build symbol lifecycle windows and optionally truncate the derived price panel.

Fetches FMP delisted + symbol-change registries, combines with price-gap
detection, writes data/quality/symbol_lifecycle.parquet, and (with --apply)
NaNs out prices outside each window in data/factors/prices.parquet (raw layer
untouched). After --apply, rebuild factors.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build_symbol_lifecycle.py
    /opt/anaconda3/envs/quant/bin/python scripts/build_symbol_lifecycle.py --apply
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

from core.data.lifecycle import (
    LIFECYCLE_PATH,
    apply_lifecycle_to_panel,
    build_lifecycle_windows,
    fetch_all_delisted,
    fetch_all_symbol_changes,
    write_lifecycle_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_symbol_lifecycle")

PRICES_PATH = ROOT / "data" / "factors" / "prices.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/apply symbol lifecycle windows")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="NaN prices outside windows in the derived panel and note factor rebuild",
    )
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help="Re-apply existing lifecycle parquet to the price panel (no FMP refetch)",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(PRICES_PATH)
    lifecycle_path = ROOT / LIFECYCLE_PATH

    if args.apply_only:
        if not lifecycle_path.exists():
            logger.error("No lifecycle file at %s; run without --apply-only first", lifecycle_path)
            sys.exit(1)
        windows = pd.read_parquet(lifecycle_path)
        truncated, n_cleared = apply_lifecycle_to_panel(prices, windows)
        truncated.to_parquet(PRICES_PATH)
        logger.info("Re-applied lifecycle: cleared %d price cells", n_cleared)
        return

    logger.info("Fetching FMP symbol-change + delisted registries...")
    changes = fetch_all_symbol_changes()
    delisted = fetch_all_delisted()
    logger.info("symbol-change rows=%d delisted rows=%d", len(changes), len(delisted))

    windows = build_lifecycle_windows(prices, changes, delisted)
    write_lifecycle_windows(windows, lifecycle_path)

    truncated_notes = windows[windows["source_notes"].str.contains("price_gap|symbol_change|delistedDate")]
    logger.info(
        "Windows built for %d symbols; %d have a truncation reason beyond price_span",
        len(windows),
        len(truncated_notes),
    )
    if not truncated_notes.empty:
        sample = truncated_notes.head(15)
        for row in sample.itertuples():
            logger.info(
                "  %s  %s → %s  [%s]",
                row.symbol,
                pd.Timestamp(row.valid_from).date(),
                pd.Timestamp(row.valid_to).date(),
                row.source_notes,
            )

    if not args.apply:
        logger.info("Dry run only. Re-run with --apply to truncate the derived price panel.")
        return

    truncated, n_cleared = apply_lifecycle_to_panel(prices, windows)
    truncated.to_parquet(PRICES_PATH)
    logger.info(
        "Applied lifecycle truncation: cleared %d price cells. Rebuild factors next "
        "(update_daily rebuild, or build_price_factors).",
        n_cleared,
    )


if __name__ == "__main__":
    main()
