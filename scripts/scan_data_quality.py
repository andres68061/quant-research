#!/usr/bin/env python3
"""
Run the data-quality scanner over the price panel and update the quarantine list.

Compares the live FMP panel against the pre-cutover yfinance backup (if
present) for entity-mismatch detection, then persists
``data/quality/quarantine.parquet`` (+ CSV). Manual ``cleared`` statuses in the
existing list survive rescans.

Run after every data update (scripts/update_daily.py invokes this automatically).

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/scan_data_quality.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.quality import (
    QUARANTINE_PATH,
    load_quarantine_list,
    merge_with_existing,
    scan_price_panel,
    write_quarantine_list,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("scan_data_quality")

PRICES_PATH = ROOT / "data" / "factors" / "prices.parquet"


def find_reference_panel() -> pd.DataFrame | None:
    """Latest pre-FMP-cutover backup, used as the second vendor for entity checks."""
    backups = sorted((ROOT / "data" / "factors").glob("prices_backup_*_pre_fmp_cutover.parquet"))
    if not backups:
        logger.warning("No pre-cutover backup found; skipping entity-mismatch check")
        return None
    logger.info("Reference panel: %s", backups[-1].name)
    return pd.read_parquet(backups[-1])


def main() -> None:
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Scanning %d symbols x %d dates", prices.shape[1], len(prices))

    findings = scan_price_panel(prices, reference_prices=find_reference_panel())
    merged = merge_with_existing(findings, load_quarantine_list())
    write_quarantine_list(merged, ROOT / QUARANTINE_PATH)

    by_status = merged.groupby("status")["symbol"].nunique() if not merged.empty else {}
    logger.info("Scan complete. Symbols by status: %s", dict(by_status))
    quarantined = sorted(set(merged.loc[merged["status"] == "quarantined", "symbol"]))
    logger.info("Quarantined (%d): %s", len(quarantined), ", ".join(quarantined))


if __name__ == "__main__":
    main()
