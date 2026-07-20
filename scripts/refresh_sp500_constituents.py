#!/usr/bin/env python3
"""
Refresh point-in-time S&P 500 membership from FMP and reconcile vs the CSV.

Writes:
  data/raw/fmp/constituents/sp500_changes.parquet     # raw add/remove events
  data/raw/fmp/constituents/sp500_membership.parquet  # reconstructed snapshots
  data/quality/sp500_reconciliation.txt               # human-readable diff

Does NOT silently overwrite the hanshof CSV. Promotion is a manual step after
reviewing the reconciliation report (or automatic only when mean Jaccard ≥ 0.95).

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/refresh_sp500_constituents.py
    /opt/anaconda3/envs/quant/bin/python scripts/refresh_sp500_constituents.py --promote
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.constituents import (
    build_membership_snapshots,
    fetch_current_sp500,
    fetch_sp500_change_events,
    reconcile_membership,
)
from core.data.sp500_constituents import SP500Constituents, resolve_sp500_historical_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("refresh_sp500_constituents")

RAW_DIR = ROOT / "data" / "raw" / "fmp" / "constituents"
REPORT_PATH = ROOT / "data" / "quality" / "sp500_reconciliation.txt"
PROMOTE_JACCARD = 0.95


def snapshots_to_csv_frame(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Convert list-of-tickers snapshots to the hanshof CSV on-disk format."""
    return pd.DataFrame(
        {
            "date": snapshots.index.strftime("%Y-%m-%d"),
            "tickers": snapshots["tickers"].map(lambda t: ",".join(t)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh S&P 500 membership from FMP")
    parser.add_argument(
        "--promote",
        action="store_true",
        help=f"Overwrite hanshof CSV when mean Jaccard ≥ {PROMOTE_JACCARD}",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    events = fetch_sp500_change_events()
    current = fetch_current_sp500()
    events.to_parquet(RAW_DIR / "sp500_changes.parquet")
    snapshots = build_membership_snapshots(events, current)
    snapshots.to_parquet(RAW_DIR / "sp500_membership.parquet")
    logger.info(
        "FMP membership: %d change events → %d snapshots; current size=%d",
        len(events),
        len(snapshots),
        len(current),
    )

    csv_path = resolve_sp500_historical_csv()
    csv_snapshots = SP500Constituents(csv_path=csv_path).load()
    report = reconcile_membership(snapshots, csv_snapshots)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.summary() + "\n")
    logger.info("Reconciliation: %s", report.summary())

    if args.promote:
        if report.mean_jaccard < PROMOTE_JACCARD:
            logger.error(
                "Refusing to promote: mean Jaccard %.3f < %.2f",
                report.mean_jaccard,
                PROMOTE_JACCARD,
            )
            sys.exit(1)
        out_name = (
            f"S&P 500 Historical Components & Changes({date.today().strftime('%m-%d-%Y')}).csv"
        )
        out_path = ROOT / "data" / out_name
        snapshots_to_csv_frame(snapshots).to_csv(out_path, index=False)
        logger.info("Promoted FMP membership to %s", out_path)


if __name__ == "__main__":
    main()
