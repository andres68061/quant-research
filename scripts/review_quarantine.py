#!/usr/bin/env python3
"""
Record manual review decisions on quarantine findings.

Part of the quarantine workflow: the scanner auto-flags/quarantines
(scripts/scan_data_quality.py), findings show up on the frontend Data Coverage
page, and decisions are recorded here. Decisions survive rescans.

Usage:
    # list open findings
    /opt/anaconda3/envs/quant/bin/python scripts/review_quarantine.py --list

    # clear a finding (keep the data) with a justification
    /opt/anaconda3/envs/quant/bin/python scripts/review_quarantine.py \
        --symbol COST --check entity_mismatch --status cleared \
        --note "Divergence is confined to pre-1994 where yfinance is corrupt; FMP verified vs splits"

    # escalate a flagged finding to quarantined
    /opt/anaconda3/envs/quant/bin/python scripts/review_quarantine.py \
        --symbol XYZ --check extreme_returns --status quarantined --note "confirmed bad prints"

After changing statuses, restart the API (data is filtered at load time).
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.quality import QUARANTINE_PATH, load_quarantine_list, set_review_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("review_quarantine")


def list_findings() -> None:
    quarantine = load_quarantine_list(ROOT / QUARANTINE_PATH)
    if quarantine.empty:
        logger.info("Quarantine list is empty.")
        return
    if "review_note" not in quarantine.columns:
        quarantine["review_note"] = ""
    for status in ("quarantined", "flagged", "cleared"):
        subset = quarantine[quarantine["status"] == status]
        if subset.empty:
            continue
        logger.info("== %s (%d) ==", status.upper(), len(subset))
        for row in subset.itertuples():
            note = f"  [{row.review_note}]" if row.review_note else ""
            logger.info("  %-6s %-16s %s%s", row.symbol, row.check, row.detail, note)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review quarantine findings")
    parser.add_argument("--list", action="store_true", help="List all findings and exit")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--check", type=str)
    parser.add_argument("--status", type=str, choices=["cleared", "quarantined", "flagged"])
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    if args.list:
        list_findings()
        return

    if not (args.symbol and args.check and args.status):
        parser.error("--symbol, --check and --status are required (or use --list)")
    if not args.note:
        parser.error("--note is required: every manual decision needs a justification")

    set_review_status(args.symbol, args.check, args.status, args.note, ROOT / QUARANTINE_PATH)


if __name__ == "__main__":
    main()
