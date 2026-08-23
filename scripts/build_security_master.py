#!/usr/bin/env python3
"""
Build (or extend) the security master: permanent ids for every entity we store.

Safe to re-run. Existing qids are always preserved; only genuinely new entities
get new ids. See :mod:`core.data.security_master` for why the id is opaque and
why CIK is the anchor.

Network use is optional and off by default: ``--fetch-cik`` pulls SEC's public
ticker→CIK map (one request, no key required) so that companies which change
ticker keep their identity. Without it the master still builds, matching on
normalized ticker alone, and marks those rows ``id_source="symbol"``.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build_security_master.py
    /opt/anaconda3/envs/quant/bin/python scripts/build_security_master.py --fetch-cik
    /opt/anaconda3/envs/quant/bin/python scripts/build_security_master.py --dry-run
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

from core.data.artifacts import write_artifact  # noqa: E402
from core.data.security_master import (  # noqa: E402
    ALIAS_FILE,
    MASTER_FILE,
    build_security_master,
    load_security_master,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_security_master")

UNIVERSE_FILE = ROOT / "data" / "raw" / "fmp" / "universe" / "us_equity_universe.parquet"


def load_cik_map() -> dict[str, str]:
    """Ticker -> CIK from SEC. Failure is non-fatal: the master degrades, not fails."""
    from core.data.sec.client import fetch_company_tickers

    try:
        mapping = fetch_company_tickers()
        logger.info("Fetched %d ticker->CIK mappings from SEC", len(mapping))
        return mapping
    except Exception as exc:  # noqa: BLE001 - an optional enrichment must not break the build
        logger.warning("CIK fetch failed (%s); building with symbol matching only", exc)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the security master")
    parser.add_argument(
        "--fetch-cik",
        action="store_true",
        help="Fetch SEC ticker->CIK so re-tickered companies keep their qid",
    )
    parser.add_argument("--universe-file", type=Path, default=UNIVERSE_FILE)
    parser.add_argument("--dry-run", action="store_true", help="Report, write nothing")
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Discard existing ids and mint from scratch. DESTRUCTIVE: any stored "
            "reference to a qid becomes wrong. Valid only while nothing has "
            "consumed the master yet."
        ),
    )
    args = parser.parse_args()

    if not args.universe_file.exists():
        logger.error("No universe file at %s", args.universe_file)
        return 1

    universe = pd.read_parquet(args.universe_file)
    logger.info("Universe: %d symbols", len(universe))

    existing = load_security_master()
    if args.reset and not existing.securities.empty:
        logger.warning(
            "--reset: discarding %d existing ids. Every stored qid reference is " "now invalid.",
            len(existing.securities),
        )
        existing = None
    elif not existing.securities.empty:
        logger.info(
            "Existing master: %d securities, next id %s",
            len(existing.securities),
            existing.next_id,
        )

    cik_map = load_cik_map() if args.fetch_cik else {}
    master = build_security_master(universe, existing=existing, cik_by_symbol=cik_map)

    matched = int((master.securities["id_source"] == "cik").sum())
    logger.info(
        "Result: %d securities, %d anchored to a CIK (%.1f%%), %d aliases",
        len(master.securities),
        matched,
        100.0 * matched / max(len(master.securities), 1),
        len(master.aliases),
    )

    if args.dry_run:
        logger.info("Dry run: nothing written")
        return 0

    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(master.securities, MASTER_FILE, label="security_master")
    write_artifact(master.aliases, ALIAS_FILE, label="symbol_aliases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
