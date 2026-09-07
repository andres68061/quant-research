#!/usr/bin/env python3
"""
Check every endpoint spec against the live vendor before a long run uses it.

A spec with a missing required parameter does not fail loudly — the vendor
returns an empty list, the runner records 9,011 honest "empty" rows, and twelve
hours later you have a directory full of empty files that looks like a dataset
with no coverage. ``analyst-estimates`` was exactly this: it needs ``period``,
and without it returns nothing for every symbol.

So each spec is called once with a liquid, long-listed symbol that should have
data for almost everything. An endpoint returning no rows for that symbol is not
proof of a bug, but it is the short list worth reading before spending a day of
calls.

Usage:
    python scripts/validate_fmp_manifest.py
    python scripts/validate_fmp_manifest.py --symbol MSFT --waves 2,3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.transport import make_fetcher
from core.ingest.catalog import build_specs, load_manifest, select_specs
from core.ingest.plan import expand_spec
from core.ingest.ratelimit import TokenBucket
from core.ingest.spec import Partition

logger = logging.getLogger("validate_fmp_manifest")


def main() -> int:
    """Probe one task per spec and report those returning nothing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL", help="probe symbol for per-symbol specs")
    parser.add_argument("--waves", default=None, help="comma-separated waves; default all")
    parser.add_argument("--rate", type=float, default=300.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    specs = build_specs(load_manifest("fmp"))
    waves = {int(w) for w in args.waves.split(",")} if args.waves else None
    specs = select_specs(specs, waves=waves, include_derivable=True)

    fetch = make_fetcher()
    bucket = TokenBucket(args.rate)
    keys = {
        Partition.PER_SYMBOL: [args.symbol],
        Partition.BATCH_SYMBOLS: [args.symbol],
        Partition.PER_CIK: ["0000320193"],
        Partition.PER_SECTOR: ["Technology"],
        Partition.PER_INDUSTRY: ["Semiconductors"],
        Partition.PER_EXCHANGE: ["NASDAQ"],
        Partition.PER_NAME: ["Pelosi"],
    }

    empty: list[str] = []
    errors: list[tuple[str, int, str]] = []
    checked = 0

    for spec in specs:
        tasks = expand_spec(spec, keys)
        if not tasks:
            continue
        task = tasks[0]
        bucket.acquire()
        result = fetch(spec.endpoint, task.params)
        checked += 1
        if result.status_code != 200:
            errors.append((spec.name, result.status_code, (result.error or "")[:80]))
        elif spec.payload.value == "json" and not (result.rows or []):
            empty.append(spec.name)

    logger.info("checked %d specs with symbol=%s", checked, args.symbol)
    if errors:
        logger.info("\n%d specs returned a non-200:", len(errors))
        for name, code, body in errors:
            logger.info("  %-46s HTTP %-4s %s", name, code, body)
    if empty:
        logger.info(
            "\n%d specs returned 200 with no rows — check for a missing required parameter:",
            len(empty),
        )
        for name in empty:
            logger.info("  %s", name)
    if not errors and not empty:
        logger.info("every spec returned rows")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
