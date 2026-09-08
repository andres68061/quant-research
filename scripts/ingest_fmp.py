#!/usr/bin/env python3
"""
Ingest the FMP vendor surface into the raw layer, by wave, resumably.

The endpoint catalog lives in ``config/vendors/fmp.json`` — this script only
resolves partition keys, plans, and runs. Adding an endpoint means editing the
manifest, not this file.

Waves (``--wave``), lowest first:

    1  global reference tables (universe, calendars, constituents, COT, news)
    2  core per-symbol history (statements, ratios, events, analyst, insider)
    3  per-symbol snapshots (quotes, TTM metrics, consensus)
    4  EOD price variants, date-chunked
    5  vendor technical indicators (derivable from prices we already hold)
    6  intraday charts (very large; opt in explicitly)

Completion is a file on disk, so re-running skips what landed and an interrupted
run resumes. Every outcome — success, vendor-empty, failure and why — is written
to the SQLite journal at ``data/quality/ingest_journal.db``.

Usage:
    # cost a run without spending a call
    python scripts/ingest_fmp.py --wave 1,2 --dry-run

    # run waves 1 and 2
    python scripts/ingest_fmp.py --wave 1,2

    # one endpoint, re-fetching what exists
    python scripts/ingest_fmp.py --endpoints earnings --force

    # what happened last run
    python scripts/ingest_fmp.py --report
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.keys import (
    ECONOMIC_INDICATORS,
    insider_reporting_names,
    issuer_names,
    legislator_names,
    universe_ciks,
)
from core.data.fmp.transport import make_fetcher
from core.ingest.catalog import build_specs, load_manifest, select_specs
from core.ingest.journal import IngestJournal
from core.ingest.plan import plan_run
from core.ingest.pool import execute
from core.ingest.ratelimit import TokenBucket
from core.ingest.report import render_report, write_report
from core.ingest.runner import IngestRunner, existing_output
from core.ingest.spec import Partition

logger = logging.getLogger("ingest_fmp")

JOURNAL_PATH = ROOT / "data" / "quality" / "ingest_journal.db"
SECURITY_MASTER = ROOT / "data" / "universe" / "security_master.parquet"
REPORT_DIR = ROOT / "data" / "quality" / "ingest_reports"


def resolve_keys(raw_root: Path, symbols_override: list[str] | None) -> dict[object, list[str]]:
    """
    Collect the partition keys each endpoint family needs.

    Symbols come from the security master. The remaining key sources — CIKs,
    sectors, industries, exchanges — are themselves wave-1 downloads, so they are
    read from the raw layer when present. A missing source is not fatal: the
    planner logs and skips those endpoints, and a later run picks them up once
    wave 1 has landed.

    Args:
        raw_root: Vendor raw layer root.
        symbols_override: Explicit symbol list, bypassing the security master.

    Returns:
        Partition keys by partition type.
    """
    keys: dict[Partition, list[str]] = {}

    if symbols_override:
        symbols = sorted(set(symbols_override))
    elif SECURITY_MASTER.is_file():
        symbols = sorted(pd.read_parquet(SECURITY_MASTER)["symbol"].dropna().unique().tolist())
    else:
        symbols = []
        logger.warning(
            "no security master at %s; per-symbol endpoints will be skipped", SECURITY_MASTER
        )
    keys[Partition.PER_SYMBOL] = symbols
    keys[Partition.BATCH_SYMBOLS] = symbols

    def column_from(name: str, column: str) -> list[str]:
        path = raw_root / name / "_all.parquet"
        if not path.is_file():
            return []
        frame = pd.read_parquet(path)
        if column not in frame.columns:
            return []
        return sorted({str(v) for v in frame[column].dropna().tolist()})

    keys[Partition.PER_SECTOR] = column_from("available_sectors", "sector")
    keys[Partition.PER_INDUSTRY] = column_from("available_industries", "industry")
    keys[Partition.PER_EXCHANGE] = column_from("available_exchanges", "exchange")

    # CIK- and name-keyed endpoints take their keys from wave-1 downloads, so
    # they fill in as earlier waves land rather than being hard-coded. CIKs are
    # scoped to our own universe: FMP lists 491,000 registrants, and running the
    # five per-CIK endpoints across all of them would be 2.45 million requests.
    keys[Partition.PER_CIK] = universe_ciks(raw_root)

    # Name-keyed endpoints do NOT share one pool: economic-indicators takes 21
    # economic series, the by-name trade endpoints take legislators, the search
    # endpoints take company names. Pooling them would spend 12,000 requests per
    # endpoint on names that cannot match. Each spec names its own source.
    keys[Partition.PER_NAME] = []
    keys["economic_indicators"] = list(ECONOMIC_INDICATORS)
    keys["legislators"] = legislator_names(raw_root)
    keys["issuer_names"] = issuer_names(raw_root)
    keys["insider_names"] = insider_reporting_names(raw_root)

    for source, values in keys.items():
        label = source.value if isinstance(source, Partition) else str(source)
        logger.info("%-20s %d keys", label, len(values))
    return keys


def main() -> int:
    """Parse arguments, plan, and run. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--vendor", default="fmp")
    parser.add_argument("--wave", default="1,2", help="comma-separated wave numbers, or 'all'")
    parser.add_argument("--endpoints", default=None, help="comma-separated endpoint names")
    parser.add_argument("--symbols", default=None, help="comma-separated symbol override")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rate", type=float, default=None, help="calls/min; default from manifest")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="re-fetch existing files")
    parser.add_argument("--include-derivable", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="plan and cost only")
    parser.add_argument(
        "--report", action="store_true", help="print the last run's report and exit"
    )
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    if args.report:
        print(render_report(JOURNAL_PATH))
        return 0

    manifest = load_manifest(args.vendor)
    all_specs = build_specs(manifest)
    raw_root = ROOT / manifest.get("raw_root", f"data/raw/{args.vendor}")

    names = set(args.endpoints.split(",")) if args.endpoints else None
    waves = None if args.wave == "all" else {int(w) for w in args.wave.split(",")}
    specs = select_specs(
        all_specs,
        waves=waves,
        names=names,
        include_derivable=args.include_derivable or names is not None,
    )
    if not specs:
        logger.error("no endpoints selected")
        return 2

    keys = resolve_keys(raw_root, args.symbols.split(",") if args.symbols else None)
    tasks = plan_run(specs, keys)

    rate = args.rate or manifest.get("rate_limit_per_minute", 500)
    spec_by_name = {spec.name: spec for spec in specs}
    remaining = tasks
    if not args.force:
        # Completion is a file on disk, so the honest cost of a run is what is
        # still missing — not the plan size, which is meaningless once a
        # previous run has landed 60,000 files.
        remaining = [
            task
            for task in tasks
            if existing_output(raw_root, spec_by_name[task.spec_name], task.key) is None
        ]
    logger.info(
        "planned %d tasks across %d endpoints (waves=%s); %d already on disk, %d to fetch",
        len(tasks),
        len(specs),
        args.wave,
        len(tasks) - len(remaining),
        len(remaining),
    )
    logger.info(
        "at %.0f calls/min with %d workers: %.1f hours remaining",
        rate,
        args.workers,
        len(remaining) / rate / 60,
    )

    if args.dry_run:
        from collections import Counter

        planned = Counter(task.spec_name for task in tasks)
        todo = Counter(task.spec_name for task in remaining)
        logger.info("  %-46s %9s %9s", "endpoint", "on disk", "to fetch")
        for name, count in planned.most_common():
            logger.info("  %-46s %9d %9d", name, count - todo.get(name, 0), todo.get(name, 0))
        return 0

    run_id = f"{args.vendor}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    journal = IngestJournal(JOURNAL_PATH, args.vendor, run_id, argv=" ".join(sys.argv[1:]))
    bucket = TokenBucket(rate)
    runner = IngestRunner(
        fetch=make_fetcher(),
        raw_root=raw_root,
        journal=journal,
        bucket=bucket,
        workers=args.workers,
        max_retries=args.max_retries,
    )
    runner.install_signal_handlers()

    logger.info("run_id=%s journal=%s", run_id, JOURNAL_PATH)
    summary = execute(
        runner, journal, tasks, {s.name: s for s in specs}, workers=args.workers, force=args.force
    )

    logger.info(
        "run %s finished in %.1f min: %s%s",
        run_id,
        summary.elapsed_seconds / 60,
        ", ".join(f"{k}={v}" for k, v in sorted(summary.counts.items())),
        " (INTERRUPTED)" if summary.interrupted else "",
    )
    report_path = write_report(JOURNAL_PATH, REPORT_DIR, run_id)
    logger.info("report written to %s", report_path)
    journal.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
