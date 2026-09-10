#!/usr/bin/env python3
"""
Report ingestion progress cheaply, without attaching to the running process.

Written so that checking on a multi-hour backfill costs one command and no
supervision. It reads the journal and the raw layer, never the vendor, so it is
free to run as often as you like and safe to run while ingestion is in flight.

Usage:
    python scripts/ingest/ingest_status.py
    python scripts/ingest/ingest_status.py --failures 20
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "data" / "quality" / "ingest_journal.db"
HEARTBEAT = ROOT / "data" / "quality" / "ingest_heartbeat.txt"
RAW = ROOT / "data" / "raw" / "fmp"


def main() -> int:
    """Print a compact progress summary. Returns 0 always; this is a read-only view."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", type=int, default=10, help="failed tasks to list")
    args = parser.parse_args()

    if HEARTBEAT.is_file():
        print(f"heartbeat : {HEARTBEAT.read_text().strip()}")

    files = sum(1 for _ in RAW.rglob("*.parquet")) if RAW.is_dir() else 0
    print(f"raw files : {files:,} parquet under {RAW.relative_to(ROOT)}")

    if not JOURNAL.is_file():
        print("journal   : none yet")
        return 0

    connection = sqlite3.connect(str(JOURNAL))
    try:
        run = connection.execute(
            "SELECT run_id, started_at, finished_at, n_tasks FROM runs ORDER BY started_at DESC"
            " LIMIT 1"
        ).fetchone()
        if run is None:
            print("journal   : no runs recorded")
            return 0
        run_id, started, finished, n_tasks = run
        counts = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM tasks WHERE run_id=? GROUP BY status", (run_id,)
            ).fetchall()
        )
        done = sum(counts.values())
        elapsed = (finished or time.time()) - started
        # Skips are file-existence checks, not vendor calls. Counting them in the
        # rate inflates it by orders of magnitude at the start of a resumed run
        # and makes the estimate meaningless, so pace on fetched tasks only.
        fetched = done - counts.get("skipped", 0)
        rate = fetched / elapsed * 60 if elapsed > 0 else 0.0
        remaining = (n_tasks or 0) - done
        eta_minutes = remaining / rate if rate > 0 else float("nan")

        state = "finished" if finished else "RUNNING"
        print(f"run       : {run_id} ({state})")
        print(f"progress  : {done:,} / {n_tasks or 0:,} tasks  ({done / max(n_tasks or 1, 1):.1%})")
        print(
            f"rate      : {rate:.0f} fetches/min (skips excluded)   "
            f"eta {eta_minutes / 60:.1f} h for {remaining:,} remaining"
        )
        print("outcomes  : " + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))

        failures = connection.execute(
            "SELECT spec_name, COUNT(*), MIN(COALESCE(http_code,0)), MIN(error)"
            " FROM tasks WHERE run_id=? AND status='failed' GROUP BY spec_name"
            " ORDER BY COUNT(*) DESC LIMIT ?",
            (run_id, args.failures),
        ).fetchall()
        if failures:
            print("\nfailing endpoints")
            for name, count, code, sample in failures:
                print(f"  {name:<44}{count:>7,}  HTTP {code}  {(sample or '')[:60]}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
