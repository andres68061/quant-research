#!/usr/bin/env python3
"""
Run the data watchdog, persist its verdict, and escalate failures to the desktop.

Why escalation matters: every earlier safety mechanism in this repo wrote to a
log, and logs are only read after someone already suspects a problem. This script
closes that loop with three channels, in increasing order of intrusiveness:

1. ``data/quality/watchdog_status.json`` — machine-readable, served by
   ``GET /watchdog`` and rendered as a banner in the app. Passive.
2. ``runtime/logs/watchdog.log`` — the audit trail. Passive.
3. A macOS notification, **only on error**. Active: it appears whether or not
   anyone is looking at the platform.

The baseline update is the subtle part. Row counts are recorded as the new
reference *only when the run is clean*, so a collapse stays visible on every
subsequent run instead of quietly becoming the new normal.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/ops/run_watchdog.py
    /opt/anaconda3/envs/quant/bin/python scripts/ops/run_watchdog.py --deep
    /opt/anaconda3/envs/quant/bin/python scripts/ops/run_watchdog.py --set-baseline
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.quality.watchdog import (  # noqa: E402
    current_row_counts,
    run_watchdog,
    write_baseline,
    write_status,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("watchdog")

NOTIFY_TITLE = "Quant platform — data problem"


def notify_desktop(title: str, message: str) -> None:
    """
    Raise a macOS notification. Best-effort: never fail the run over the alert.

    Truncated because the notification centre silently drops long bodies, and a
    dropped alert is worse than a terse one.
    """
    body = message if len(message) <= 220 else message[:217] + "..."
    # Text goes in as argv, never interpolated into the script: a verdict with
    # an em dash or quotes broke the AppleScript parser and the alert was lost.
    script = (
        "on run argv\n"
        '  display notification (item 1 of argv) with title (item 2 of argv) sound name "Basso"\n'
        "end run"
    )
    try:
        subprocess.run(["osascript", "-e", script, body, title], check=False, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Desktop notification failed: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run data watchdog checks")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also scan panel values for structural violations (~10s slower)",
    )
    parser.add_argument(
        "--set-baseline",
        action="store_true",
        help="Record current row counts as the clean reference and exit",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the desktop notification")
    args = parser.parse_args()

    if args.set_baseline:
        counts = current_row_counts()
        write_baseline(counts)
        logger.info("Baseline recorded for %d panels", len(counts))
        return 0

    snapshot = run_watchdog(deep=args.deep)
    write_status(snapshot)

    for check in snapshot["checks"]:
        level = {"error": logging.ERROR, "warning": logging.WARNING}.get(
            check["status"], logging.INFO
        )
        logger.log(level, "%s [%s] %s", check["name"], check["status"], check["detail"])

    logger.info("Watchdog verdict: %s — %s", snapshot["status"], snapshot["summary"])

    if snapshot["status"] == "error":
        if not args.quiet:
            notify_desktop(NOTIFY_TITLE, snapshot["summary"])
        # Non-zero exit so a CI step or a wrapping job fails loudly too.
        return 1

    # Only a clean run advances the baseline; otherwise damage becomes the norm.
    if snapshot["status"] == "ok":
        write_baseline(current_row_counts())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
