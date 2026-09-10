"""Watchdog status endpoint — the app-facing half of unattended monitoring.

``scripts/ops/run_watchdog.py`` writes a verdict to disk on a schedule. This serves
it, so the platform can show a banner instead of relying on anyone opening a log
file. Serving the persisted snapshot (rather than re-running the checks) keeps
the endpoint instant and means the banner reflects exactly what the scheduled run
saw.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from core.data.quality.watchdog import STATUS_FILE, load_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/watchdog", tags=["watchdog"])


@router.get("")
def get_watchdog_status() -> dict[str, Any]:
    """
    The last watchdog verdict.

    Returns ``status: "unknown"`` rather than 404 when the watchdog has never
    run — a missing monitor is itself worth showing in the UI, and a 404 would
    render as a generic network error instead.
    """
    snapshot = load_status()
    if snapshot is None:
        return {
            "status": "unknown",
            "generated_at": None,
            "n_errors": 0,
            "n_warnings": 0,
            "checks": [],
            "summary": (
                "The data watchdog has never run. Schedule "
                "scripts/ops/run_watchdog.py or run it once to populate "
                f"{STATUS_FILE.name}."
            ),
        }
    return snapshot
