"""Data-health endpoints: audit snapshot, flaw registry, per-symbol drilldown.

Thin handlers over ``core.data.quality.health``. The snapshot itself is precomputed by
``scripts/ops/audit_data_health.py`` (a full audit walks ~50k parquet footers — too
slow for a request), so ``GET /data-health`` serves the persisted JSON and
reports its age. The per-symbol drilldown reads ~20 small files and is served
live.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from config.settings import PROJECT_ROOT
from core.data.quality.health import load_symbol_detail

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-health", tags=["data-health"])

SNAPSHOT_PATH = Path(PROJECT_ROOT) / "data" / "quality" / "data_health.json"


@router.get("")
def get_data_health() -> dict[str, Any]:
    """The full audit snapshot: funnel, coverage, survivorship, flaws, panels."""
    if not SNAPSHOT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No audit snapshot. Run scripts/ops/audit_data_health.py first.",
        )
    return json.loads(SNAPSHOT_PATH.read_text())


@router.get("/symbol/{symbol}")
def get_symbol_detail(symbol: str) -> dict[str, Any]:
    """Everything held for one symbol, across every raw layer and dataset."""
    detail = load_symbol_detail(symbol.upper())
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No data held for {symbol.upper()}")
    return detail
