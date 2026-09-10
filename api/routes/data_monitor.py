"""Data Monitor: exploratory views over every commodity and macro series.

Thin handlers. All statistics come from ``core.research.eda`` via
``core.research.monitor``; this module only loads the two panels (cached on
file mtime), validates parameters, and serialises.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from core.data.factors.macro import RAW_MACRO_PARQUET
from core.data.vendors.commodities import CommodityDataFetcher
from core.research.caveats import SURFACE_DATA_MONITOR, as_dicts, caveats_for_surface
from core.research.eda import staleness as compute_staleness
from core.research.monitor import (
    GROUP_LABELS,
    GROUP_ORDER,
    curve_report,
    find_series,
    monitored_series_catalog,
    pivot_raw_macro,
    series_report,
    staleness_board,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-monitor", tags=["data-monitor"])

_TRANSFORMS = ("level", "diff", "pct_change", "log_return")


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@lru_cache(maxsize=4)
def _commodities(mtime: float) -> pd.DataFrame:
    """The commodity close panel; ``mtime`` is the cache key."""
    return CommodityDataFetcher().load_prices()


@lru_cache(maxsize=4)
def _macro(mtime: float) -> pd.DataFrame:
    """Raw FRED long table pivoted to native frequency; ``mtime`` is the cache key."""
    if not RAW_MACRO_PARQUET.exists():
        return pd.DataFrame()
    return pivot_raw_macro(pd.read_parquet(RAW_MACRO_PARQUET))


def _panels() -> Dict[str, pd.DataFrame]:
    fetcher = CommodityDataFetcher()
    return {
        "fmp": _commodities(_mtime(fetcher.prices_file)),
        "fred": _macro(_mtime(RAW_MACRO_PARQUET)),
    }


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


@router.get("/catalog")
def catalog() -> dict:
    """Every monitored series, grouped for the picker."""
    groups: Dict[str, List[dict]] = {g: [] for g in GROUP_ORDER}
    for spec in monitored_series_catalog():
        groups.setdefault(spec.group, []).append(spec.to_dict())
    return {
        "groups": [
            {"id": g, "label": GROUP_LABELS.get(g, g), "series": groups[g]}
            for g in GROUP_ORDER
            if groups.get(g)
        ],
        "transforms": list(_TRANSFORMS),
    }


@router.get("/series/{series_id}")
def series(
    series_id: str,
    transform: Optional[str] = Query(default=None, description="Override the default transform"),
    start: Optional[str] = Query(default=None, description="Restrict history to >= this date"),
    bins: int = Query(default=50, ge=5, le=200),
) -> dict:
    """Distribution, level profile, seasonality, annual paths and staleness for one series."""
    spec = find_series(series_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown series '{series_id}'")
    if transform is not None and transform not in _TRANSFORMS:
        raise HTTPException(status_code=422, detail=f"transform must be one of {_TRANSFORMS}")

    panel = _panels()[spec.source]
    if spec.id not in panel.columns:
        raise HTTPException(
            status_code=503,
            detail=f"'{series_id}' is catalogued but not on disk; run its fetch script",
        )
    values = panel[spec.id]
    # Freshness is judged on the full series; the window only narrows the statistics.
    full_values = values
    if start:
        try:
            values = values.loc[pd.Timestamp(start) :]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Bad start date: {start}") from exc

    report = series_report(
        values,
        spec,
        as_of=_today(),
        transform=transform,  # type: ignore[arg-type]
        bins=bins,
    )
    report["staleness"] = compute_staleness(
        full_values, spec.id, _today(), spec.expected_max_gap_days
    ).to_dict()
    report["caveats"] = as_dicts(caveats_for_surface(SURFACE_DATA_MONITOR))
    return report


@router.get("/yield-curve")
def yield_curve(
    dates: Optional[List[str]] = Query(
        default=None,
        description="Snapshot dates (YYYY-MM-DD). Default: today, 1M, 3M, 1Y and 2Y ago.",
    ),
) -> dict:
    """Treasury curve snapshots and the history of its shape."""
    today = _today()
    if dates:
        try:
            stamps = [pd.Timestamp(d) for d in dates]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        stamps = [
            today,
            today - pd.DateOffset(months=1),
            today - pd.DateOffset(months=3),
            today - pd.DateOffset(years=1),
            today - pd.DateOffset(years=2),
        ]
    macro = _panels()["fred"]
    if macro.empty:
        raise HTTPException(
            status_code=503, detail="Raw macro panel missing; run fetch_raw_macro.py"
        )
    report = curve_report(macro, stamps)
    report["caveats"] = as_dicts(caveats_for_surface(SURFACE_DATA_MONITOR))
    return report


@router.get("/staleness")
def staleness() -> dict:
    """Freshness of every monitored series, worst first."""
    rows = staleness_board(_panels(), _today())
    counts = {
        s: sum(1 for r in rows if r["status"] == s) for s in ("fresh", "late", "stale", "empty")
    }
    return {"as_of": str(_today().date()), "counts": counts, "series": rows}


__all__ = ["router"]
