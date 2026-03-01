"""FRED (Federal Reserve Economic Data) endpoints."""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from core.data.fred import INDICATOR_CATALOG, get_fred_series, get_recession_periods

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fred", tags=["fred"])


@router.get("/catalog")
def catalog() -> dict:
    """Return the indicator catalog grouped by category."""
    categories = [
        {"category": cat, "indicators": indicators}
        for cat, indicators in INDICATOR_CATALOG.items()
    ]
    return {"categories": categories}


@router.get("/series")
def series(
    ids: List[str] = Query(...),
    start: Optional[str] = None,
    end: Optional[str] = None,
    yoy: bool = False,
) -> dict:
    """Fetch time series for the requested FRED indicators."""
    try:
        df = get_fred_series(ids, start=start, end=end, yoy=yoy)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if df.empty:
        return {"series": [], "metadata": {}}

    flat_catalog = {
        ind["id"]: {"name": ind["name"], "unit": ind["unit"]}
        for indicators in INDICATOR_CATALOG.values()
        for ind in indicators
    }
    metadata = {sid: flat_catalog.get(sid, {"name": sid, "unit": ""}) for sid in ids}

    records = []
    for date, row in df.iterrows():
        point: dict = {"date": str(date.date()) if hasattr(date, "date") else str(date)}
        for sid in ids:
            val = row.get(sid)
            point[sid] = round(float(val), 4) if val is not None and val == val else None
        records.append(point)

    return {"series": records[-2000:], "metadata": metadata}


@router.get("/recessions")
def recessions(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Return NBER recession periods."""
    try:
        periods = get_recession_periods(start=start, end=end)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"periods": periods}
