"""Sector breakdown analytics endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from core.data.sector_classification import (
    get_sector_summary,
    load_sector_classifications,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sectors", tags=["sectors"])


@router.get("/summary")
def sector_summary() -> dict:
    """Return sector distribution summary (counts + percentages)."""
    df = load_sector_classifications()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Sector data not available")

    summary = get_sector_summary()
    total = int(summary["count"].sum()) if "count" in summary.columns else len(df)

    rows = []
    for _, row in summary.iterrows():
        rows.append({
            "sector": str(row.get("sector", row.name)),
            "count": int(row.get("count", 0)),
            "pct": round(float(row.get("count", 0)) / total * 100, 1) if total > 0 else 0.0,
        })

    return {"total_symbols": total, "sectors": rows}


@router.get("/breakdown")
def sector_breakdown(sector: Optional[str] = Query(None)) -> dict:
    """Return symbols grouped by sector/industry."""
    df = load_sector_classifications()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Sector data not available")

    if sector:
        df = df[df["sector"] == sector]

    symbols = []
    for _, row in df.iterrows():
        symbols.append({
            "symbol": str(row.get("symbol", row.name)),
            "sector": str(row.get("sector", "Unknown")),
            "industry": str(row.get("industry", "Unknown")),
            "type": str(row.get("quoteType", "EQUITY")),
        })

    return {"symbols": symbols}
