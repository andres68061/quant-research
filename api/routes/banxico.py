"""Banxico (Banco de México) endpoints -- CETES 28 risk-free rate."""

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/banxico", tags=["banxico"])


@router.get("/cetes28")
def get_cetes28() -> dict:
    """Fetch the latest CETES 28 annualized rate from Banxico."""
    try:
        from core.data.banxico_api import get_current_cetes28_rate

        rate, dt = get_current_cetes28_rate()
        return {"rate": round(rate * 100, 4), "date": str(dt.date())}
    except Exception as exc:
        logger.exception("Failed to fetch CETES 28")
        raise HTTPException(status_code=502, detail=f"Banxico API error: {exc}") from exc
