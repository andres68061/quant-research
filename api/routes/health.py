"""Health-check endpoint."""

from fastapi import APIRouter

from api.dependencies import get_factors, get_prices

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """Return service status and data availability."""
    factors = get_factors()
    prices = get_prices()
    return {
        "status": "ok",
        "data": {
            "factors_loaded": factors is not None,
            "factors_shape": list(factors.shape) if factors is not None else None,
            "prices_loaded": prices is not None,
            "prices_shape": list(prices.shape) if prices is not None else None,
        },
    }
