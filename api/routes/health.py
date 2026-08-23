"""Health-check endpoint."""

from fastapi import APIRouter

from api.dependencies import get_factor_store, get_prices

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """Return service status and data availability."""
    store = get_factor_store()
    prices = get_prices()
    return {
        "status": "ok",
        "data": {
            "factors_loaded": store is not None,
            "factors_available": len(store.available_factors) if store is not None else 0,
            "prices_loaded": prices is not None,
            "prices_shape": list(prices.shape) if prices is not None else None,
        },
    }
