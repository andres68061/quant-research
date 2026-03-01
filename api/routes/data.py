"""Data discovery endpoints: assets, factors, prices."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from api.dependencies import get_factors, get_prices, get_sectors
from core.data.asset_classification import categorize_asset_type

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/assets")
def list_assets() -> dict:
    """Return available assets in the price universe."""
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    sectors = get_sectors()
    assets = []
    for symbol in sorted(prices.columns):
        asset_type = categorize_asset_type(symbol, sectors)
        assets.append({"symbol": symbol, "type": asset_type})
    return {"count": len(assets), "assets": assets}


@router.get("/factors")
def list_factors() -> dict:
    """Return available factor columns."""
    factors = get_factors()
    if factors is None:
        raise HTTPException(status_code=503, detail="Factor data not loaded")

    factor_cols = [c for c in factors.columns if c != "signal"]
    return {"count": len(factor_cols), "factors": factor_cols}


@router.get("/prices")
def get_price_series(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Return price series for a single symbol as JSON."""
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    if symbol not in prices.columns:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    series = prices[symbol].dropna()
    if start:
        series = series[series.index >= start]
    if end:
        series = series[series.index <= end]

    return {
        "symbol": symbol,
        "count": len(series),
        "data": [
            {"date": str(d.date()), "price": float(v)}
            for d, v in series.items()
        ],
    }
