"""Sortino momentum analysis endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_prices

router = APIRouter(prefix="/momentum", tags=["momentum"])
logger = logging.getLogger(__name__)


@router.get("/grid-search")
def momentum_grid_search(
    symbol: str = Query(...),
    sortino_window: int = Query(252, ge=60),
) -> dict:
    """Run Sortino momentum grid search for a symbol."""
    from core.signals.momentum import analyze_momentum_grid_search

    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    if symbol not in prices.columns:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    returns = prices[symbol].pct_change().dropna()
    if len(returns) < sortino_window + 60:
        raise HTTPException(status_code=400, detail="Not enough data for grid search")

    df = analyze_momentum_grid_search(
        returns, sortino_window=sortino_window, min_signals=10
    )
    records = df.head(20).to_dict(orient="records")
    return {"symbol": symbol, "results": records}


@router.get("/bootstrap")
def momentum_bootstrap(
    symbol: str = Query(...),
    x: int = Query(10, ge=1),
    k: int = Query(10, ge=1),
    sortino_window: int = Query(252, ge=60),
) -> dict:
    """Run bootstrap significance test for a (x, k) pair."""
    from core.signals.momentum import bootstrap_significance_test

    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    if symbol not in prices.columns:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    returns = prices[symbol].pct_change().dropna()
    result = bootstrap_significance_test(
        returns, x=x, k=k, sortino_window=sortino_window
    )
    dist = result.pop("bootstrap_dist", None)
    return {"symbol": symbol, "x": x, "k": k, "bootstrap_dist": dist or [], **result}


@router.get("/regime")
def momentum_regime(
    symbol: str = Query(...),
    x: int = Query(10, ge=1),
    k: int = Query(10, ge=1),
    sortino_window: int = Query(252, ge=60),
) -> dict:
    """Get current momentum regime for a symbol."""
    from core.signals.momentum import get_current_regime

    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    if symbol not in prices.columns:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    returns = prices[symbol].pct_change().dropna()
    regime = get_current_regime(returns, x=x, k=k, sortino_window=sortino_window)
    return {"symbol": symbol, "regime": regime}
