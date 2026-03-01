"""Metrics endpoints: performance metrics and VaR."""

from typing import Optional

from fastapi import APIRouter, HTTPException

from api.dependencies import get_prices
from api.schemas.metrics import AllVarResult, PerformanceMetrics, VarResult
from core.metrics.performance import calculate_performance_metrics
from core.metrics.risk import calculate_all_var

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/performance")
def performance_metrics(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute performance metrics for a single asset."""
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    if symbol not in prices.columns:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    returns = prices[symbol].pct_change().dropna()
    if start:
        returns = returns[returns.index >= start]
    if end:
        returns = returns[returns.index <= end]

    metrics = calculate_performance_metrics(
        returns, risk_free_rate=risk_free_rate
    )
    return PerformanceMetrics(**metrics).model_dump()


@router.get("/var")
def value_at_risk(
    symbol: str,
    confidence: int = 95,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Compute Historical, Parametric, and Monte-Carlo VaR."""
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    if symbol not in prices.columns:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    returns = prices[symbol].pct_change().dropna()
    if start:
        returns = returns[returns.index >= start]
    if end:
        returns = returns[returns.index <= end]

    all_var = calculate_all_var(returns.values, confidence=confidence)

    return AllVarResult(
        historical=VarResult(**all_var["historical"]),
        parametric=VarResult(**all_var["parametric"]),
        monte_carlo=VarResult(**all_var["monte_carlo"]),
        confidence=confidence,
    ).model_dump()
