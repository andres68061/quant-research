"""Portfolio optimization endpoints (efficient frontier, tangency, simulation)."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_prices
from core.metrics.performance import calculate_performance_metrics
from core.optimization.portfolio import (
    calculate_cal_points,
    calculate_efficient_frontier,
    find_min_variance_portfolio,
    find_tangency_portfolio,
    portfolio_stats,
    simulate_rebalanced_portfolio,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class OptimizeRequest(BaseModel):
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    risk_free_rate: float = 0.08
    borrowing_rate: float = 0.11


class SimulateRequest(BaseModel):
    symbols: List[str]
    weights: Dict[str, float]
    freq: str = "Annual"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


def _get_returns(symbols: List[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"Symbols not found: {missing}")

    df = prices[symbols].dropna()
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    if len(df) < 60:
        raise HTTPException(status_code=400, detail="Insufficient data (need >=60 rows)")

    return df.pct_change().dropna()


@router.post("/optimize")
def optimize(req: OptimizeRequest) -> dict:
    """Compute efficient frontier, tangency, and min-vol portfolios."""
    returns = _get_returns(req.symbols, req.start_date, req.end_date)

    mean_ret = (returns.mean() * 252).values
    cov = (returns.cov() * 252).values
    std = (returns.std() * np.sqrt(252)).values

    tangency = find_tangency_portfolio(mean_ret, cov, req.risk_free_rate)
    min_vol = find_min_variance_portfolio(mean_ret, cov)
    frontier = calculate_efficient_frontier(mean_ret, cov, req.risk_free_rate)
    cal = calculate_cal_points(
        tangency["ret"], tangency["volatility"],
        req.risk_free_rate, req.borrowing_rate,
    )

    tangency["weights"] = {s: w for s, w in zip(req.symbols, tangency["weights"])}
    min_vol["weights"] = {s: w for s, w in zip(req.symbols, min_vol["weights"])}

    individual = [
        {"symbol": s, "ret": float(mean_ret[i]), "volatility": float(std[i])}
        for i, s in enumerate(req.symbols)
    ]

    return {
        "tangency": tangency,
        "min_vol": min_vol,
        "frontier": frontier,
        "cal": cal,
        "individual": individual,
    }


@router.post("/simulate")
def simulate(req: SimulateRequest) -> dict:
    """Simulate portfolio NAV with periodic rebalancing."""
    symbols = req.symbols
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"Symbols not found: {missing}")

    df = prices[symbols].dropna()
    if req.start_date:
        df = df[df.index >= req.start_date]
    if req.end_date:
        df = df[df.index <= req.end_date]

    weights = np.array([req.weights.get(s, 0.0) for s in symbols])
    weights = weights / weights.sum()

    nav = simulate_rebalanced_portfolio(df, weights, req.freq)
    daily_ret = nav.pct_change().dropna()
    metrics = calculate_performance_metrics(daily_ret)

    nav_list = [
        {"date": str(d.date()), "value": float(v)}
        for d, v in nav.items()
    ]

    return {"nav": nav_list, "metrics": metrics}
