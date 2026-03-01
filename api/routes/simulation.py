"""
Monte-Carlo simulation endpoints for educational demos.

Provides the Sharpe-comparison simulation that generates investments
with identical Sharpe ratios but different risk profiles.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulation", tags=["simulation"])


class InvestmentMetrics(BaseModel):
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    total_return: float
    annualized_vol: float
    skewness: float
    kurtosis: float
    win_rate: float
    best_day: float
    worst_day: float


class SimulatedInvestment(BaseModel):
    name: str
    color: str
    prices: List[float]
    daily_returns: List[float]
    metrics: InvestmentMetrics


class SharpeComparisonResponse(BaseModel):
    target_sharpe: float
    n_days: int
    seed: int
    investments: List[SimulatedInvestment]


def _generate_investment(
    name: str,
    target_sharpe: float,
    n_days: int,
    rng: np.random.Generator,
    *,
    vol_level: str = "medium",
    skew: float = 0.0,
    kurtosis: float = 0.0,
    drawdown_events: int = 0,
    color: str = "#1f77b4",
    rf: float = 0.02,
) -> SimulatedInvestment:
    vol_map = {"low": 0.12, "medium": 0.18, "high": 0.30}
    daily_vol = vol_map.get(vol_level, 0.18) / np.sqrt(252)

    returns = rng.normal(0, daily_vol, n_days)

    if skew < 0:
        crash_days = rng.choice(n_days, size=int(abs(skew) * 50), replace=False)
        returns[crash_days] -= daily_vol * rng.uniform(2, 5, len(crash_days))

    if kurtosis > 0:
        extreme_days = rng.choice(n_days, size=int(kurtosis * 20), replace=False)
        signs = rng.choice([-1.0, 1.0], len(extreme_days))
        returns[extreme_days] += signs * daily_vol * rng.uniform(3, 6, len(extreme_days))

    if drawdown_events > 0:
        starts = rng.choice(range(50, n_days - 50), size=drawdown_events, replace=False)
        for s in starts:
            dur = int(rng.integers(10, 30))
            crash = float(rng.uniform(0.10, 0.25))
            returns[s : s + dur] -= crash / dur

    current_std = returns.std()
    target_daily_mean = rf / 252 + target_sharpe * current_std / np.sqrt(252)
    returns = returns - returns.mean() + target_daily_mean

    prices = (100 * np.cumprod(1 + returns)).tolist()

    excess = returns - rf / 252
    sharpe_actual = float(np.sqrt(252) * excess.mean() / returns.std())

    down = returns[returns < 0]
    down_std = down.std() if len(down) > 0 else returns.std()
    sortino = float(np.sqrt(252) * excess.mean() / down_std)

    peak = np.maximum.accumulate(np.array(prices))
    dd = (np.array(prices) - peak) / peak
    max_dd = float(dd.min())

    ann_ret = float(returns.mean() * 252)
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    return SimulatedInvestment(
        name=name,
        color=color,
        prices=prices,
        daily_returns=(returns * 100).tolist(),
        metrics=InvestmentMetrics(
            sharpe=round(sharpe_actual, 4),
            sortino=round(sortino, 4),
            max_drawdown=round(max_dd * 100, 2),
            calmar=round(calmar, 4),
            total_return=round((prices[-1] / 100 - 1) * 100, 2),
            annualized_vol=round(float(returns.std() * np.sqrt(252) * 100), 2),
            skewness=round(float(sp_stats.skew(returns)), 4),
            kurtosis=round(float(sp_stats.kurtosis(returns)), 4),
            win_rate=round(float((returns > 0).sum() / len(returns) * 100), 2),
            best_day=round(float(returns.max() * 100), 4),
            worst_day=round(float(returns.min() * 100), 4),
        ),
    )


@router.post("/sharpe-comparison", response_model=SharpeComparisonResponse)
def sharpe_comparison(
    target_sharpe: float = Query(1.5, ge=0.1, le=5.0),
    n_days: int = Query(1260, ge=252, le=5040),
    seed: int = Query(42, ge=1, le=99999),
):
    """Generate five investments with identical Sharpe but different risk profiles."""
    rng = np.random.default_rng(seed)

    configs = [
        {"name": "Steady Eddie", "vol_level": "low", "color": "#2ecc71"},
        {"name": "Rollercoaster", "vol_level": "high", "color": "#e74c3c"},
        {"name": "Sneaky Losses", "vol_level": "medium", "skew": -2.0, "color": "#9b59b6"},
        {"name": "Fat Tails", "vol_level": "medium", "kurtosis": 3.0, "color": "#f39c12"},
        {"name": "Crash & Recover", "vol_level": "medium", "drawdown_events": 3, "color": "#3498db"},
    ]

    investments = [
        _generate_investment(
            target_sharpe=target_sharpe,
            n_days=n_days,
            rng=rng,
            **cfg,
        )
        for cfg in configs
    ]

    return SharpeComparisonResponse(
        target_sharpe=target_sharpe,
        n_days=n_days,
        seed=seed,
        investments=investments,
    )
