"""
Synthetic-data explorer for performance measures.

Lets you see how the two main diagnostic measures (``cid1_ratio``,
``cid2_ratio``) and the reference measures (Sharpe / Sortino / Calmar /
Pain Ratio / Martin Ratio) behave on the *same* made-up data: first for
a single stock (five risk shapes), then for a 2-asset portfolio blend of
two of those shapes (to show how a measure that isn't a linear function of
its inputs can differ from a naive weighted average of the legs' own
values), and finally across many random single-stock draws so
measure-vs-measure relationships (linear or not) are visible directly.

Synthetic path generation lives here rather than in ``core/`` deliberately,
matching the existing ``api/routes/simulation.py`` precedent: this is a
pedagogical random-data generator, not quant math over real data. Every
actual metric computation is delegated to
``core.metrics.performance.calculate_performance_metrics``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.metrics.performance import calculate_performance_metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/measures-lab", tags=["measures-lab"])

_ARCHETYPES: List[Dict[str, Any]] = [
    {"name": "Steady Eddie", "vol_level": "low", "color": "#2ecc71"},
    {"name": "Rollercoaster", "vol_level": "high", "color": "#e74c3c"},
    {"name": "Sneaky Losses", "vol_level": "medium", "skew": -2.0, "color": "#9b59b6"},
    {"name": "Fat Tails", "vol_level": "medium", "kurtosis": 3.0, "color": "#f39c12"},
    {"name": "Crash & Recover", "vol_level": "medium", "drawdown_events": 3, "color": "#3498db"},
]
_MEASURE_FIELDS = [
    "cid1_ratio",
    "cid2_ratio",
    "total_return",
    "typical_period_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "pain_ratio",
    "martin_ratio",
]


def _build_returns(
    n_days: int,
    rng: np.random.Generator,
    *,
    vol_level: str = "medium",
    skew: float = 0.0,
    kurtosis: float = 0.0,
    drawdown_events: int = 0,
    daily_mean: float = 0.0004,
) -> np.ndarray:
    vol_map = {"low": 0.12, "medium": 0.18, "high": 0.30}
    daily_vol = vol_map.get(vol_level, 0.18) / np.sqrt(252)
    returns = rng.normal(daily_mean, daily_vol, n_days)

    if skew < 0:
        crash_days = rng.choice(n_days, size=max(1, int(abs(skew) * 50)), replace=False)
        returns[crash_days] -= daily_vol * rng.uniform(2, 5, len(crash_days))
    if kurtosis > 0:
        extreme_days = rng.choice(n_days, size=max(1, int(kurtosis * 20)), replace=False)
        signs = rng.choice([-1.0, 1.0], len(extreme_days))
        returns[extreme_days] += signs * daily_vol * rng.uniform(3, 6, len(extreme_days))
    if drawdown_events > 0:
        starts = rng.choice(range(50, max(51, n_days - 50)), size=drawdown_events, replace=False)
        for s in starts:
            dur = int(rng.integers(10, 30))
            crash = float(rng.uniform(0.10, 0.25))
            returns[s : s + dur] -= crash / dur
    return returns


class MeasureSet(BaseModel):
    cid1_ratio: float
    cid2_ratio: float
    total_return: float
    typical_period_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    pain_ratio: float
    martin_ratio: float


class SyntheticSeries(BaseModel):
    name: str
    color: str
    prices: List[float]
    measures: MeasureSet


class MeasuresLabRequest(BaseModel):
    n_days: int = Field(756, ge=252, le=2520, description="~3 years by default")
    n_relationship_draws: int = Field(80, ge=10, le=300)
    seed: int = Field(42, ge=1, le=99999)
    portfolio_weight_a: float = Field(0.5, ge=0.0, le=1.0)
    portfolio_a: str = Field("Rollercoaster")
    portfolio_b: str = Field("Sneaky Losses")


class MeasuresLabResponse(BaseModel):
    single_stock_examples: List[SyntheticSeries]
    portfolio_example: SyntheticSeries
    portfolio_legs: List[SyntheticSeries]
    relationship_scatter: Dict[str, List[float]] = Field(
        ...,
        description="metric name -> values across n_relationship_draws random single-stock draws",
    )


def _to_series(name: str, color: str, returns: np.ndarray) -> SyntheticSeries:
    metrics = calculate_performance_metrics(pd.Series(returns))
    prices = (100 * np.cumprod(1 + returns)).tolist()
    return SyntheticSeries(
        name=name,
        color=color,
        prices=prices,
        measures=MeasureSet(**{k: metrics[k] for k in _MEASURE_FIELDS}),
    )


@router.post("", response_model=MeasuresLabResponse)
def measures_lab(req: MeasuresLabRequest) -> MeasuresLabResponse:
    """Single stock -> portfolio -> many-draw relationship scatter, one call."""
    rng = np.random.default_rng(req.seed)

    single_stock: List[SyntheticSeries] = []
    raw_by_name: Dict[str, np.ndarray] = {}
    for cfg in _ARCHETYPES:
        raw = _build_returns(
            req.n_days,
            rng,
            vol_level=cfg.get("vol_level", "medium"),
            skew=float(cfg.get("skew", 0.0)),
            kurtosis=float(cfg.get("kurtosis", 0.0)),
            drawdown_events=int(cfg.get("drawdown_events", 0)),
        )
        raw_by_name[cfg["name"]] = raw
        single_stock.append(_to_series(cfg["name"], cfg["color"], raw))

    leg_a_name = req.portfolio_a if req.portfolio_a in raw_by_name else _ARCHETYPES[0]["name"]
    leg_b_name = req.portfolio_b if req.portfolio_b in raw_by_name else _ARCHETYPES[1]["name"]
    raw_a, raw_b = raw_by_name[leg_a_name], raw_by_name[leg_b_name]
    w = req.portfolio_weight_a
    blended = w * raw_a + (1 - w) * raw_b

    portfolio_example = _to_series(
        f"{leg_a_name} {w:.0%} / {leg_b_name} {1 - w:.0%}", "#facc15", blended
    )
    portfolio_legs = [
        _to_series(leg_a_name, "#e74c3c", raw_a),
        _to_series(leg_b_name, "#9b59b6", raw_b),
    ]

    scatter: Dict[str, List[float]] = {k: [] for k in _MEASURE_FIELDS}
    for _ in range(req.n_relationship_draws):
        raw = _build_returns(
            req.n_days,
            rng,
            vol_level=str(rng.choice(["low", "medium", "high"])),
            skew=float(rng.uniform(-3, 0)),
            kurtosis=float(rng.uniform(0, 4)),
            drawdown_events=int(rng.integers(0, 4)),
            daily_mean=float(rng.uniform(-0.0003, 0.0008)),
        )
        metrics = calculate_performance_metrics(pd.Series(raw))
        for k in _MEASURE_FIELDS:
            scatter[k].append(round(float(metrics[k]), 6))

    return MeasuresLabResponse(
        single_stock_examples=single_stock,
        portfolio_example=portfolio_example,
        portfolio_legs=portfolio_legs,
        relationship_scatter=scatter,
    )
