"""
Monte-Carlo simulation endpoints for educational demos.

Provides ratio-comparison simulations that generate investments with identical
Sharpe, Sortino, or Calmar ratios but different risk profiles.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from scipy import stats as sp_stats
from scipy.optimize import brentq

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


class MultiRatioComparisonResponse(BaseModel):
    """Same ``target`` applied to three calibrations (Sharpe, Sortino, Calmar)."""

    target: float = Field(
        ...,
        description="Target ratio used for each block (Sharpe, Sortino, Calmar).",
    )
    n_days: int
    seed: int
    by_sharpe: List[SimulatedInvestment]
    by_sortino: List[SimulatedInvestment]
    by_calmar: List[SimulatedInvestment]


def _build_raw_returns(
    n_days: int,
    rng: np.random.Generator,
    *,
    vol_level: str = "medium",
    skew: float = 0.0,
    kurtosis: float = 0.0,
    drawdown_events: int = 0,
) -> np.ndarray:
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

    return returns


def _shift_mean(raw: np.ndarray, daily_mean: float) -> np.ndarray:
    return raw - raw.mean() + daily_mean


def _realized_sharpe(returns: np.ndarray, rf: float) -> float:
    excess = returns - rf / 252
    sig = returns.std(ddof=0)
    if sig < 1e-14:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / sig)


def _realized_sortino(returns: np.ndarray, rf: float) -> float:
    excess = returns - rf / 252
    down = returns[returns < 0]
    # Use downside obs whenever any exist (including a single negative day).
    # Falling back only when there are zero downside days matches _metrics_and_lists.
    down_std = down.std(ddof=0) if len(down) > 0 else returns.std(ddof=0)
    if down_std < 1e-14:
        return float(np.sign(excess.mean()) * 1e6) if excess.mean() != 0 else 0.0
    return float(np.sqrt(252) * excess.mean() / down_std)


def _realized_calmar(returns: np.ndarray) -> float:
    prices = 100 * np.cumprod(1 + returns)
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / peak
    max_dd = float(dd.min())
    ann_ret = float(returns.mean() * 252)
    if abs(max_dd) < 1e-14:
        return float(np.sign(ann_ret) * 1e6) if ann_ret != 0 else 0.0
    return ann_ret / abs(max_dd)


def _metrics_and_lists(returns: np.ndarray, rf: float) -> SimulatedInvestment:
    prices_arr = 100 * np.cumprod(1 + returns)
    prices = prices_arr.tolist()
    excess = returns - rf / 252
    peak = np.maximum.accumulate(prices_arr)
    dd = (prices_arr - peak) / peak
    max_dd = float(dd.min())
    ann_ret = float(returns.mean() * 252)
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    sortino = _realized_sortino(returns, rf)
    ret_std = float(returns.std(ddof=0))
    sharpe_actual = float(np.sqrt(252) * excess.mean() / ret_std) if ret_std > 1e-14 else 0.0

    return SimulatedInvestment(
        name="",
        color="",
        prices=prices,
        daily_returns=(returns * 100).tolist(),
        metrics=InvestmentMetrics(
            sharpe=round(sharpe_actual, 4),
            sortino=round(sortino, 4),
            max_drawdown=round(max_dd * 100, 2),
            calmar=round(calmar, 4),
            total_return=round((prices[-1] / 100 - 1) * 100, 2),
            annualized_vol=round(float(ret_std * np.sqrt(252) * 100), 2),
            skewness=round(float(sp_stats.skew(returns)), 4),
            kurtosis=round(float(sp_stats.kurtosis(returns)), 4),
            win_rate=round(float((returns > 0).sum() / len(returns) * 100), 2),
            best_day=round(float(returns.max() * 100), 4),
            worst_day=round(float(returns.min() * 100), 4),
        ),
    )


def _solve_mean_sharpe(raw: np.ndarray, target: float, rf: float) -> float:
    sigma = raw.std(ddof=0)
    if sigma < 1e-14:
        return rf / 252
    return float(rf / 252 + target * sigma / np.sqrt(252))


def _brentq_bracket(
    g,
    center: float,
    *,
    max_rounds: int = 96,
) -> Tuple[float, float]:
    span = 5e-5
    lo, hi = center - span, center + span
    for _ in range(max_rounds):
        glo, ghi = g(lo), g(hi)
        if not np.isfinite(glo) or not np.isfinite(ghi):
            span *= 1.7
            lo, hi = center - span, center + span
            continue
        if glo == 0:
            return lo, lo
        if ghi == 0:
            return hi, hi
        if glo * ghi < 0:
            return lo, hi
        span *= 1.65
        lo, hi = center - span, center + span
    raise ValueError("Could not bracket root")


def _solve_mean_sortino(raw: np.ndarray, target: float, rf: float) -> float:
    def g(daily_mean: float) -> float:
        r = _shift_mean(raw, daily_mean)
        return _realized_sortino(r, rf) - target

    center = _solve_mean_sharpe(raw, target, rf)
    try:
        lo, hi = _brentq_bracket(g, center)
        return float(brentq(g, lo, hi, maxiter=400))
    except (ValueError, RuntimeError):
        logger.warning("Sortino mean-shift brentq failed; falling back to Sharpe-matched mean")
        return center


def _solve_mean_calmar(raw: np.ndarray, target: float) -> float:
    def g(daily_mean: float) -> float:
        r = _shift_mean(raw, daily_mean)
        return _realized_calmar(r) - target

    center = float(raw.mean())
    try:
        lo, hi = _brentq_bracket(g, center)
        return float(brentq(g, lo, hi, maxiter=400))
    except (ValueError, RuntimeError):
        logger.warning("Calmar mean-shift brentq failed; using raw mean")
        return center


def _finalize_investment(
    name: str,
    color: str,
    raw: np.ndarray,
    daily_mean: float,
    rf: float,
) -> SimulatedInvestment:
    returns = _shift_mean(raw, daily_mean)
    base = _metrics_and_lists(returns, rf)
    return SimulatedInvestment(
        name=name,
        color=color,
        prices=base.prices,
        daily_returns=base.daily_returns,
        metrics=base.metrics,
    )


@router.post("/sharpe-comparison", response_model=MultiRatioComparisonResponse)
def sharpe_comparison(
    target_sharpe: float = Query(
        1.5,
        ge=0.1,
        le=5.0,
        description=(
            "Target ratio for all three blocks: Sharpe, Sortino, and Calmar are each "
            "mean-calibrated to this value for the same synthetic paths."
        ),
    ),
    n_days: int = Query(1260, ge=252, le=5040),
    seed: int = Query(42, ge=1, le=99999),
):
    """Generate five investments per ratio: same shapes, mean calibrated per metric."""
    tval = target_sharpe
    rng = np.random.default_rng(seed)
    rf = 0.02

    configs: List[dict] = [
        {"name": "Steady Eddie", "vol_level": "low", "color": "#2ecc71"},
        {"name": "Rollercoaster", "vol_level": "high", "color": "#e74c3c"},
        {"name": "Sneaky Losses", "vol_level": "medium", "skew": -2.0, "color": "#9b59b6"},
        {"name": "Fat Tails", "vol_level": "medium", "kurtosis": 3.0, "color": "#f39c12"},
        {"name": "Crash & Recover", "vol_level": "medium", "drawdown_events": 3, "color": "#3498db"},
    ]

    by_sharpe: List[SimulatedInvestment] = []
    by_sortino: List[SimulatedInvestment] = []
    by_calmar: List[SimulatedInvestment] = []

    for cfg in configs:
        name = cfg["name"]
        color = cfg["color"]
        raw = _build_raw_returns(
            n_days,
            rng,
            vol_level=cfg.get("vol_level", "medium"),
            skew=float(cfg.get("skew", 0.0)),
            kurtosis=float(cfg.get("kurtosis", 0.0)),
            drawdown_events=int(cfg.get("drawdown_events", 0)),
        )
        m_s = _solve_mean_sharpe(raw, tval, rf)
        m_so = _solve_mean_sortino(raw, tval, rf)
        m_c = _solve_mean_calmar(raw, tval)
        by_sharpe.append(_finalize_investment(name, color, raw, m_s, rf))
        by_sortino.append(_finalize_investment(name, color, raw, m_so, rf))
        by_calmar.append(_finalize_investment(name, color, raw, m_c, rf))

    return MultiRatioComparisonResponse(
        target=tval,
        n_days=n_days,
        seed=seed,
        by_sharpe=by_sharpe,
        by_sortino=by_sortino,
        by_calmar=by_calmar,
    )
