"""
Mean-variance portfolio optimization.

Implements Markowitz efficient frontier, tangency (max-Sharpe) portfolio,
minimum-variance portfolio, Capital Allocation Line, and rebalancing
simulation.  All functions are pure (no I/O) and operate on NumPy/Pandas
objects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> Tuple[float, float]:
    """Return annualized (return, volatility) for a given weight vector."""
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    return ret, vol


def _neg_sharpe(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    ret, vol = portfolio_stats(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol


def _portfolio_vol(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    return portfolio_stats(weights, mean_returns, cov_matrix)[1]


def find_tangency_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> Dict:
    """Find the max-Sharpe (tangency) portfolio.

    Returns
    -------
    dict with keys: weights, ret, volatility, sharpe
    """
    n = len(mean_returns)
    init = np.full(n, 1.0 / n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        _neg_sharpe,
        init,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    ret, vol = portfolio_stats(result.x, mean_returns, cov_matrix)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return {
        "weights": result.x.tolist(),
        "ret": ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


def find_min_variance_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> Dict:
    """Find the global minimum-variance portfolio.

    Returns
    -------
    dict with keys: weights, ret, volatility, sharpe (sharpe uses rf=0)
    """
    n = len(mean_returns)
    init = np.full(n, 1.0 / n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        _portfolio_vol,
        init,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    ret, vol = portfolio_stats(result.x, mean_returns, cov_matrix)
    return {
        "weights": result.x.tolist(),
        "ret": ret,
        "volatility": vol,
        "sharpe": ret / vol if vol > 0 else 0.0,
    }


def calculate_efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    n_points: int = 50,
) -> List[Dict]:
    """Compute the efficient frontier (long-only).

    Returns a list of dicts with keys: volatility, ret.
    """
    n = len(mean_returns)
    init = np.full(n, 1.0 / n)
    bounds = tuple((0.0, 1.0) for _ in range(n))

    min_result = minimize(
        _portfolio_vol,
        init,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    min_ret, _ = portfolio_stats(min_result.x, mean_returns, cov_matrix)
    max_ret = float(np.max(mean_returns))

    targets = np.linspace(min_ret, max_ret, n_points)
    frontier: List[Dict] = []

    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
        ]
        res = minimize(
            _portfolio_vol,
            init,
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000},
        )
        if res.success:
            ret, vol = portfolio_stats(res.x, mean_returns, cov_matrix)
            frontier.append({"volatility": vol, "ret": ret})

    return frontier


def calculate_cal_points(
    tangency_ret: float,
    tangency_vol: float,
    risk_free_rate: float,
    borrowing_rate: float,
    n_points: int = 60,
) -> List[Dict]:
    """Compute Capital Allocation Line points (lending + borrowing)."""
    max_vol = tangency_vol * 2.0
    vols = np.linspace(0, max_vol, n_points)

    lending_slope = (tangency_ret - risk_free_rate) / tangency_vol if tangency_vol > 0 else 0.0
    borrowing_slope = (tangency_ret - borrowing_rate) / tangency_vol if tangency_vol > 0 else 0.0

    points: List[Dict] = []
    for v in vols:
        if v <= tangency_vol:
            r = risk_free_rate + lending_slope * v
        else:
            r = tangency_ret + borrowing_slope * (v - tangency_vol)
        points.append({"volatility": float(v), "ret": float(r)})
    return points


def simulate_rebalanced_portfolio(
    prices_df: pd.DataFrame,
    weights: np.ndarray,
    freq: str = "Annual",
) -> pd.Series:
    """Simulate a portfolio NAV with periodic rebalancing.

    Parameters
    ----------
    prices_df : DataFrame with datetime index and one column per asset
    weights   : target allocation (same order as columns)
    freq      : "Annual", "Quarterly", or "Monthly"

    Returns
    -------
    pd.Series of portfolio NAV (starting at 1.0)
    """
    returns = prices_df.pct_change().dropna()
    dates = returns.index
    n = len(dates)
    if n == 0:
        return pd.Series(dtype=float)

    nav = np.ones(n)
    current_w = weights.copy().astype(float)

    if freq == "Quarterly":
        rebal_set = set(dates[dates.is_quarter_start])
    elif freq == "Monthly":
        rebal_set = set(dates[dates.is_month_start])
    else:
        years = pd.Series(dates).dt.year.unique()
        rebal_set = set()
        for yr in years:
            yr_dates = dates[pd.Series(dates).dt.year == yr]
            if len(yr_dates):
                rebal_set.add(yr_dates[0])

    for t in range(n - 1):
        r = returns.iloc[t].values
        nav[t + 1] = nav[t] * (1.0 + np.dot(current_w, r))

        if dates[t] in rebal_set:
            current_w = weights.copy().astype(float)
        else:
            current_w = current_w * (1.0 + r)
            s = current_w.sum()
            if s > 0:
                current_w /= s

    return pd.Series(nav, index=dates)
