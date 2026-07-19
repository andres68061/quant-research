"""
Mean-variance portfolio optimization.

Implements Markowitz efficient frontier, tangency (max-Sharpe) portfolio,
minimum-variance portfolio, Capital Allocation Line, and rebalancing
simulation.  All functions are pure (no I/O) and operate on NumPy/Pandas
objects.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

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


def run_walk_forward_tangency(
    prices: pd.DataFrame,
    symbols: List[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_months: int = 24,
    rebalance_months: int = 6,
    risk_free_rate: float = 0.0,
    portfolio_kind: str = "tangency",
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Roll tangency (or min-variance) weights forward with no lookahead.

    ``POST /portfolio/optimize`` fits weights on ``[start, end]`` and the UI's
    "Simulate" then backtests those same weights over the identical window —
    the optimizer has seen the returns it is later graded on (Markowitz
    estimation-risk look-ahead). This function is the actual fix: at each
    rebalance date, fit mean/cov on the trailing ``lookback_months`` of
    returns ending strictly before that date, hold the resulting weights
    (buy-and-hold drift, no interim rebalancing) for the next
    ``rebalance_months``, then re-fit. Concatenating the held periods'
    *realized* returns is a genuine out-of-sample NAV.

    Args:
        prices: Wide adj_close panel (date x symbol).
        symbols: Assets to include (equal starting universe every period).
        start: First rebalance date. Needs ``lookback_months`` of history
            before this for the first fit — the panel must extend that far
            back (this function does not raise if it doesn't; it just uses
            however much history is available, which may make the first
            fit noisier than intended).
        end: Last date in scope; final hold period truncated to it.
        lookback_months: Trailing window used to fit mean/cov at each
            rebalance.
        rebalance_months: Holding period before weights are re-fit.
        risk_free_rate: Annual risk-free rate for the tangency objective.
        portfolio_kind: ``"tangency"`` (max-Sharpe) or ``"min_variance"``.
        periods_per_year: Trading days/year used to annualize mean/cov for
            the fit (238-252 typical; does not affect the realized OOS
            returns, only the weight-selection objective).

    Returns:
        Dict with ``net_returns`` (concatenated realized OOS return series),
        ``periods`` (per-period fit/hold window dates and chosen weights),
        and ``symbols``.
    """
    if portfolio_kind not in {"tangency", "min_variance"}:
        raise ValueError("portfolio_kind must be 'tangency' or 'min_variance'")
    if lookback_months < 3:
        raise ValueError("lookback_months must be >= 3")
    if rebalance_months < 1:
        raise ValueError("rebalance_months must be >= 1")
    if start >= end:
        raise ValueError("start must be before end")

    panel = prices[symbols].sort_index().dropna()
    if len(panel) < 60:
        raise ValueError("Insufficient joint history across symbols")

    period_returns: List[pd.Series] = []
    periods: List[Dict[str, Any]] = []

    anchor = start
    while True:
        hold_start = anchor
        hold_end = min(anchor + pd.DateOffset(months=rebalance_months), end)
        if hold_start >= end or hold_end <= hold_start:
            break

        fit_start = hold_start - pd.DateOffset(months=lookback_months)
        fit_returns = panel.loc[(panel.index >= fit_start) & (panel.index < hold_start)]
        fit_returns = fit_returns.pct_change().dropna()

        hold_index = panel.index[(panel.index >= hold_start) & (panel.index < hold_end)]
        if len(fit_returns) >= 40:
            mean_ret = (fit_returns.mean() * periods_per_year).to_numpy()
            cov = (fit_returns.cov() * periods_per_year).to_numpy()
            if portfolio_kind == "tangency":
                fit = find_tangency_portfolio(mean_ret, cov, risk_free_rate)
            else:
                fit = find_min_variance_portfolio(mean_ret, cov)
            w = np.asarray(fit["weights"], dtype=float)
        else:
            w = np.full(len(symbols), 1.0 / len(symbols))
        weights_dict = dict(zip(symbols, w.tolist(), strict=True))

        # pct_change on the panel up through hold_end (not just the hold slice)
        # so the first hold day's return is relative to the prior calendar day
        # (causal, already-known information) instead of NaN.
        hold_returns = panel.loc[panel.index <= hold_end].pct_change().loc[hold_index]

        realized = []
        cur_w = w.copy()
        for dt in hold_index:
            r = hold_returns.loc[dt].reindex(symbols).fillna(0.0).to_numpy()
            realized.append(float(np.dot(cur_w, r)))
            cur_w = cur_w * (1.0 + r)
            s = cur_w.sum()
            if s > 0:
                cur_w = cur_w / s

        period_returns.append(pd.Series(realized, index=hold_index))
        periods.append(
            {
                "fit_start": str(pd.Timestamp(fit_start).date()),
                "hold_start": str(pd.Timestamp(hold_start).date()),
                "hold_end": str(pd.Timestamp(hold_end).date()),
                "fit_n_obs": int(len(fit_returns)),
                "weights": weights_dict,
            }
        )

        anchor = anchor + pd.DateOffset(months=rebalance_months)

    net_returns = (
        pd.concat(period_returns).sort_index() if period_returns else pd.Series(dtype=float)
    )
    logger.info(
        "walk-forward %s: symbols=%d periods=%d total_days=%d",
        portfolio_kind,
        len(symbols),
        len(periods),
        len(net_returns),
    )
    return {"net_returns": net_returns, "periods": periods, "symbols": symbols}
