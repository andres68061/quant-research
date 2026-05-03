"""Performance metrics calculation for portfolio analysis.

This module provides functions to calculate portfolio performance metrics
including returns, downside risk, drawdowns, and statistical measures.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from core.metrics.risk import calculate_historical_var


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns from a return series.

    Args:
        returns: Series of periodic returns

    Returns:
        Series of cumulative returns (wealth index)

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> cum_returns = calculate_cumulative_returns(returns)
    """
    return (1 + returns).cumprod()


def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from returns.

    Args:
        returns: Series of periodic returns

    Returns:
        Series of drawdowns (negative values)
    """
    cum_returns = calculate_cumulative_returns(returns)
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from returns.

    Args:
        returns: Series of periodic returns

    Returns:
        Maximum drawdown as a decimal (negative value)
    """
    drawdown = calculate_drawdown(returns)
    return float(drawdown.min())


def calculate_time_underwater(returns: pd.Series) -> int:
    """Calculate the longest drawdown duration in periods.

    Args:
        returns: Series of periodic returns.

    Returns:
        Longest streak where cumulative wealth is below its running peak.

    Example:
        >>> calculate_time_underwater(pd.Series([0.1, -0.1, 0.0, 0.2]))
        2
    """
    returns_clean = returns.dropna()
    if returns_clean.empty:
        return 0

    cumulative_returns = calculate_cumulative_returns(returns_clean)
    running_max = cumulative_returns.expanding().max()
    underwater = cumulative_returns < running_max

    longest = 0
    current = 0
    for is_underwater in underwater:
        if is_underwater:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def calculate_loss_probability(
    returns: pd.Series,
    horizon_days: int,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    block_length: int = 21,
) -> float:
    """Estimate probability of losing money over a horizon via block bootstrap.

    Args:
        returns: Series of periodic returns as decimals.
        horizon_days: Number of return observations in each simulated path.
        n_bootstrap: Number of bootstrap paths.
        seed: Random seed for reproducibility.
        block_length: Average sampled block length in observations.

    Returns:
        Probability that the bootstrapped cumulative return is below zero.

    Example:
        >>> r = pd.Series([0.01, -0.02, 0.005, 0.004])
        >>> 0.0 <= calculate_loss_probability(r, 2, n_bootstrap=100, seed=1) <= 1.0
        True
    """
    returns_clean = returns.dropna().astype(float)
    if returns_clean.empty or horizon_days <= 0 or n_bootstrap <= 0:
        return 0.0

    values = returns_clean.to_numpy()
    rng = np.random.default_rng(seed)
    losses = 0
    block_length = max(1, int(block_length))

    for _ in range(n_bootstrap):
        sampled = []
        while len(sampled) < horizon_days:
            start = int(rng.integers(0, len(values)))
            length = int(rng.geometric(1.0 / block_length))
            positions = (start + np.arange(length)) % len(values)
            sampled.extend(values[positions].tolist())

        path = np.array(sampled[:horizon_days], dtype=float)
        cumulative_return = float(np.prod(1.0 + path) - 1.0)
        losses += cumulative_return < 0.0

    return float(losses / n_bootstrap)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0.0
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio (uses downside deviation).

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    max_dd = abs(calculate_max_drawdown(returns))
    if max_dd == 0:
        return 0.0

    ann_return = returns.mean() * periods_per_year
    calmar = ann_return / max_dd
    return float(calmar)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate information ratio (excess return / tracking error).

    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Information ratio
    """
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)

    if tracking_error == 0:
        return 0.0

    ir = (excess_returns.mean() * periods_per_year) / tracking_error
    return float(ir)


def calculate_performance_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    loss_probability_horizons: Optional[Sequence[int]] = None,
    loss_probability_bootstraps: int = 10_000,
    loss_probability_seed: int = 42,
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Series of periodic returns
        benchmark_returns: Optional benchmark returns for relative metrics
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of periods per year
        loss_probability_horizons: Optional horizons, in observations, for
            bootstrapped loss probability estimates.
        loss_probability_bootstraps: Number of bootstrap paths per horizon.
        loss_probability_seed: Random seed for bootstrap reproducibility.

    Returns:
        Dictionary of performance metrics

    Example:
        >>> metrics = calculate_performance_metrics(portfolio_returns)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    returns_clean = returns.dropna()

    if len(returns_clean) == 0:
        empty_metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "cvar_95": 0.0,
            "cvar_99": 0.0,
            "time_underwater_days": 0,
        }
        if loss_probability_horizons is not None:
            for horizon_days in loss_probability_horizons:
                empty_metrics[f"loss_probability_{int(horizon_days)}d"] = 0.0
        return empty_metrics

    total_return = (1 + returns_clean).prod() - 1
    ann_return = returns_clean.mean() * periods_per_year
    ann_vol = returns_clean.std() * np.sqrt(periods_per_year)

    sharpe = calculate_sharpe_ratio(returns_clean, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns_clean, risk_free_rate, periods_per_year)
    max_dd = calculate_max_drawdown(returns_clean)
    calmar = calculate_calmar_ratio(returns_clean, periods_per_year)
    cvar_95 = calculate_historical_var(returns_clean.to_numpy(), confidence=95)["cvar"] / 100
    cvar_99 = calculate_historical_var(returns_clean.to_numpy(), confidence=99)["cvar"] / 100
    time_underwater_days = calculate_time_underwater(returns_clean)

    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "cvar_95": float(cvar_95),
        "cvar_99": float(cvar_99),
        "time_underwater_days": int(time_underwater_days),
        "n_periods": int(len(returns_clean)),
    }

    if loss_probability_horizons is not None:
        for horizon_days in loss_probability_horizons:
            key = f"loss_probability_{int(horizon_days)}d"
            metrics[key] = calculate_loss_probability(
                returns_clean,
                horizon_days=int(horizon_days),
                n_bootstrap=loss_probability_bootstraps,
                seed=loss_probability_seed,
            )

    if benchmark_returns is not None:
        benchmark_clean = benchmark_returns.reindex(returns_clean.index).fillna(0)
        ir = calculate_information_ratio(returns_clean, benchmark_clean, periods_per_year)

        cov = returns_clean.cov(benchmark_clean)
        var = benchmark_clean.var()
        beta = cov / var if var != 0 else 0.0

        benchmark_ann_return = benchmark_clean.mean() * periods_per_year
        alpha = ann_return - (risk_free_rate + beta * (benchmark_ann_return - risk_free_rate))

        metrics["information_ratio"] = float(ir)
        metrics["beta"] = float(beta)
        metrics["alpha"] = float(alpha)

    return metrics


def format_performance_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Format performance metrics as a table for display.

    Args:
        metrics: Dictionary of performance metrics

    Returns:
        DataFrame formatted for display
    """
    display_names = {
        "total_return": "Total Return",
        "annualized_return": "Annualized Return",
        "annualized_volatility": "Annualized Volatility",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "max_drawdown": "Max Drawdown",
        "calmar_ratio": "Calmar Ratio",
        "cvar_95": "CVaR 95%",
        "cvar_99": "CVaR 99%",
        "time_underwater_days": "Time Underwater (Days)",
        "information_ratio": "Information Ratio",
        "beta": "Beta",
        "alpha": "Alpha",
        "n_periods": "Number of Periods",
    }

    rows = []
    for key, value in metrics.items():
        if key.startswith("loss_probability_"):
            horizon = key.removeprefix("loss_probability_").removesuffix("d")
            display_name = f"Loss Probability ({horizon}d)"
        else:
            display_name = display_names.get(key)

        if display_name is not None:
            if (
                "return" in key
                or "drawdown" in key
                or "alpha" in key
                or "cvar" in key
                or "loss_probability" in key
            ):
                formatted_value = f"{value * 100:.2f}%"
            elif "ratio" in key or "beta" in key:
                formatted_value = f"{value:.2f}"
            elif key in {"n_periods", "time_underwater_days"}:
                formatted_value = f"{int(value)}"
            else:
                formatted_value = f"{value:.4f}"

            rows.append({"Metric": display_name, "Value": formatted_value})

    return pd.DataFrame(rows)
