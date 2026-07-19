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


def calculate_pain_index(returns: pd.Series) -> float:
    """
    Mean absolute drawdown over the period (a.k.a. Pain Index).

    Unlike ``calculate_max_drawdown`` (the single worst point), this
    integrates both the *depth* and *duration* of every drawdown: a
    strategy that spends a long time moderately underwater scores worse
    than one with the same max drawdown that recovers immediately. Days at
    a new high (drawdown = 0) contribute 0, so this is naturally weighted
    by how much of the period was spent how far underwater.

    Args:
        returns: Series of periodic returns.

    Returns:
        Mean |drawdown| as a decimal (>= 0).
    """
    drawdown = calculate_drawdown(returns)
    if drawdown.empty:
        return 0.0
    return float(drawdown.abs().mean())


def calculate_pain_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized return per unit of Pain Index — reward per unit of
    time-and-depth "pain", as opposed to Calmar's reward per unit of
    worst-single-drawdown.

    Args:
        returns: Series of periodic returns.
        periods_per_year: Number of periods per year.

    Returns:
        Pain ratio (0.0 if there is no drawdown to divide by).
    """
    pain = calculate_pain_index(returns)
    if pain == 0:
        return 0.0
    ann_return = returns.mean() * periods_per_year
    return float(ann_return / pain)


def calculate_ulcer_index(returns: pd.Series) -> float:
    """
    Root-mean-square drawdown (a.k.a. Ulcer Index).

    Same spirit as ``calculate_pain_index`` but squares each drawdown
    before averaging, so it penalizes deep drawdowns more than shallow
    ones of the same duration (Pain Index weights all depths linearly).

    Args:
        returns: Series of periodic returns.

    Returns:
        RMS drawdown as a decimal (>= 0).
    """
    drawdown = calculate_drawdown(returns)
    if drawdown.empty:
        return 0.0
    return float(np.sqrt((drawdown**2).mean()))


def calculate_martin_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized return per unit of Ulcer Index (a.k.a. Ulcer Performance Index).

    Args:
        returns: Series of periodic returns.
        periods_per_year: Number of periods per year.

    Returns:
        Martin ratio (0.0 if there is no drawdown to divide by).
    """
    ui = calculate_ulcer_index(returns)
    if ui == 0:
        return 0.0
    ann_return = returns.mean() * periods_per_year
    return float(ann_return / ui)


def calculate_cost_basis_pain(returns: pd.Series) -> float:
    """
    Sum of daily shortfall below the *initial investment* (cost basis),
    not the running peak: :math:`\\sum_t \\max(0,\\, 1 - w_t)` where
    :math:`w_t = \\prod_{i \\le t}(1 + r_i)` is wealth relative to a
    starting value of 1.0.

    Differs from ``calculate_pain_index`` in two ways: (1) the reference
    point is the original cost basis, not the running peak — a strategy
    that falls, recovers to a new high, then falls again only "hurts" here
    relative to day zero, not relative to its own prior peak; (2) this is
    a **sum** over the whole history, not a mean, so it grows with how
    long the series is (by design — paired with total, not annualized,
    return in ``calculate_cid1_ratio``).

    Args:
        returns: Series of periodic returns.

    Returns:
        Total below-cost-basis shortfall as a decimal (>= 0).
    """
    wealth = calculate_cumulative_returns(returns)
    if wealth.empty:
        return 0.0
    shortfall = (1.0 - wealth).clip(lower=0.0)
    return float(shortfall.sum())


def calculate_cid1_ratio(returns: pd.Series) -> float:
    """
    Total (compounded) return to date, divided by ``calculate_cost_basis_pain``.

    Not a Pain Ratio variant despite the shared denominator concept — both
    the numerator (total return, not annualized excess return) and the
    denominator (cost-basis shortfall, not mean drawdown-from-peak) differ
    from the classic definition, so it gets its own name rather than
    borrowing one that no longer applies.

    Deliberately not annualized and not based on the arithmetic daily mean
    (unlike every other ratio in this module): total return over the full
    elapsed history, divided by the cumulative below-cost-basis pain over
    that same history. Two paths with identical Sharpe can have very
    different scores here if one spent far longer below its starting
    value on the way to the same endpoint.

    Args:
        returns: Series of periodic returns.

    Returns:
        Cid-1 ratio (0.0 if the series never fell below its starting value).
    """
    pain = calculate_cost_basis_pain(returns)
    if pain == 0:
        return 0.0
    wealth = calculate_cumulative_returns(returns)
    total_return = float(wealth.iloc[-1] - 1.0) if not wealth.empty else 0.0
    return total_return / pain


def calculate_typical_period_return(returns: pd.Series, period_days: int = 252) -> float:
    """
    Average *compounded* return over sequential, non-overlapping blocks of
    ``period_days`` observations (e.g. 252 ≈ "typical 1-year return").

    This is deliberately different from ``annualized_return`` elsewhere in
    this module, which annualizes the arithmetic daily mean
    (``returns.mean() * periods_per_year``). This instead answers "what
    does a typical `period_days`-long holding actually return," compounding
    fully within each block and simple-averaging the block outcomes — no
    daily arithmetic mean, no annualization factor. A trailing partial
    block (fewer than ``period_days`` observations) is dropped.

    Args:
        returns: Series of periodic returns.
        period_days: Length of each block in observations (252 ≈ 1 year).

    Returns:
        Mean compounded per-block return as a decimal (0.0 if fewer than
        one full block is available).
    """
    if period_days < 1:
        raise ValueError("period_days must be >= 1")
    values = returns.to_numpy(dtype=float)
    n_blocks = len(values) // period_days
    if n_blocks == 0:
        return 0.0
    block_returns = [
        float(np.prod(1.0 + values[i * period_days : (i + 1) * period_days]) - 1.0)
        for i in range(n_blocks)
    ]
    return float(np.mean(block_returns))


def calculate_cid2_ratio(returns: pd.Series, period_days: int = 252) -> float:
    """
    ``calculate_typical_period_return`` divided by ``calculate_cost_basis_pain``.

    Same cost-basis-pain denominator as ``calculate_cid1_ratio``, but the
    numerator is "what a typical holding period actually returned" rather
    than the single total-return-to-date figure — useful when the question
    is "if I invest for about `period_days` days, what should I expect, per
    unit of pain endured getting there."

    Args:
        returns: Series of periodic returns.
        period_days: Length of each block in observations (252 ≈ 1 year).

    Returns:
        Cid-2 ratio (0.0 if there's no pain to divide by).
    """
    pain = calculate_cost_basis_pain(returns)
    if pain == 0:
        return 0.0
    typical = calculate_typical_period_return(returns, period_days)
    return typical / pain


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
            "pain_index": 0.0,
            "pain_ratio": 0.0,
            "ulcer_index": 0.0,
            "martin_ratio": 0.0,
            "cid1_ratio": 0.0,
            "typical_period_return": 0.0,
            "cid2_ratio": 0.0,
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
    pain_index = calculate_pain_index(returns_clean)
    pain_ratio = calculate_pain_ratio(returns_clean, periods_per_year)
    ulcer_index = calculate_ulcer_index(returns_clean)
    martin_ratio = calculate_martin_ratio(returns_clean, periods_per_year)
    cid1_ratio = calculate_cid1_ratio(returns_clean)
    typical_period_return = calculate_typical_period_return(returns_clean, periods_per_year)
    cid2_ratio = calculate_cid2_ratio(returns_clean, periods_per_year)
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
        "pain_index": float(pain_index),
        "pain_ratio": float(pain_ratio),
        "ulcer_index": float(ulcer_index),
        "martin_ratio": float(martin_ratio),
        "cid1_ratio": float(cid1_ratio),
        "typical_period_return": float(typical_period_return),
        "cid2_ratio": float(cid2_ratio),
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
        "pain_index": "Pain Index",
        "pain_ratio": "Pain Ratio",
        "ulcer_index": "Ulcer Index",
        "martin_ratio": "Martin Ratio",
        "cid1_ratio": "Cid-1 Ratio",
        "typical_period_return": "Typical Period Return",
        "cid2_ratio": "Cid-2 Ratio",
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
