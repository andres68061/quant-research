"""
Performance metrics calculation for portfolio analysis.

This module provides functions to calculate various portfolio performance metrics
including risk-adjusted returns, drawdowns, and statistical measures.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


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


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
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
    periods_per_year: int = 252
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
    
    sortino = (
        excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
    )
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
    periods_per_year: int = 252
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
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        returns: Series of periodic returns
        benchmark_returns: Optional benchmark returns for relative metrics
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of periods per year
        
    Returns:
        Dictionary of performance metrics
        
    Example:
        >>> metrics = calculate_performance_metrics(portfolio_returns)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }
    
    # Basic metrics
    total_return = (1 + returns_clean).prod() - 1
    n_periods = len(returns_clean)
    ann_return = returns_clean.mean() * periods_per_year
    ann_vol = returns_clean.std() * np.sqrt(periods_per_year)
    
    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(returns_clean, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns_clean, risk_free_rate, periods_per_year)
    max_dd = calculate_max_drawdown(returns_clean)
    calmar = calculate_calmar_ratio(returns_clean, periods_per_year)
    
    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "n_periods": int(n_periods),
    }
    
    # Add benchmark-relative metrics if provided
    if benchmark_returns is not None:
        benchmark_clean = benchmark_returns.reindex(returns_clean.index).fillna(0)
        ir = calculate_information_ratio(
            returns_clean, benchmark_clean, periods_per_year
        )
        
        # Beta calculation
        cov = returns_clean.cov(benchmark_clean)
        var = benchmark_clean.var()
        beta = cov / var if var != 0 else 0.0
        
        # Alpha calculation (CAPM)
        benchmark_ann_return = benchmark_clean.mean() * periods_per_year
        alpha = ann_return - (risk_free_rate + beta * (benchmark_ann_return - risk_free_rate))
        
        metrics["information_ratio"] = float(ir)
        metrics["beta"] = float(beta)
        metrics["alpha"] = float(alpha)
    
    return metrics


def format_performance_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Format performance metrics as a nice table for display.
    
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
        "information_ratio": "Information Ratio",
        "beta": "Beta",
        "alpha": "Alpha",
        "n_periods": "Number of Periods",
    }
    
    rows = []
    for key, value in metrics.items():
        if key in display_names:
            # Format as percentage for returns and drawdown
            if "return" in key or "drawdown" in key or "alpha" in key:
                formatted_value = f"{value * 100:.2f}%"
            # Format ratios with 2 decimals
            elif "ratio" in key or "beta" in key:
                formatted_value = f"{value:.2f}"
            # Format counts as integers
            elif key == "n_periods":
                formatted_value = f"{int(value)}"
            else:
                formatted_value = f"{value:.4f}"
            
            rows.append({"Metric": display_names[key], "Value": formatted_value})
    
    return pd.DataFrame(rows)

