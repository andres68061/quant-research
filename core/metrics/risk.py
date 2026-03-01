"""
Risk metrics: Value at Risk (VaR) and Conditional VaR (CVaR).

Provides three VaR methodologies:
1. Historical VaR -- non-parametric, uses actual past returns
2. Parametric VaR -- assumes normal distribution
3. Monte Carlo VaR -- simulation-based
"""

from typing import Dict

import numpy as np
from scipy import stats as scipy_stats


def calculate_historical_var(
    returns: np.ndarray,
    confidence: int = 95,
) -> Dict[str, float]:
    """
    Historical VaR and CVaR from actual past returns.

    Args:
        returns: Array of periodic returns
        confidence: Confidence level as integer (e.g. 95 for 95%)

    Returns:
        Dictionary with 'var' and 'cvar' as positive percentages
    """
    cutoff = np.percentile(returns, 100 - confidence)
    var = -cutoff * 100
    tail = returns[returns <= cutoff]
    cvar = -tail.mean() * 100 if len(tail) > 0 else var
    return {"var": float(var), "cvar": float(cvar)}


def calculate_parametric_var(
    returns: np.ndarray,
    confidence: int = 95,
) -> Dict[str, float]:
    """
    Parametric (Gaussian) VaR and CVaR.

    Assumes returns are normally distributed.

    Args:
        returns: Array of periodic returns
        confidence: Confidence level as integer (e.g. 95 for 95%)

    Returns:
        Dictionary with 'var' and 'cvar' as positive percentages
    """
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1 - confidence / 100
    z_score = scipy_stats.norm.ppf(alpha)
    var = -(mu + z_score * sigma) * 100
    cvar = -(mu - sigma * scipy_stats.norm.pdf(z_score) / alpha) * 100
    return {"var": float(var), "cvar": float(cvar)}


def calculate_monte_carlo_var(
    returns: np.ndarray,
    confidence: int = 95,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Monte Carlo VaR and CVaR via simulated returns.

    Samples from a Gaussian fitted to the empirical return distribution.

    Args:
        returns: Array of periodic returns
        confidence: Confidence level as integer (e.g. 95 for 95%)
        n_simulations: Number of simulation draws
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'var' and 'cvar' as positive percentages
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean()
    sigma = returns.std()
    mc_returns = rng.normal(mu, sigma, n_simulations)
    cutoff = np.percentile(mc_returns, 100 - confidence)
    var = -cutoff * 100
    tail = mc_returns[mc_returns <= cutoff]
    cvar = -tail.mean() * 100 if len(tail) > 0 else var
    return {"var": float(var), "cvar": float(cvar)}


def calculate_all_var(
    returns: np.ndarray,
    confidence: int = 95,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute all three VaR methodologies at once.

    Args:
        returns: Array of periodic returns
        confidence: Confidence level (e.g. 95)
        n_simulations: Monte Carlo draws
        seed: Random seed

    Returns:
        Dictionary keyed by method name, each containing 'var' and 'cvar'

    Example:
        >>> results = calculate_all_var(portfolio_returns.values, confidence=95)
        >>> results['historical']['var']  # Historical VaR in %
    """
    return {
        "historical": calculate_historical_var(returns, confidence),
        "parametric": calculate_parametric_var(returns, confidence),
        "monte_carlo": calculate_monte_carlo_var(
            returns, confidence, n_simulations, seed
        ),
    }
