"""Probabilistic and Deflated Sharpe Ratio (Bailey & López de Prado).

The Probabilistic Sharpe Ratio (PSR) answers: given the estimated Sharpe,
the sample length, and the non-normality of the returns (skew, kurtosis),
what is the probability that the *true* Sharpe exceeds a benchmark level?
The Deflated Sharpe Ratio (DSR) is the PSR evaluated against the Sharpe
one would expect the *best of N trials* to show under the null of zero
skill — the formal correction for "we tried several configurations and
are reporting the best one" (selection bias under multiple testing).

References:
    Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier",
    Journal of Risk 15(2) — PSR.
    Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting and Non-Normality", Journal of
    Portfolio Management 40(5) — DSR.

All Sharpe ratios in this module are **per-period** (same frequency as the
returns series), NOT annualized. Convert an annualized Sharpe to per-period
by dividing by ``sqrt(periods_per_year)`` before passing it in.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "calculate_probabilistic_sharpe_ratio",
    "expected_max_sharpe_under_null",
    "calculate_deflated_sharpe_ratio",
]

_EULER_MASCHERONI = 0.5772156649015329
_EPS = 1e-10


def _per_period_sharpe(values: np.ndarray) -> float:
    std = values.std(ddof=1)
    if std < _EPS:
        return 0.0
    return float(values.mean() / std)


def calculate_probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sharpe: float = 0.0
) -> float:
    """
    Probability that the true (per-period) Sharpe exceeds ``benchmark_sharpe``.

    PSR = Phi( (SR_hat - SR*) * sqrt(n - 1)
               / sqrt(1 - g3*SR_hat + (g4 - 1)/4 * SR_hat^2) )

    where ``SR_hat`` is the per-period sample Sharpe, ``g3`` the sample
    skewness, ``g4`` the sample kurtosis (Pearson, normal = 3), and ``n``
    the number of observations. Negative skew and fat tails widen the
    Sharpe estimator's sampling distribution and therefore *lower* the PSR
    at the same point estimate — the reason a short, lucky, negatively
    skewed track record should not be trusted at face value.

    Args:
        returns: Series of periodic returns.
        benchmark_sharpe: Per-period Sharpe to beat (0.0 = "any skill at all").

    Returns:
        PSR in [0, 1]; 0.0 when fewer than 3 observations or zero variance.
    """
    values = returns.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 3:
        return 0.0
    sr = _per_period_sharpe(values)
    if sr == 0.0 and values.std(ddof=1) < _EPS:
        return 0.0
    g3 = float(stats.skew(values))
    g4 = float(stats.kurtosis(values, fisher=False))
    denom_sq = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denom_sq < _EPS:
        return 0.0
    z = (sr - benchmark_sharpe) * np.sqrt(n - 1.0) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def expected_max_sharpe_under_null(n_trials: int, trial_sharpe_variance: float) -> float:
    """
    Expected maximum per-period Sharpe across ``n_trials`` independent
    trials when every trial's true Sharpe is zero (pure selection luck).

    E[max SR] ≈ sqrt(V) * ( (1 - γ) * Phi^{-1}(1 - 1/N)
                            + γ * Phi^{-1}(1 - 1/(N e)) )

    where ``V`` is the cross-trial variance of the estimated Sharpes and γ
    is the Euler–Mascheroni constant. This is the benchmark the best
    trial's Sharpe must clear before it is evidence of anything.

    Args:
        n_trials: Number of strategy configurations tried (N >= 1).
        trial_sharpe_variance: Variance of the per-period Sharpe estimates
            across those trials.

    Returns:
        Expected max per-period Sharpe under the null (0.0 when N < 2 or
        variance is 0 — a single trial needs no deflation).
    """
    if n_trials < 2 or trial_sharpe_variance <= 0.0:
        return 0.0
    n = float(n_trials)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(
        np.sqrt(trial_sharpe_variance) * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)
    )


def calculate_deflated_sharpe_ratio(
    returns: pd.Series, trial_sharpes: Sequence[float]
) -> dict[str, Any]:
    """
    Deflated Sharpe Ratio: PSR of ``returns`` against the expected max
    Sharpe of the whole trial family under the zero-skill null.

    ``trial_sharpes`` must contain the **per-period** Sharpe of every
    configuration tried in the family — including failed ones and including
    the reported one. Under-counting trials overstates the DSR; when in
    doubt, count more trials, not fewer.

    Args:
        returns: Periodic returns of the *selected* (best) configuration.
        trial_sharpes: Per-period Sharpe estimates of all N trials.

    Returns:
        Dict with ``dsr`` (probability the selected config has real skill
        after the multiple-testing correction), ``sharpe_per_period``,
        ``expected_max_sharpe`` (the null benchmark), and ``n_trials``.
    """
    values = returns.dropna().to_numpy(dtype=float)
    sr_hat = _per_period_sharpe(values) if len(values) >= 2 else 0.0
    trials = np.asarray(list(trial_sharpes), dtype=float)
    n_trials = len(trials)
    variance = float(trials.var(ddof=1)) if n_trials >= 2 else 0.0
    sr_star = expected_max_sharpe_under_null(n_trials, variance)
    dsr = calculate_probabilistic_sharpe_ratio(returns, benchmark_sharpe=sr_star)
    return {
        "dsr": dsr,
        "sharpe_per_period": sr_hat,
        "expected_max_sharpe": sr_star,
        "n_trials": n_trials,
    }
