"""Factor-model alpha regression with Newey-West (HAC) standard errors.

Answers the question a raw Sharpe ratio cannot: after controlling for
exposure to known factor premia (market, size, value, profitability,
investment), does the strategy earn anything on its own? The deliverable
sentence is "alpha of Z bps/day vs FF5, t = X with Newey-West lags = L",
not "the strategy returned Y%".

Daily strategy returns are serially correlated and heteroskedastic, so
plain OLS standard errors overstate significance; Newey-West (HAC)
standard errors are the standard correction.

References:
    Fama & French (2015), "A Five-Factor Asset Pricing Model", JFE 116(1).
    Newey & West (1987), "A Simple, Positive Semi-definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
    Econometrica 55(3).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from core.exceptions import DataSchemaError

__all__ = ["regress_alpha_on_factors", "default_hac_lags"]

FF5_FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rmw", "cma"]


def default_hac_lags(n_obs: int) -> int:
    """Newey-West truncation lag by the standard rule of thumb.

    ``floor(4 * (n/100)^(2/9))`` (Newey & West 1994 plug-in choice) —
    about 6 lags for a few years of daily data.

    Args:
        n_obs: Number of return observations.

    Returns:
        Lag count (>= 1).
    """
    if n_obs < 2:
        return 1
    return max(1, int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def regress_alpha_on_factors(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factor_columns: Optional[Sequence[str]] = None,
    rf: Optional[pd.Series] = None,
    hac_lags: Optional[int] = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Regress strategy returns on factor returns with HAC inference.

    Fits ``r_strategy - rf = alpha + sum(beta_i * factor_i) + eps`` by OLS
    and reports Newey-West t-statistics. Long-short factor portfolios are
    already zero-investment, so ``rf`` is typically omitted for them and
    supplied for long-only strategies.

    Args:
        strategy_returns: Daily (or other periodic) decimal returns of the
            strategy. Index is aligned with ``factor_returns`` by date;
            tz-aware indexes are compared to naive FF dates by localizing
            the naive side is NOT done here — pass indexes already
            reconciled per the repo timezone convention.
        factor_returns: DataFrame of decimal factor returns (e.g. the FF5
            panel with columns ``mkt_rf, smb, hml, rmw, cma``).
        factor_columns: Subset of ``factor_returns`` columns to regress on
            (default: the FF5 columns present in the frame).
        rf: Optional risk-free series (decimal, same frequency) subtracted
            from ``strategy_returns`` before regressing.
        hac_lags: Newey-West truncation lag. Default: rule-of-thumb
            ``default_hac_lags(n)``.
        periods_per_year: For annualizing alpha (252 for daily).

    Returns:
        Dict with ``alpha`` (per-period decimal), ``alpha_bps_per_period``,
        ``alpha_ann_pct``, ``alpha_tstat``, ``alpha_pvalue``, ``betas``
        ({factor: coef}), ``beta_tstats``, ``r_squared``, ``n_obs``,
        ``hac_lags``.

    Raises:
        DataSchemaError: If no factor columns are usable or the aligned
            sample has fewer than 30 observations.
    """
    import statsmodels.api as sm

    if factor_columns is None:
        factor_columns = [c for c in FF5_FACTOR_COLUMNS if c in factor_returns.columns]
        if not factor_columns:
            factor_columns = list(factor_returns.columns)
    missing = [c for c in factor_columns if c not in factor_returns.columns]
    if missing:
        raise DataSchemaError(f"factor_returns missing columns: {missing}")

    y = strategy_returns.dropna()
    if rf is not None:
        y = (y - rf.reindex(y.index)).dropna()

    x = factor_returns[list(factor_columns)].dropna()
    common = y.index.intersection(x.index)
    if len(common) < 30:
        raise DataSchemaError(
            f"Only {len(common)} overlapping observations between strategy and "
            "factor returns; need at least 30 for HAC inference"
        )
    y = y.loc[common]
    x = x.loc[common]

    lags = hac_lags if hac_lags is not None else default_hac_lags(len(common))

    design = sm.add_constant(x.to_numpy(dtype=float))
    model = sm.OLS(y.to_numpy(dtype=float), design)
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    alpha = float(fit.params[0])
    return {
        "alpha": alpha,
        "alpha_bps_per_period": alpha * 1e4,
        "alpha_ann_pct": alpha * periods_per_year * 100.0,
        "alpha_tstat": float(fit.tvalues[0]),
        "alpha_pvalue": float(fit.pvalues[0]),
        "betas": {c: float(b) for c, b in zip(factor_columns, fit.params[1:], strict=True)},
        "beta_tstats": {c: float(t) for c, t in zip(factor_columns, fit.tvalues[1:], strict=True)},
        "r_squared": float(fit.rsquared),
        "n_obs": int(len(common)),
        "hac_lags": int(lags),
    }
