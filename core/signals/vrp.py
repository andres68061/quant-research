"""Variance Risk Premium (VRP) proxy signal.

Mathematical framework
----------------------
Under a general stochastic-volatility model for the stock price S_t:

    dS_t / S_t = μ_t dt + σ_t dW_t

By Itô's lemma on ln S_t, the *quadratic variation* (integrated variance) over
[t, t+T] is:

    QV_{t,t+T} = ∫_t^{t+T} σ_s² ds

A *variance swap* written at t pays QV_{t,t+T} - K_var at T, where the
fair strike under no-arbitrage equals the risk-neutral expectation:

    K_var = E^Q_t[QV_{t,t+T}]

The CBOE VIX index is the square root of a model-free estimate of K_var for a
21-trading-day horizon (scaled to annualised %, ×100):

    VIX_t² / 100² ≈ E^Q_t[QV_{t,t+21}] × (252 / 21)   [annualised]

The **Variance Risk Premium** (Carr & Wu 2009, Bollerslev et al. 2009) is:

    VRP_t = E^Q_t[RV] - E^P_t[RV]
          ≈ (VIX_t / 100)² - RV_{t-W, t}

where RV_{t-W,t} is the backward-looking annualised realised variance:

    RV_{t-W,t} = (252 / W) × Σ_{i=1}^{W} r_{t-i+1}²

**Sign interpretation:**
    VRP_t > 0  →  options market priced in more variance than realised
                  (investors paid a premium to hedge variance risk)
    VRP_t < 0  →  realised variance exceeded what the market expected
                  (volatility surprise; typically follows market crashes)

**Equity-return prediction (BTZ 2009):**
High VRP signals high compensation for bearing variance risk.  Bollerslev,
Tauchen & Zhou (2009) show VRP explains ~8-12 % of S&P 500 return variation
at the 1-3 month horizon in a regression, controlling for the dividend yield,
term spread, and default spread.

References
----------
Carr, P. & Wu, L. (2009). Variance Risk Premiums. Review of Financial
    Studies 22(3), 1311-1341.
Bollerslev, T., Tauchen, G. & Zhou, H. (2009). Expected Stock Returns and
    Variance Risk Premia. Review of Financial Studies 22(11), 4463-4492.
Drechsler, I. & Yaron, A. (2011). What's Vol Got to Do with It. Review of
    Financial Studies 24(1), 1-45.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS: int = 252
_EPS: float = 1e-10


def compute_realized_variance(
    market_returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Compute rolling annualised realised variance from daily arithmetic returns.

    Formula:
        RV_t = (252 / window) × Σ_{i=1}^{window} r_{t-i+1}²

    This is the standard close-to-close estimator used in the VRP literature.
    For a 21-day window it corresponds to one calendar month of trading data.

    Args:
        market_returns: Daily arithmetic returns as decimals (e.g. 0.01 = 1 %).
            Index must be a DatetimeIndex; tz-aware is accepted and stripped.
        window: Rolling look-back window in trading days.
            Default 21 matches the VIX 30-day horizon (≈21 trading days).

    Returns:
        Series named ``"realized_variance"`` (annualised, decimal² units).
        The first ``window - 1`` observations are NaN.

    Raises:
        ValueError: If ``window`` is not positive.

    Example:
        >>> import numpy as np, pandas as pd
        >>> r = pd.Series(np.full(25, 0.01), index=pd.bdate_range("2020-01-01", periods=25))
        >>> rv = compute_realized_variance(r, window=21)
        >>> abs(rv.iloc[-1] - 252 * 0.01**2) < 1e-10
        True
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    ret = _strip_tz(market_returns.astype(float)).sort_index()
    rv = (ret**2).rolling(window=window, min_periods=window).sum() * (_TRADING_DAYS / window)
    rv.name = "realized_variance"

    logger.debug(
        "RV (window=%d): %d non-NaN observations, mean=%.6f",
        window,
        rv.notna().sum(),
        rv.mean(),
    )
    return rv


def compute_vrp_proxy(
    vix: pd.Series,
    market_returns: pd.Series,
    rv_window: int = 21,
) -> pd.Series:
    """Compute the VRP proxy: VIX-implied variance minus realised variance.

    VRP_t ≈ (VIX_t / 100)² − RV_{t−rv_window, t}

    Both quantities are annualised (decimal² units, i.e. variance not vol).
    The two series are inner-joined on their date index before subtraction.

    Args:
        vix: Daily VIX close level in percentage points (e.g. 20.0 = 20 %).
        market_returns: Daily arithmetic returns of the equity index.
        rv_window: Look-back window for realised variance (default 21 days).

    Returns:
        Series named ``"vrp_proxy"``.  Positive means implied > realised
        (variance risk premium exists); negative means realised > implied
        (variance surprise).

    Example:
        >>> vix = pd.Series([20.0] * 30, index=pd.bdate_range("2020-01-01", periods=30))
        >>> r = pd.Series([0.0] * 30, index=pd.bdate_range("2020-01-01", periods=30))
        >>> vrp = compute_vrp_proxy(vix, r)
        >>> (vrp.dropna() > 0).all()
        True
    """
    vix_clean = _strip_tz(vix.astype(float)).sort_index()
    implied_var = (vix_clean / 100.0) ** 2  # annualised decimal variance

    rv = compute_realized_variance(market_returns, window=rv_window)

    aligned = pd.concat(
        [implied_var.rename("implied_var"), rv],
        axis=1,
    ).dropna()

    vrp = (aligned["implied_var"] - aligned["realized_variance"]).rename("vrp_proxy")

    pct_positive = float((vrp > 0).mean()) * 100
    logger.info(
        "VRP proxy: %d observations, mean=%.5f, pct_positive=%.1f%%",
        len(vrp),
        float(vrp.mean()),
        pct_positive,
    )
    return vrp


def vrp_exposure(
    vrp: pd.Series,
    mode: str = "binary",
    quantile_window: int = 252,
) -> pd.Series:
    """Convert the VRP proxy into a portfolio exposure signal ∈ [0, 1].

    The output is shifted forward by **1 business day** to ensure full
    causality: today's VRP generates *tomorrow's* position.

    Args:
        vrp: Output of :func:`compute_vrp_proxy`.
        mode: Signal construction method:
            ``"binary"``     — 1.0 when VRP > 0, 0.0 when VRP ≤ 0.
            ``"continuous"`` — rolling z-score of VRP passed through the
                               standard normal CDF → continuous [0, 1].
                               More conviction when VRP is unusually high.
        quantile_window: Rolling window (trading days) for z-score in
            ``"continuous"`` mode.  Requires ``min_periods=30``.

    Returns:
        Series named ``"vrp_exposure"`` with values in ``[0, 1]``,
        shifted 1 business day forward.

    Raises:
        ValueError: If ``mode`` is not ``"binary"`` or ``"continuous"``.
        ValueError: If ``quantile_window`` is not positive.

    Example:
        >>> vrp = pd.Series(
        ...     [0.01, -0.005, 0.02],
        ...     index=pd.bdate_range("2020-01-01", periods=3),
        ... )
        >>> vrp_exposure(vrp, mode="binary").dropna().tolist()
        [1.0, 0.0, 1.0]
    """
    if mode not in ("binary", "continuous"):
        raise ValueError(f"mode must be 'binary' or 'continuous', got {mode!r}")
    if quantile_window <= 0:
        raise ValueError(f"quantile_window must be positive, got {quantile_window}")

    vrp_clean = _strip_tz(vrp.astype(float)).sort_index()

    if mode == "binary":
        raw = (vrp_clean > 0).astype(float)
    else:  # continuous
        rolling_mean = vrp_clean.rolling(window=quantile_window, min_periods=30).mean()
        rolling_std = vrp_clean.rolling(window=quantile_window, min_periods=30).std()
        z_score = (vrp_clean - rolling_mean) / (rolling_std + _EPS)
        from scipy.stats import norm  # local import to keep top-level deps light

        raw = pd.Series(norm.cdf(z_score.values), index=z_score.index, dtype=float)

    # 1-day forward shift: signal generated at t is executed at open on t+1
    exposure = raw.shift(1)
    exposure.name = "vrp_exposure"
    return exposure


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_tz(series: pd.Series) -> pd.Series:
    """Return series with a tz-naive DatetimeIndex (no-op if already naive)."""
    if getattr(series.index, "tz", None) is not None:
        series = series.copy()
        series.index = series.index.tz_localize(None)
    return series
