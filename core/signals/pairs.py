"""Pairs trading: Engle–Granger cointegration and spread z-score signals.

Uses log-prices for the hedge regression (standard in the pairs literature).
Trading signals are built from a *rolling* hedge ratio and rolling z-score so
the backtest does not use a full-sample OLS fit (lookahead).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

_EPS = 1e-10

__all__ = [
    "align_pair_log_prices",
    "engle_granger_test",
    "rolling_hedge_ratio",
    "spread_from_hedge",
    "rolling_spread_zscore",
    "pairs_position_from_zscore",
]


def align_pair_log_prices(
    prices: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Extract and align two positive price series; return log-prices.

    Args:
        prices: Wide adj_close panel (date × symbol).
        symbol_y: Dependent leg of the hedge (spread long when undervalued).
        symbol_x: Independent / hedge leg.

    Returns:
        ``(log_y, log_x)`` aligned on the common non-NaN index.

    Raises:
        KeyError: If either symbol is missing.
        ValueError: If fewer than 2 overlapping observations remain.
    """
    if symbol_y not in prices.columns:
        raise KeyError(f"symbol_y '{symbol_y}' not in price panel")
    if symbol_x not in prices.columns:
        raise KeyError(f"symbol_x '{symbol_x}' not in price panel")

    pair = prices[[symbol_y, symbol_x]].astype(float).replace(0.0, np.nan).dropna()
    if len(pair) < 2:
        raise ValueError(f"Insufficient overlapping history for {symbol_y}/{symbol_x}")
    log_y = np.log(pair[symbol_y]).rename(symbol_y)
    log_x = np.log(pair[symbol_x]).rename(symbol_x)
    return log_y, log_x


def engle_granger_test(
    log_y: pd.Series,
    log_x: pd.Series,
    *,
    maxlag: Optional[int] = None,
) -> dict[str, float]:
    """
    Engle–Granger two-step cointegration test on log-prices.

    Step 1: OLS ``log_y = a + b * log_x + e``.
    Step 2: ADF unit-root test on residual ``e`` (no constant in the ADF
    regression of the residual is the textbook EG convention via
    ``regression='c'`` on levels of e — we use ``regression='n'`` only when
    residuals are demeaned by construction of OLS with intercept; here we
    keep ``regression='c'`` for robustness).

    Returns:
        Dict with ``hedge_ratio``, ``intercept``, ``adf_stat``, ``adf_pvalue``,
        ``n_obs``.
    """
    aligned = pd.concat([log_y, log_x], axis=1).dropna()
    y = aligned.iloc[:, 0].astype(float)
    x = add_constant(aligned.iloc[:, 1].astype(float))
    fit = OLS(y, x).fit()
    residual = fit.resid
    adf_stat, adf_pvalue, *_ = adfuller(
        residual,
        maxlag=maxlag,
        autolag="AIC" if maxlag is None else None,
        regression="c",
    )
    return {
        "hedge_ratio": float(fit.params.iloc[1]),
        "intercept": float(fit.params.iloc[0]),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "n_obs": float(len(residual)),
    }


def rolling_hedge_ratio(
    log_y: pd.Series,
    log_x: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Rolling OLS hedge ratio ``b_t`` from ``log_y ~ a + b log_x`` over ``window``.

    NaN until ``window`` observations exist. Units: dimensionless elasticity
    of log_y to log_x.
    """
    if window < 10:
        raise ValueError("window must be >= 10")

    aligned = pd.concat([log_y.rename("y"), log_x.rename("x")], axis=1).dropna()
    y_vals = aligned["y"].to_numpy(dtype=float)
    x_vals = aligned["x"].to_numpy(dtype=float)
    betas = np.full(len(aligned), np.nan, dtype=float)

    for i in range(window - 1, len(aligned)):
        y_w = y_vals[i - window + 1 : i + 1]
        x_w = x_vals[i - window + 1 : i + 1]
        x_design = np.column_stack([np.ones(window), x_w])
        try:
            coef, *_ = np.linalg.lstsq(x_design, y_w, rcond=None)
            betas[i] = coef[1]
        except np.linalg.LinAlgError:
            betas[i] = np.nan

    return pd.Series(betas, index=aligned.index, name="hedge_ratio")


def spread_from_hedge(
    log_y: pd.Series,
    log_x: pd.Series,
    hedge_ratio: pd.Series,
) -> pd.Series:
    """
    Cointegrating residual proxy: ``spread = log_y - b * log_x``.

    Intercept is omitted (absorbed into the rolling z-score demeaning).
    """
    aligned = pd.concat(
        [log_y.rename("y"), log_x.rename("x"), hedge_ratio.rename("b")],
        axis=1,
    ).dropna()
    spread = aligned["y"] - aligned["b"] * aligned["x"]
    return spread.rename("spread")


def rolling_spread_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score of the spread.

        z_t = (s_t - mean_{t-w+1:t}(s)) / std_{t-w+1:t}(s)
    """
    if window < 5:
        raise ValueError("z-score window must be >= 5")
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std()
    return ((spread - mean) / (std + _EPS)).rename("spread_z")


def pairs_position_from_zscore(
    zscore: pd.Series,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """
    Map spread z-score to a discrete spread position in ``{-1, 0, +1}``.

    Convention (spread = log_y − b log_x):
      - ``+1`` long the spread (long y, short x) when z ≤ −entry_z
      - ``−1`` short the spread when z ≥ +entry_z
      - exit toward 0 when |z| ≤ exit_z
      - otherwise hold the previous position (hysteresis)

    The returned series is *causal for next-day PnL* when the caller shifts it
    by one day before multiplying by returns.
    """
    if entry_z <= exit_z:
        raise ValueError("entry_z must be strictly greater than exit_z")
    if entry_z <= 0 or exit_z < 0:
        raise ValueError("entry_z must be > 0 and exit_z >= 0")

    z = zscore.astype(float)
    positions = pd.Series(np.nan, index=z.index, dtype=float)
    pos = 0.0
    for ts, value in z.items():
        if not np.isfinite(value):
            positions.loc[ts] = np.nan
            continue
        if pos == 0.0:
            if value <= -entry_z:
                pos = 1.0
            elif value >= entry_z:
                pos = -1.0
        elif pos > 0.0 and value >= -exit_z:
            pos = 0.0
        elif pos < 0.0 and value <= exit_z:
            pos = 0.0
        positions.loc[ts] = pos
    return positions.rename("spread_position")
