"""Rolling-window primitives shared by the price/volume technical factor family.

These are the pieces pandas does not give directly: a rolling ordinary-least-squares
fit of a series against time (slope, R-squared, residual dispersion), the *position*
of the rolling extreme inside its window, and division/correlation helpers that
return NaN instead of infinity when a denominator collapses.

Every function here is **causal**: a value at row ``t`` is a function of rows
``t - window + 1 .. t`` only. Windows are never centred and never require a full
window to be completed in the future, so nothing computed here can leak.

Implementation note: the regression and argmax/argmin helpers use
``sliding_window_view`` rather than ``Rolling.apply``. The view is a strided
read-only view over the same buffer, so the window matrix costs one allocation for
the arithmetic result rather than one Python callback per row.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

EPS = 1e-10

TREND_COLUMNS: tuple[str, ...] = ("trend_slope", "trend_rsqr", "trend_resid")


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    positive_only: bool = True,
) -> pd.Series:
    """
    Divide two aligned Series, returning NaN where the result is meaningless.

    Mirrors the house pattern in :mod:`core.data.factors.fundamental_factors`: a
    denominator at or below zero does not make a normalised factor "very large",
    it makes it undefined, and NaN is the honest answer.

    Args:
        numerator: Numerator Series.
        denominator: Denominator Series, aligned to ``numerator``.
        positive_only: Mask denominators <= 0 (default). Set False when only an
            exact zero is disqualifying and a negative denominator is meaningful.

    Returns:
        float64 Series aligned to the inputs.
    """
    denom = denominator.astype("float64")
    denom = denom.where(denom > EPS) if positive_only else denom.where(denom.abs() > EPS)
    return numerator.astype("float64") / denom


def _sliding_windows(series: pd.Series, window: int) -> np.ndarray | None:
    """Strided (n - window + 1, window) view of ``series``, or None if too short."""
    values = series.to_numpy(dtype="float64", copy=False)
    if values.size < window:
        return None
    return sliding_window_view(values, window)


def _empty_like(series: pd.Series) -> pd.Series:
    """All-NaN float64 Series on ``series``'s index."""
    return pd.Series(np.nan, index=series.index, dtype="float64")


def _pad_leading(tail_values: np.ndarray, length: int) -> np.ndarray:
    """Right-align ``tail_values`` in a NaN array of ``length`` (warm-up rows stay NaN)."""
    padded = np.full(length, np.nan, dtype="float64")
    padded[length - tail_values.size :] = tail_values
    return padded


def compute_rolling_trend(series: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling OLS fit of ``series`` against time within each trailing window.

    Fits ``y_i = a + b * i`` for ``i = 0 .. window - 1`` over the trailing window
    and reports the slope, the fit quality, and the dispersion left over. Together
    they separate "moved a lot" from "moved a lot in a straight line" — a trending
    name and a whipsawing name can share a return but never share an R-squared.

    Args:
        series: Level Series (typically ``adj_close``) for one symbol, in
            chronological order.
        window: Trailing window in trading days. Must be >= 3 for the residual
            leg, which uses ``window - 2`` degrees of freedom.

    Returns:
        DataFrame indexed like ``series`` with :data:`TREND_COLUMNS`:
        ``trend_slope`` (level change per day, in the units of ``series``),
        ``trend_rsqr`` (in [0, 1]), and ``trend_resid`` (residual standard
        deviation, same units as ``series``). Rows before the first complete
        window are NaN, as are windows containing any NaN.

    Raises:
        ValueError: If ``window`` < 3.
    """
    if window < 3:
        raise ValueError(f"compute_rolling_trend needs window >= 3, got {window}")

    result = pd.DataFrame(
        {name: _empty_like(series) for name in TREND_COLUMNS},
        index=series.index,
    )
    windows = _sliding_windows(series, window)
    if windows is None:
        return result

    positions = np.arange(window, dtype="float64")
    centred_positions = positions - positions.mean()
    sum_xx = float(centred_positions @ centred_positions)

    centred = windows - windows.mean(axis=1, keepdims=True)
    sum_xy = centred @ centred_positions
    sum_yy = np.einsum("ij,ij->i", centred, centred)

    slope = sum_xy / sum_xx
    with np.errstate(invalid="ignore", divide="ignore"):
        r_squared = np.where(sum_yy > EPS, sum_xy**2 / (sum_xx * sum_yy), np.nan)
        # A flat window is a perfect fit with zero slope, not an undefined one.
        r_squared = np.where((sum_yy <= EPS) & np.isfinite(sum_yy), 1.0, r_squared)
    residual_sum_squares = np.clip(sum_yy - slope * sum_xy, 0.0, None)
    residual_std = np.sqrt(residual_sum_squares / (window - 2))

    result["trend_slope"] = _pad_leading(slope, series.shape[0])
    result["trend_rsqr"] = _pad_leading(np.clip(r_squared, 0.0, 1.0), series.shape[0])
    result["trend_resid"] = _pad_leading(residual_std, series.shape[0])
    return result


def _rolling_extreme_age(series: pd.Series, window: int, find_max: bool) -> pd.Series:
    """Trading days since the window's extreme (0 = the extreme is today)."""
    windows = _sliding_windows(series, window)
    if windows is None:
        return _empty_like(series)

    # np.argmax over a window containing NaN is meaningless, so mask those rows.
    incomplete = np.isnan(windows).any(axis=1)
    # Scan each window newest-first so numpy's first-match tie-break resolves to
    # the most recent occurrence, and the offset it returns *is* the age in days.
    reversed_windows = windows[:, ::-1]
    ages = np.argmax(reversed_windows, axis=1) if find_max else np.argmin(reversed_windows, axis=1)
    padded = _pad_leading(np.where(incomplete, np.nan, ages.astype("float64")), series.shape[0])
    return pd.Series(padded, index=series.index, dtype="float64")


def compute_days_since_high(series: pd.Series, window: int) -> pd.Series:
    """
    Trading days since the highest value in the trailing window.

    "How long since the high" is information a max/close ratio throws away: two
    names can sit 5% below their 60-day high with one having peaked yesterday and
    the other three months ago.

    Args:
        series: Level Series for one symbol, in chronological order.
        window: Trailing window in trading days.

    Returns:
        Series in ``[0, window - 1]``; 0 means today set the window high. Ties
        resolve to the most recent occurrence. NaN before the first full window.
    """
    return _rolling_extreme_age(series, window, find_max=True)


def compute_days_since_low(series: pd.Series, window: int) -> pd.Series:
    """
    Trading days since the lowest value in the trailing window.

    Args:
        series: Level Series for one symbol, in chronological order.
        window: Trailing window in trading days.

    Returns:
        Series in ``[0, window - 1]``; 0 means today set the window low. Ties
        resolve to the most recent occurrence. NaN before the first full window.
    """
    return _rolling_extreme_age(series, window, find_max=False)


def compute_rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Percentile rank of the current value within its own trailing window.

    Scale-free by construction, so it survives the cross section without any
    winsorisation: 1.0 means today is the highest value in the window.

    Args:
        series: Level Series for one symbol, in chronological order.
        window: Trailing window in trading days.

    Returns:
        Series in ``(0, 1]``. NaN before the first full window.
    """
    return series.astype("float64").rolling(window).rank(pct=True)


def compute_rolling_correlation(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """
    Rolling Pearson correlation, with degenerate windows mapped to NaN.

    Args:
        left: First Series.
        right: Second Series, aligned to ``left``.
        window: Trailing window in trading days.

    Returns:
        Series in ``[-1, 1]``. A window where either leg is constant has no
        defined correlation and yields NaN rather than an infinity.
    """
    correlation = left.astype("float64").rolling(window).corr(right.astype("float64"))
    return correlation.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)


def compute_log_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Log change over ``periods`` rows, undefined wherever either leg is not positive.

    Args:
        series: Strictly positive level Series (price or volume).
        periods: Lag in rows.

    Returns:
        float64 Series of log changes; NaN where a level is zero or negative.
    """
    positive = series.astype("float64").where(lambda values: values > 0)
    return np.log(positive) - np.log(positive.shift(periods))
