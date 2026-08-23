"""Price/volume technical factors derived from daily OHLCV bars.

This is the *technical* factor family: everything computable from a single symbol's
own bar history, with no vendor data beyond the bars already on disk. It is adapted
from the concepts behind Microsoft qlib's **Alpha158** feature set — the K-line shape
ratios, the rolling-regression "trend quality" block, the extreme-position and
extreme-*timing* block, and the price/volume confirmation block. Only the definitions
are borrowed; the implementations here are our own and qlib is not a dependency.

Five families, all computed per symbol:

- **Bar shape (K-line)** — body and shadow sizes relative to the bar's own range. A day
  that opens low and closes on its high differs from one with the same return that
  gapped up and faded, and the close-to-close series cannot tell them apart.
- **Momentum, dispersion, directional balance** — rate of change, price over its own
  moving average, price-level dispersion, and the share of the window's absolute price
  movement contributed by up days (the RSI construction).
- **Trend quality** — rolling OLS of price on time: slope, R-squared, residual
  dispersion. The block we most conspicuously lacked; it separates "up 10%" from "up
  10% in a straight line", and only the first of those was already measured.
- **Extreme position and timing** — rolling max/min and 80th/20th percentiles against
  the close, the close's rank in its own window, and the *age* of the window's high and
  low: two names 5% off their 60-day high differ enormously if one peaked yesterday.
- **Price/volume confirmation** — rolling correlation of price with volume and of price
  change with volume change, plus volume-activity dispersion.

Publication-lag note: every input is an end-of-day price or volume print, where
``publication_date`` equals ``reference_date``, so unlike fundamentals and macro these
need no lag alignment — the only timing discipline required is the usual
``signal_lag_days`` between observing a factor and trading on it, applied downstream.

Every window is trailing and must be complete; nothing is centred and no factor reads a
row later than its own. Inputs must be **split- and dividend-adjusted** OHLCV, the same
schema :mod:`core.data.factors.microstructure` consumes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.data.factors.price_technical_helpers import (
    compute_days_since_high,
    compute_days_since_low,
    compute_log_change,
    compute_rolling_correlation,
    compute_rolling_percentile_rank,
    compute_rolling_trend,
    safe_ratio,
)
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)
MIN_WINDOW = 3
UPPER_QUANTILE = 0.8
LOWER_QUANTILE = 0.2

REQUIRED_BAR_COLUMNS: tuple[str, ...] = (
    "adj_open", "adj_high", "adj_low", "adj_close", "volume"
)  # fmt: skip

BAR_SHAPE_FACTOR_COLUMNS: tuple[str, ...] = (
    "bar_body_ratio", "bar_range_pct", "bar_upper_shadow_ratio",
    "bar_lower_shadow_ratio", "bar_close_position",
)  # fmt: skip

WINDOWED_FACTOR_STEMS: tuple[str, ...] = (
    # momentum / dispersion / directional balance
    "roc", "ma_ratio", "price_std", "sump",
    # trend quality
    "trend_slope", "trend_rsqr", "trend_resid",
    # extreme position and timing
    "max_to_close", "min_to_close", "qtlu_to_close", "qtld_to_close", "close_rank",
    "days_since_high", "days_since_low", "high_low_age_gap",
    # price/volume confirmation
    "corr_close_volume", "corr_return_volume_change", "volume_std_ratio",
    "volume_to_mean", "wvma",
)  # fmt: skip


def build_price_technical_columns(windows: tuple[int, ...] = DEFAULT_WINDOWS) -> tuple[str, ...]:
    """
    Full ordered column list emitted for a given set of windows.

    Args:
        windows: Trailing windows in trading days.

    Returns:
        Bar-shape columns first, then every windowed stem suffixed ``_{window}d``.
    """
    windowed = tuple(f"{stem}_{window}d" for stem in WINDOWED_FACTOR_STEMS for window in windows)
    return BAR_SHAPE_FACTOR_COLUMNS + windowed


PRICE_TECHNICAL_FACTOR_COLUMNS: tuple[str, ...] = build_price_technical_columns()


def compute_bar_shape_factors(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Single-bar K-line shape ratios, normalised by the bar's own range.

    Normalising by the range rather than by the open leaves four of the five bounded,
    so they survive the cross section without winsorisation. ``bar_close_position`` is
    the single-bar, [-1, 1]-scaled sibling of ``close_location_21d`` in
    :mod:`core.data.factors.microstructure`, which averages the same quantity over 21
    days and so discards the day's own information.

    Args:
        bars: DataFrame with ``adj_open``, ``adj_high``, ``adj_low``, ``adj_close``.

    Returns:
        DataFrame indexed like ``bars`` with :data:`BAR_SHAPE_FACTOR_COLUMNS`. Zero-range
        bars (halted or untraded) yield NaN rather than an infinity.
    """
    open_price = bars["adj_open"].astype("float64")
    high = bars["adj_high"].astype("float64")
    low = bars["adj_low"].astype("float64")
    close = bars["adj_close"].astype("float64")

    bar_range = high - low
    body = pd.concat([open_price, close], axis=1)

    factors = pd.DataFrame(index=bars.index)
    factors["bar_body_ratio"] = safe_ratio(close - open_price, bar_range)
    factors["bar_range_pct"] = safe_ratio(bar_range, close)
    factors["bar_upper_shadow_ratio"] = safe_ratio(high - body.max(axis=1), bar_range)
    factors["bar_lower_shadow_ratio"] = safe_ratio(body.min(axis=1) - low, bar_range)
    factors["bar_close_position"] = safe_ratio(2.0 * close - high - low, bar_range)
    return factors[list(BAR_SHAPE_FACTOR_COLUMNS)]


def compute_momentum_factors(close: pd.Series, window: int) -> pd.DataFrame:
    """
    Rate of change, moving-average ratio, price-level dispersion, and up-day share.

    Args:
        close: Adjusted closes for one symbol, in chronological order.
        window: Trailing window in trading days.

    Returns:
        DataFrame with ``roc_{w}d`` (simple return over the window), ``ma_ratio_{w}d``
        (close over its trailing mean, minus one), ``price_std_{w}d`` (rolling standard
        deviation of the close over the close — qlib's STD, a coefficient of variation of
        the price *level*) and ``sump_{w}d`` in [0, 1], the RSI construction as a bounded
        ratio where 0.5 is balanced and 1.0 means every move was upward. The down-share
        and the up-minus-down spread are exact linear functions of ``sump``, so only the
        one column is emitted.
    """
    close = close.astype("float64")
    suffix = f"_{window}d"
    price_change = close.diff()

    factors = pd.DataFrame(index=close.index)
    factors[f"roc{suffix}"] = safe_ratio(close, close.shift(window)) - 1.0
    factors[f"ma_ratio{suffix}"] = safe_ratio(close, close.rolling(window).mean()) - 1.0
    factors[f"price_std{suffix}"] = safe_ratio(close.rolling(window).std(), close)
    factors[f"sump{suffix}"] = safe_ratio(
        price_change.clip(lower=0.0).rolling(window).sum(),
        price_change.abs().rolling(window).sum(),
    )
    return factors


def compute_trend_quality_factors(close: pd.Series, window: int) -> pd.DataFrame:
    """
    Scale-free rolling-regression trend block: slope, R-squared, residual dispersion.

    Args:
        close: Adjusted closes for one symbol, in chronological order.
        window: Trailing window in trading days (must be >= 3).

    Returns:
        DataFrame with ``trend_slope_{w}d`` (fractional drift per day),
        ``trend_rsqr_{w}d`` in [0, 1] and ``trend_resid_{w}d`` (residual standard
        deviation as a fraction of price). Slope and residual dispersion are divided
        by the current close so they compare across price levels.

    Raises:
        ValueError: If ``window`` < 3, propagated from the regression helper.
    """
    close = close.astype("float64")
    suffix = f"_{window}d"
    trend = compute_rolling_trend(close, window)

    factors = pd.DataFrame(index=close.index)
    factors[f"trend_slope{suffix}"] = safe_ratio(trend["trend_slope"], close)
    factors[f"trend_rsqr{suffix}"] = trend["trend_rsqr"]
    factors[f"trend_resid{suffix}"] = safe_ratio(trend["trend_resid"], close)
    return factors


def compute_extreme_position_factors(close: pd.Series, window: int) -> pd.DataFrame:
    """
    Where the close sits in its window's distribution, and when the extremes happened.

    Args:
        close: Adjusted closes for one symbol, in chronological order.
        window: Trailing window in trading days.

    Returns:
        DataFrame with ``max_to_close_{w}d`` (>= 0), ``min_to_close_{w}d`` (<= 0),
        ``qtlu_to_close_{w}d`` / ``qtld_to_close_{w}d`` (80th/20th percentile of the
        close relative to the close), ``close_rank_{w}d`` in (0, 1],
        ``days_since_high_{w}d`` / ``days_since_low_{w}d`` in [0, window - 1] and
        ``high_low_age_gap_{w}d`` (positive = the low is the more recent extreme).
    """
    close = close.astype("float64")
    suffix = f"_{window}d"
    rolling_close = close.rolling(window)
    high_age = compute_days_since_high(close, window)
    low_age = compute_days_since_low(close, window)

    factors = pd.DataFrame(index=close.index)
    factors[f"max_to_close{suffix}"] = safe_ratio(rolling_close.max(), close) - 1.0
    factors[f"min_to_close{suffix}"] = safe_ratio(rolling_close.min(), close) - 1.0
    factors[f"qtlu_to_close{suffix}"] = (
        safe_ratio(rolling_close.quantile(UPPER_QUANTILE), close) - 1.0
    )
    factors[f"qtld_to_close{suffix}"] = (
        safe_ratio(rolling_close.quantile(LOWER_QUANTILE), close) - 1.0
    )
    factors[f"close_rank{suffix}"] = compute_rolling_percentile_rank(close, window)
    factors[f"days_since_high{suffix}"] = high_age
    factors[f"days_since_low{suffix}"] = low_age
    factors[f"high_low_age_gap{suffix}"] = high_age - low_age
    return factors


def compute_volume_price_factors(close: pd.Series, volume: pd.Series, window: int) -> pd.DataFrame:
    """
    Price/volume confirmation and volume-activity factors. Both correlations are taken
    in log space so one volume spike cannot dominate a window, and both are NaN when
    either leg is constant.

    Args:
        close: Adjusted closes for one symbol, in chronological order.
        volume: Share volume aligned to ``close``.
        window: Trailing window in trading days.

    Returns:
        DataFrame with ``corr_close_volume_{w}d`` (log price vs log volume),
        ``corr_return_volume_change_{w}d`` (log return vs log volume change),
        ``volume_std_ratio_{w}d`` (volume coefficient of variation),
        ``volume_to_mean_{w}d`` (today's volume over its trailing mean, minus one) and
        ``wvma_{w}d`` (dispersion of volume-weighted absolute returns over their mean —
        high when the big moves cluster on the heavy-volume days).
    """
    close = close.astype("float64")
    volume = volume.astype("float64")
    suffix = f"_{window}d"

    log_return = compute_log_change(close)
    mean_volume = volume.rolling(window).mean()
    weighted_move = log_return.abs() * volume

    factors = pd.DataFrame(index=close.index)
    factors[f"corr_close_volume{suffix}"] = compute_rolling_correlation(
        np.log(close.where(close > 0)), np.log(volume.where(volume > 0)), window
    )
    factors[f"corr_return_volume_change{suffix}"] = compute_rolling_correlation(
        log_return, compute_log_change(volume), window
    )
    factors[f"volume_std_ratio{suffix}"] = safe_ratio(volume.rolling(window).std(), mean_volume)
    factors[f"volume_to_mean{suffix}"] = safe_ratio(volume, mean_volume) - 1.0
    factors[f"wvma{suffix}"] = safe_ratio(
        weighted_move.rolling(window).std(), weighted_move.rolling(window).mean()
    )
    return factors


def compute_price_technical_factors(
    bars: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Compute every price/volume technical factor for one symbol's OHLCV history.

    Args:
        bars: DataFrame indexed by date (chronological, unique) with ``adj_open``,
            ``adj_high``, ``adj_low``, ``adj_close``, ``volume``.
        windows: Trailing windows in trading days. Each must be >= 3.

    Returns:
        DataFrame indexed like ``bars`` with the columns from
        :func:`build_price_technical_columns` for ``windows``. Empty input returns an
        empty frame carrying the same schema.

    Raises:
        DataSchemaError: If a required bar column is missing or a window is < 3.

    Example:
        >>> factors = compute_price_technical_factors(aapl_bars)  # doctest: +SKIP
    """
    missing = sorted(set(REQUIRED_BAR_COLUMNS) - set(bars.columns))
    if missing:
        raise DataSchemaError(f"bars is missing required column(s): {missing}")
    invalid = sorted(window for window in windows if window < MIN_WINDOW)
    if invalid:
        raise DataSchemaError(f"price technical windows must be >= {MIN_WINDOW}, got {invalid}")

    columns = list(build_price_technical_columns(tuple(windows)))
    if bars.empty:
        return pd.DataFrame(columns=columns, index=bars.index, dtype="float64")

    close = bars["adj_close"].astype("float64")
    volume = bars["volume"].astype("float64")

    blocks: list[pd.DataFrame] = [compute_bar_shape_factors(bars)]
    for window in windows:
        blocks.append(compute_momentum_factors(close, window))
        blocks.append(compute_trend_quality_factors(close, window))
        blocks.append(compute_extreme_position_factors(close, window))
        blocks.append(compute_volume_price_factors(close, volume, window))

    factors = pd.concat(blocks, axis=1)
    logger.debug(
        "computed price technical factors",
        extra={"rows": len(factors), "columns": len(columns), "windows": list(windows)},
    )
    return factors[columns].astype("float64")
