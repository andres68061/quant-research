"""Range- and volume-based factors from daily OHLCV bars.

The raw price layer has carried ``adj_open``/``adj_high``/``adj_low``/``volume``
since the FMP cutover, but the canonical panel kept only ``adj_close``. Everything
here is computed from bars already on disk — no additional vendor calls.

Two families:

**Range volatility estimators.** The high-low range uses the whole day's path
rather than two endpoints, so it extracts several times more information per
observation than a close-to-close estimate. Parkinson is the pure-range estimator;
Garman-Klass adds the open-close leg and is roughly 7x more efficient than
close-to-close.

**Effective spread.** Corwin & Schultz (2012) recover the bid-ask spread from the
ratio of a two-day high-low range to two consecutive one-day ranges — the daily
data equivalent of a TAQ spread measure.

All estimators assume **split-adjusted** OHLC. Using unadjusted bars silently
produces enormous fake ranges on split dates.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
DEFAULT_WINDOW = 21

# 1 / (4 ln 2), the Parkinson scaling constant.
_PARKINSON_SCALE = 1.0 / (4.0 * np.log(2.0))
# Corwin-Schultz: 3 - 2*sqrt(2), appears in the alpha denominator.
_CS_DENOMINATOR = 3.0 - 2.0 * np.sqrt(2.0)

MICROSTRUCTURE_FACTOR_COLUMNS: tuple[str, ...] = (
    "parkinson_vol",
    "garman_klass_vol",
    "range_to_close_vol",
    "corwin_schultz_spread",
    "amihud_illiquidity",
    "neg_amihud_illiquidity",
    "overnight_return_21d",
    "intraday_return_21d",
    "overnight_intraday_gap",
    "close_location_21d",
)


def _log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Log ratio that is NaN wherever either leg is not strictly positive."""
    safe_numerator = numerator.where(numerator > 0)
    safe_denominator = denominator.where(denominator > 0)
    return np.log(safe_numerator / safe_denominator)


def compute_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.Series:
    """
    Annualized Parkinson (1980) range volatility.

    ``sigma^2 = mean( ln(H/L)^2 ) / (4 ln 2)`` over the window.

    Args:
        high: Daily adjusted highs for one symbol.
        low: Daily adjusted lows, aligned to ``high``.
        window: Trailing window in trading days.

    Returns:
        Annualized volatility Series (decimal, e.g. 0.25 = 25%).

    Notes:
        Downward biased when the true process has a drift or when the security
        gaps overnight — the range only sees the intraday path.
    """
    squared_log_range = _log_ratio(high, low) ** 2
    variance = squared_log_range.rolling(window, min_periods=max(5, window // 2)).mean()
    return np.sqrt(variance * _PARKINSON_SCALE * TRADING_DAYS_PER_YEAR)


def compute_garman_klass_volatility(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.Series:
    """
    Annualized Garman-Klass (1980) volatility from the full OHLC bar.

    ``sigma^2 = 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2``.

    Args:
        open_price: Daily adjusted opens for one symbol.
        high: Daily adjusted highs.
        low: Daily adjusted lows.
        close: Daily adjusted closes.
        window: Trailing window in trading days.

    Returns:
        Annualized volatility Series. Individual-day variance estimates can go
        slightly negative; they are floored at zero before averaging.
    """
    log_range = _log_ratio(high, low)
    log_open_close = _log_ratio(close, open_price)
    daily_variance = 0.5 * log_range**2 - (2.0 * np.log(2.0) - 1.0) * log_open_close**2
    variance = (
        daily_variance.clip(lower=0.0).rolling(window, min_periods=max(5, window // 2)).mean()
    )
    return np.sqrt(variance * TRADING_DAYS_PER_YEAR)


def _adjust_for_overnight_gap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Shift each day's high/low to remove the overnight jump from the prior close.

    Corwin & Schultz require this: an overnight gap inflates the two-day range
    without inflating either one-day range, which the estimator would otherwise
    read as spread. When today's low sits above yesterday's close the whole bar is
    shifted down by the gap, and symmetrically for a downward gap.

    Args:
        high: Daily adjusted highs.
        low: Daily adjusted lows.
        close: Daily adjusted closes.

    Returns:
        ``(adjusted_high, adjusted_low)``.
    """
    prior_close = close.shift(1)
    gap_up = (low - prior_close).where(low > prior_close, 0.0)
    gap_down = (high - prior_close).where(high < prior_close, 0.0)
    gap = gap_up + gap_down
    return high - gap, low - gap


def compute_corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.Series:
    """
    Corwin-Schultz (2012) effective bid-ask spread estimator, smoothed.

    Compares a single two-day high-low range with two consecutive one-day ranges.
    Volatility scales with the square root of time but the spread does not, so the
    difference identifies the spread.

    Args:
        high: Daily adjusted highs for one symbol.
        low: Daily adjusted lows, aligned to ``high``.
        close: Daily adjusted closes, used for the overnight-gap adjustment.
        window: Trailing window over which daily estimates are averaged.

    Returns:
        Proportional spread Series (decimal, e.g. 0.002 = 20 bps).

    Notes:
        Daily estimates are frequently negative — that is the estimator's known
        small-sample behaviour, not a bug. The published remedy is to floor
        negatives at zero before averaging, which is what this does.

        The estimator was validated on 1990s-2000s data and **overstates the
        spread for modern mega-caps**, where true effective spreads are a basis
        point or two. Treat it as a cross-sectional liquidity ranking, not as a
        cost input — `core.data.liquidity` owns the cost schedule.
    """
    adjusted_high, adjusted_low = _adjust_for_overnight_gap(high, low, close)
    squared_log_range = _log_ratio(adjusted_high, adjusted_low) ** 2
    beta = squared_log_range + squared_log_range.shift(1)

    two_day_high = pd.concat([adjusted_high, adjusted_high.shift(1)], axis=1).max(axis=1)
    two_day_low = pd.concat([adjusted_low, adjusted_low.shift(1)], axis=1).min(axis=1)
    gamma = _log_ratio(two_day_high, two_day_low) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DENOMINATOR - np.sqrt(
        gamma / _CS_DENOMINATOR
    )
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0).rolling(window, min_periods=max(5, window // 2)).mean()


def compute_amihud_illiquidity(
    close: pd.Series,
    volume: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.Series:
    """
    Amihud (2002) illiquidity: average |return| per dollar of volume.

    Args:
        close: Daily adjusted closes for one symbol.
        volume: Daily share volume, aligned to ``close``.
        window: Trailing window in trading days.

    Returns:
        Illiquidity Series, scaled by 1e6 so values sit in a readable range.
        Higher = more price impact per dollar traded = less liquid.
    """
    daily_return = close.pct_change()
    dollar_volume = (close * volume).replace(0.0, np.nan)
    impact = daily_return.abs() / dollar_volume
    return impact.rolling(window, min_periods=max(5, window // 2)).mean() * 1e6


def compute_overnight_intraday_split(
    open_price: pd.Series,
    close: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Decompose total return into its overnight and intraday components.

    Overnight is close(t-1) → open(t); intraday is open(t) → close(t). The two
    legs have documented opposite-signed premia: for US equities most of the
    equity premium has historically accrued overnight while intraday returns are
    close to flat, so the split is a signal, not an accounting curiosity.

    Args:
        open_price: Daily adjusted opens for one symbol.
        close: Daily adjusted closes, aligned to ``open_price``.
        window: Trailing window for the cumulative sums.

    Returns:
        DataFrame with ``overnight_return_21d``, ``intraday_return_21d`` (trailing
        sums of log returns) and ``overnight_intraday_gap`` (their difference).
    """
    overnight = _log_ratio(open_price, close.shift(1))
    intraday = _log_ratio(close, open_price)
    # The first bar has an intraday leg but no overnight leg (no prior close).
    # Leaving it in would make the two trailing sums cover different days, so the
    # decomposition would not add up to the total return in the first window.
    intraday = intraday.where(overnight.notna())
    min_periods = max(5, window // 2)

    result = pd.DataFrame(index=close.index)
    result["overnight_return_21d"] = overnight.rolling(window, min_periods=min_periods).sum()
    result["intraday_return_21d"] = intraday.rolling(window, min_periods=min_periods).sum()
    result["overnight_intraday_gap"] = (
        result["overnight_return_21d"] - result["intraday_return_21d"]
    )
    return result


def compute_close_location(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.Series:
    """
    Average position of the close within the day's range (0 = low, 1 = high).

    A close persistently near the high indicates buying pressure into the bell.

    Args:
        high: Daily adjusted highs for one symbol.
        low: Daily adjusted lows.
        close: Daily adjusted closes.
        window: Trailing window in trading days.

    Returns:
        Series in [0, 1]; NaN on zero-range days (halted or untraded).
    """
    day_range = (high - low).replace(0.0, np.nan)
    location = ((close - low) / day_range).clip(0.0, 1.0)
    return location.rolling(window, min_periods=max(5, window // 2)).mean()


def compute_microstructure_factors(
    bars: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Compute every microstructure factor for one symbol's OHLCV history.

    Args:
        bars: DataFrame indexed by date with ``adj_open``, ``adj_high``,
            ``adj_low``, ``adj_close``, ``volume``.
        window: Trailing window shared by all estimators.

    Returns:
        DataFrame indexed like ``bars`` with
        :data:`MICROSTRUCTURE_FACTOR_COLUMNS`. Empty input returns an empty frame
        with the right columns.

    Example:
        >>> factors = compute_microstructure_factors(aapl_bars)  # doctest: +SKIP
    """
    if bars.empty:
        return pd.DataFrame(columns=list(MICROSTRUCTURE_FACTOR_COLUMNS), index=bars.index)

    open_price = bars["adj_open"].astype("float64")
    high = bars["adj_high"].astype("float64")
    low = bars["adj_low"].astype("float64")
    close = bars["adj_close"].astype("float64")
    volume = bars["volume"].astype("float64")

    factors = pd.DataFrame(index=bars.index)
    factors["parkinson_vol"] = compute_parkinson_volatility(high, low, window)
    factors["garman_klass_vol"] = compute_garman_klass_volatility(
        open_price, high, low, close, window
    )
    close_to_close_vol = close.pct_change().rolling(
        window, min_periods=max(5, window // 2)
    ).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    # >1 means the intraday path is wilder than the close-to-close series implies.
    factors["range_to_close_vol"] = factors["parkinson_vol"] / close_to_close_vol.where(
        close_to_close_vol > 0
    )
    factors["corwin_schultz_spread"] = compute_corwin_schultz_spread(high, low, close, window)
    factors["amihud_illiquidity"] = compute_amihud_illiquidity(close, volume, window)
    factors["neg_amihud_illiquidity"] = -factors["amihud_illiquidity"]
    factors = factors.join(compute_overnight_intraday_split(open_price, close, window))
    factors["close_location_21d"] = compute_close_location(high, low, close, window)

    return factors[list(MICROSTRUCTURE_FACTOR_COLUMNS)]
