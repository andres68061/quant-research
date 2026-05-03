"""Simple causal regime baseline exposure rules.

These helpers provide transparent alternatives to the HMM regime overlay.
They intentionally use only information observable through each date so they
can be compared against model-based regime probabilities without look-ahead.
"""

from __future__ import annotations

import pandas as pd


def vix_threshold_exposure(
    vix: pd.Series,
    low: float = 15.0,
    high: float = 30.0,
) -> pd.Series:
    """Convert VIX levels into a continuous risk exposure scale.

    Exposure is 1.0 when VIX is at or below ``low``, 0.0 when VIX is at or
    above ``high``, and linearly interpolated between those thresholds.

    Args:
        vix: VIX level series indexed by date.
        low: VIX level at or below which exposure is fully on.
        high: VIX level at or above which exposure is fully off.

    Returns:
        Series named ``"vix_threshold_exposure"`` with values in ``[0, 1]``.

    Example:
        >>> vix = pd.Series([15.0, 22.5, 30.0])
        >>> vix_threshold_exposure(vix).tolist()
        [1.0, 0.5, 0.0]
    """
    if high <= low:
        raise ValueError("high must be greater than low")

    clean_vix = _strip_tz(vix.astype(float).sort_index())
    exposure = ((high - clean_vix) / (high - low)).clip(0.0, 1.0)
    exposure.name = "vix_threshold_exposure"
    return exposure


def moving_average_exposure(
    prices: pd.DataFrame,
    market_symbol: str = "^GSPC",
    window: int = 200,
) -> pd.Series:
    """Convert a market price trend rule into a binary exposure scale.

    Exposure is 1.0 when the market close is at or above its trailing moving
    average and 0.0 otherwise. Values are NaN until the moving-average window
    has enough observations.

    Args:
        prices: Wide price panel indexed by date.
        market_symbol: Column containing the market benchmark close.
        window: Trailing moving-average window in observations.

    Returns:
        Series named ``"ma{window}_exposure"`` with values 0.0, 1.0, or NaN.

    Example:
        >>> prices = pd.DataFrame({"^GSPC": [1.0, 2.0, 3.0]})
        >>> moving_average_exposure(prices, window=2).tolist()
        [nan, 1.0, 1.0]
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if market_symbol not in prices.columns:
        raise ValueError(f"market_symbol {market_symbol!r} not found in prices")

    market_close = _strip_tz(prices[market_symbol].astype(float).sort_index())
    moving_average = market_close.rolling(window=window, min_periods=window).mean()
    exposure = (market_close >= moving_average).astype(float)
    exposure[moving_average.isna()] = pd.NA
    exposure.name = f"ma{window}_exposure"
    return exposure


def _strip_tz(series: pd.Series) -> pd.Series:
    """Return *series* with a tz-naive DatetimeIndex if applicable."""
    if getattr(series.index, "tz", None) is not None:
        series = series.copy()
        series.index = series.index.tz_localize(None)
    return series
