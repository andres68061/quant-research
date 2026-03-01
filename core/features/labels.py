"""
Target / label construction for ML models.

Centralises the creation of forward-looking labels so that every model
uses the same conventions (log-return direction, multi-horizon, etc.).
"""

import numpy as np
import pandas as pd


def next_day_direction(prices: pd.Series) -> pd.Series:
    """
    Binary label: will price go up tomorrow?

    Args:
        prices: Price series (indexed by date)

    Returns:
        Series of 1 (up) / 0 (down), with the last row NaN
    """
    log_returns = np.log(prices / prices.shift(1))
    next_return = log_returns.shift(-1)
    return (next_return > 0).astype(float).where(next_return.notna())


def n_day_direction(prices: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Binary label: will price be higher in *horizon* days?

    Args:
        prices: Price series
        horizon: Number of forward days

    Returns:
        Series of 1/0, with trailing NaN
    """
    future_return = np.log(prices.shift(-horizon) / prices)
    return (future_return > 0).astype(float).where(future_return.notna())


def return_bucket(
    prices: pd.Series,
    horizon: int = 5,
    n_buckets: int = 3,
) -> pd.Series:
    """
    Ordinal label: assign forward return to one of *n_buckets* buckets.

    Uses expanding quantiles to avoid look-ahead bias.

    Args:
        prices: Price series
        horizon: Forward horizon in days
        n_buckets: Number of equal-frequency buckets (default 3: down/flat/up)

    Returns:
        Series of integer bucket labels (0 .. n_buckets-1)
    """
    future_return = np.log(prices.shift(-horizon) / prices)
    labels = pd.Series(np.nan, index=prices.index)

    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]

    for i in range(len(prices)):
        if np.isnan(future_return.iloc[i]):
            continue
        history = future_return.iloc[: i + 1].dropna()
        if len(history) < 30:
            continue
        cuts = np.quantile(history, quantiles)
        labels.iloc[i] = float(np.searchsorted(cuts, future_return.iloc[i]))

    return labels
