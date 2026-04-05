"""
Helpers for building implied-volatility grids from tabular option prices (no plotting).
"""

from __future__ import annotations

import numpy as np

from core.exceptions import ImpliedVolatilityError
from core.surfaces.black_scholes import OptionKind, implied_volatility


def implied_vol_surface_grid(
    market_prices: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    times_to_expiry: np.ndarray,
    r: float,
    q: float,
    *,
    option_type: OptionKind = "call",
    sigma_low: float = 1e-6,
    sigma_high: float = 10.0,
) -> np.ndarray:
    """
    Build a 2-D implied volatility surface from a matrix of market option prices.

    ``market_prices[i, j]`` is the price for expiry ``times_to_expiry[i]`` and
    strike ``strikes[j]``. Cells where inversion fails are set to ``nan``.

    Parameters
    ----------
    market_prices : ndarray, shape (n_T, n_K)
        Observed option prices.
    spot : float
        Underlying spot price ``S`` (> 0).
    strikes : ndarray, shape (n_K,)
        Strike grid.
    times_to_expiry : ndarray, shape (n_T,)
        Time to each expiry in years, aligned with axis 0 of ``market_prices``.
    r, q : float
        Annualized continuous risk-free and dividend yields.
    option_type : {'call', 'put'}
        Contract type.
    sigma_low, sigma_high : float
        Volatility search bracket for :func:`~core.surfaces.black_scholes.implied_volatility`.

    Returns
    -------
    ndarray, shape (n_T, n_K)
        Implied annualized volatilities.

    Examples
    --------
    >>> from core.surfaces.black_scholes import black_scholes_price
    >>> sig = 0.2
    >>> K = np.array([90.0, 100.0, 110.0])
    >>> T = np.array([0.25, 0.5])
    >>> prices = np.empty((2, 3))
    >>> for i, t in enumerate(T):
    ...     for j, k in enumerate(K):
    ...         prices[i, j] = black_scholes_price(100.0, k, t, 0.01, 0.0, sig)
    >>> iv = implied_vol_surface_grid(prices, 100.0, K, T, 0.01, 0.0)
    >>> np.allclose(iv, sig, atol=1e-6)
    True
    """
    prices = np.asarray(market_prices, dtype=np.float64)
    k_arr = np.asarray(strikes, dtype=np.float64)
    t_arr = np.asarray(times_to_expiry, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError(f"market_prices must be 2-D, got shape {prices.shape}")
    if t_arr.ndim != 1 or k_arr.ndim != 1:
        raise ValueError("strikes and times_to_expiry must be 1-D arrays")
    if prices.shape[0] != t_arr.shape[0] or prices.shape[1] != k_arr.shape[0]:
        raise ValueError(
            "shape mismatch: market_prices must be (len(times_to_expiry), len(strikes))"
        )

    out = np.full_like(prices, np.nan, dtype=np.float64)
    for i in range(prices.shape[0]):
        for j in range(prices.shape[1]):
            try:
                out[i, j] = implied_volatility(
                    float(prices[i, j]),
                    spot,
                    float(k_arr[j]),
                    float(t_arr[i]),
                    r,
                    q,
                    option_type=option_type,
                    sigma_low=sigma_low,
                    sigma_high=sigma_high,
                )
            except ImpliedVolatilityError:
                continue
            except ValueError:
                continue
    return out
