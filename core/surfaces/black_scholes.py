"""
Black–Scholes European option pricing and implied volatility (continuous dividend yield).

All inputs are scalars in natural units: ``T`` in years, ``r`` and ``q`` as annualized
continuous rates, ``sigma`` as annualized volatility. Computations use ``float64``.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]

from core.exceptions import ImpliedVolatilityError

_LOG = logging.getLogger(__name__)

_EPS = 1e-12
OptionKind = Literal["call", "put"]


def _validate_inputs(S: float, K: float, T: float, sigma: float) -> None:
    if not (S > 0 and math.isfinite(S)):
        raise ValueError(f"spot S must be finite and > 0, got {S!r}")
    if not (K > 0 and math.isfinite(K)):
        raise ValueError(f"strike K must be finite and > 0, got {K!r}")
    if not (T > _EPS and math.isfinite(T)):
        raise ValueError(f"time to expiry T must be finite and > {_EPS}, got {T!r}")
    if not (sigma > _EPS and math.isfinite(sigma)):
        raise ValueError(f"volatility sigma must be finite and > {_EPS}, got {sigma!r}")


def d1(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Black–Scholes :math:`d_1` for continuous dividend yield ``q``.

    Parameters
    ----------
    S, K, T, r, q, sigma
        Spot, strike, time to expiry (years), risk-free rate, dividend yield, volatility.

    Returns
    -------
    float
        :math:`d_1` value.

    Examples
    --------
    >>> round(d1(100.0, 100.0, 1.0, 0.05, 0.02, 0.2), 6)
    0.25
    """
    _validate_inputs(S, K, T, sigma)
    sqrt_t = math.sqrt(T)
    denom = sigma * sqrt_t
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / denom


def d2(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Black–Scholes :math:`d_2 = d_1 - \\sigma\\sqrt{T}`.

    Parameters match :func:`d1`.

    Returns
    -------
    float
        :math:`d_2` value.
    """
    d_1 = d1(S, K, T, r, q, sigma)
    sqrt_t = math.sqrt(T)
    return d_1 - sigma * sqrt_t


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    *,
    option_type: OptionKind = "call",
) -> float:
    """
    European Black–Scholes price with continuous dividend yield ``q``.

    Call: :math:`C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)`.

    Put: :math:`P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)`.

    Parameters
    ----------
    S : float
        Spot price (> 0).
    K : float
        Strike (> 0).
    T : float
        Time to expiry in years (> 0).
    r, q : float
        Annualized continuous risk-free and dividend yields (any finite float).
    sigma : float
        Annualized volatility (> 0).
    option_type : {'call', 'put'}
        Contract type.

    Returns
    -------
    float
        Option price in the same currency units as ``S`` and ``K``.

    Examples
    --------
    >>> p = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, option_type="call")
    >>> round(p, 4)
    9.227
    """
    _validate_inputs(S, K, T, sigma)
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    d_1 = d1(S, K, T, r, q, sigma)
    d_2 = d2(S, K, T, r, q, sigma)
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)

    if option_type == "call":
        return float(disc_s * norm.cdf(d_1) - disc_k * norm.cdf(d_2))
    return float(disc_k * norm.cdf(-d_2) - disc_s * norm.cdf(-d_1))


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    *,
    option_type: OptionKind = "call",
    sigma_low: float = 1e-6,
    sigma_high: float = 10.0,
) -> float:
    """
    Implied volatility by 1-D root finding (Brent) on :math:`\\sigma`.

    Solves ``black_scholes_price(..., sigma) == market_price`` for ``sigma`` in
    ``[sigma_low, sigma_high]``.

    Parameters
    ----------
    market_price : float
        Observed option price (same units as spot/strike).
    S, K, T, r, q
        As in :func:`black_scholes_price`.
    option_type : {'call', 'put'}
        Contract type.
    sigma_low, sigma_high : float
        Search bracket for volatility (must be > 0, ``sigma_low < sigma_high``).

    Returns
    -------
    float
        Implied annualized volatility.

    Raises
    ------
    ValueError
        Invalid inputs (spot, strike, time, bracket).
    ImpliedVolatilityError
        If ``market_price`` is outside the attainable model range on the bracket,
        or Brent fails to converge.

    Examples
    --------
    >>> sigma = 0.25
    >>> price = black_scholes_price(100.0, 105.0, 0.5, 0.03, 0.01, sigma)
    >>> round(implied_volatility(price, 100.0, 105.0, 0.5, 0.03, 0.01), 6)
    0.25
    """
    if not (sigma_low > 0 and sigma_high > sigma_low and math.isfinite(sigma_low)):
        raise ValueError(
            f"sigma_low and sigma_high must satisfy 0 < sigma_low < sigma_high, "
            f"got sigma_low={sigma_low!r}, sigma_high={sigma_high!r}"
        )
    if not math.isfinite(market_price):
        raise ValueError(f"market_price must be finite, got {market_price!r}")

    _validate_inputs(S, K, T, sigma_low)
    # Re-validate upper vol with same S,K,T (sigma_high must be positive finite)
    if not (sigma_high > _EPS and math.isfinite(sigma_high)):
        raise ValueError(f"sigma_high must be finite and > {_EPS}, got {sigma_high!r}")

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    def objective(sig: float) -> float:
        return black_scholes_price(S, K, T, r, q, sig, option_type=option_type) - float(
            market_price
        )

    p_low = black_scholes_price(S, K, T, r, q, sigma_low, option_type=option_type)
    p_high = black_scholes_price(S, K, T, r, q, sigma_high, option_type=option_type)

    tol = 1e-10 * max(1.0, abs(float(market_price)))
    if market_price < p_low - tol or market_price > p_high + tol:
        raise ImpliedVolatilityError(
            "market_price is outside the Black–Scholes price range for the given "
            f"volatility bracket [{sigma_low:g}, {sigma_high:g}]: "
            f"model prices span approximately [{p_low:.12g}, {p_high:.12g}], "
            f"market_price={market_price!r}"
        )

    f_low = objective(sigma_low)
    f_high = objective(sigma_high)
    if f_low * f_high > 0:
        raise ImpliedVolatilityError(
            "Could not bracket implied volatility root; try widening sigma_low/sigma_high."
        )

    try:
        root = brentq(objective, sigma_low, sigma_high, xtol=1e-12, rtol=1e-12)
    except ValueError as exc:
        _LOG.warning("brentq failed for implied vol", exc_info=True)
        raise ImpliedVolatilityError("Implied volatility root find failed.") from exc

    if not (root > _EPS and math.isfinite(root)):
        raise ImpliedVolatilityError(f"Non-finite or non-positive implied vol: {root!r}")
    return float(root)


def black_scholes_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    *,
    option_type: OptionKind = "call",
) -> float:
    """
    Black–Scholes delta :math:`\\partial V / \\partial S`.

    Call delta: :math:`e^{-qT} N(d_1)`; put delta: :math:`e^{-qT} (N(d_1) - 1)`.

    Parameters match :func:`black_scholes_price`.

    Returns
    -------
    float
        Delta (dimensionless).
    """
    _validate_inputs(S, K, T, sigma)
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    d_1 = d1(S, K, T, r, q, sigma)
    disc = math.exp(-q * T)
    if option_type == "call":
        return float(disc * norm.cdf(d_1))
    return float(disc * (norm.cdf(d_1) - 1.0))
