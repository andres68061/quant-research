"""Tests for Black–Scholes pricing and implied volatility."""

import math

import numpy as np
import pytest

from core.exceptions import ImpliedVolatilityError
from core.surfaces.black_scholes import (
    OptionKind,
    black_scholes_price,
    implied_volatility,
)
from core.surfaces.grid import implied_vol_surface_grid

_CALL_PUT: tuple[OptionKind, OptionKind] = ("call", "put")


@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("option_type", _CALL_PUT)
def test_implied_vol_round_trip_atm(sigma: float, option_type: OptionKind) -> None:
    """Price at sigma then recover sigma via implied_volatility."""
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
    px = black_scholes_price(S, K, T, r, q, sigma, option_type=option_type)
    iv = implied_volatility(px, S, K, T, r, q, option_type=option_type)
    assert math.isclose(iv, sigma, rel_tol=0, abs_tol=1e-8)


@pytest.mark.parametrize(
    "S, K, T",
    [
        (120.0, 100.0, 0.5),  # ITM call / OTM put
        (80.0, 100.0, 0.5),  # OTM call / ITM put
    ],
)
def test_implied_vol_round_trip_moneyness(S: float, K: float, T: float) -> None:
    sigma = 0.35
    r, q = 0.03, 0.01
    for opt in _CALL_PUT:
        px = black_scholes_price(S, K, T, r, q, sigma, option_type=opt)
        iv = implied_volatility(px, S, K, T, r, q, option_type=opt)
        assert math.isclose(iv, sigma, rel_tol=0, abs_tol=1e-7)


def test_put_call_parity() -> None:
    """C - P = S*exp(-qT) - K*exp(-rT)."""
    S, K, T, r, q, sig = 100.0, 95.0, 0.75, 0.04, 0.015, 0.22
    c = black_scholes_price(S, K, T, r, q, sig, option_type="call")
    p = black_scholes_price(S, K, T, r, q, sig, option_type="put")
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        dict(S=-1, K=100, T=1, r=0, q=0, sigma=0.2),
        dict(S=100, K=0, T=1, r=0, q=0, sigma=0.2),
        dict(S=100, K=100, T=0, r=0, q=0, sigma=0.2),
        dict(S=100, K=100, T=1, r=0, q=0, sigma=0),
    ],
)
def test_black_scholes_invalid_inputs_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        black_scholes_price(
            bad["S"],
            bad["K"],
            bad["T"],
            bad["r"],
            bad["q"],
            bad["sigma"],
        )


def test_implied_vol_out_of_range_raises() -> None:
    """Price above the Black–Scholes upper bound (≈ S e^{-qT} for calls) cannot be fit."""
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
    impossible = S * math.exp(-q * T) + 1.0  # strictly above max attainable call value
    with pytest.raises(ImpliedVolatilityError):
        implied_volatility(impossible, S, K, T, r, q, option_type="call")


def test_implied_vol_surface_grid_matches_single_vol() -> None:
    sig = 0.2
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([0.25, 0.5])
    prices = np.empty((2, 3))
    for i, t in enumerate(T):
        for j, k in enumerate(K):
            prices[i, j] = black_scholes_price(100.0, k, t, 0.01, 0.0, sig)
    iv = implied_vol_surface_grid(prices, 100.0, K, T, 0.01, 0.0)
    assert np.allclose(iv, sig, atol=1e-6)


def test_implied_vol_surface_grid_shape_error() -> None:
    with pytest.raises(ValueError):
        implied_vol_surface_grid(
            np.ones((2, 2)),
            100.0,
            np.array([1.0]),
            np.array([0.5, 1.0]),
            0.0,
            0.0,
        )
