"""Tests for simple regime baseline exposure rules."""

import numpy as np
import pandas as pd
import pytest

from core.signals.regime_baselines import (
    moving_average_exposure,
    vix_threshold_exposure,
)


def test_vix_threshold_exposure_bounds_and_thresholds() -> None:
    vix = pd.Series(
        [10.0, 15.0, 22.5, 30.0, 40.0],
        index=pd.bdate_range("2020-01-01", periods=5),
    )

    exposure = vix_threshold_exposure(vix, low=15.0, high=30.0)

    assert exposure.between(0.0, 1.0).all()
    assert exposure.iloc[1] == pytest.approx(1.0)
    assert exposure.iloc[2] == pytest.approx(0.5)
    assert exposure.iloc[3] == pytest.approx(0.0)


def test_vix_threshold_exposure_monotone_decreasing() -> None:
    vix = pd.Series(
        np.linspace(10.0, 40.0, 20),
        index=pd.bdate_range("2020-01-01", periods=20),
    )

    exposure = vix_threshold_exposure(vix)

    assert exposure.is_monotonic_decreasing


def test_vix_threshold_exposure_validates_threshold_order() -> None:
    with pytest.raises(ValueError, match="high must be greater than low"):
        vix_threshold_exposure(pd.Series([20.0]), low=30.0, high=15.0)


def test_moving_average_exposure_window_and_values() -> None:
    prices = pd.DataFrame(
        {"^GSPC": [1.0, 2.0, 3.0, 2.0, 1.0]},
        index=pd.bdate_range("2020-01-01", periods=5),
    )

    exposure = moving_average_exposure(prices, window=3)

    assert exposure.iloc[:2].isna().all()
    assert set(exposure.dropna().unique()).issubset({0.0, 1.0})
    assert exposure.iloc[2] == 1.0
    assert exposure.iloc[-1] == 0.0


def test_moving_average_exposure_ignores_future_prices() -> None:
    prices = pd.DataFrame(
        {"^GSPC": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=pd.bdate_range("2020-01-01", periods=5),
    )
    shocked = prices.copy()
    shocked.iloc[4, 0] = 1.0

    base = moving_average_exposure(prices, window=3)
    changed = moving_average_exposure(shocked, window=3)

    assert base.iloc[3] == changed.iloc[3]


def test_moving_average_exposure_requires_market_symbol() -> None:
    prices = pd.DataFrame({"SPY": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="market_symbol"):
        moving_average_exposure(prices, market_symbol="^GSPC")
