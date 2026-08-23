"""Tests for core.metrics.cross_section."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import DataSchemaError
from core.metrics.cross_section import calculate_trailing_cid1_cross_section


def _make_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8, tz="America/New_York")
    return pd.DataFrame(
        {
            # Never below start: pain == 0, +30% return.
            "UP": [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 120.0, 130.0],
            # Dips below start then recovers: pain 0.1 + 0.2 = 0.3, +20%.
            "DIP": [100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 80.0, 120.0],
            # Missing a close inside the window.
            "GAPPY": [100.0, 100.0, 100.0, 100.0, 100.0, np.nan, 101.0, 102.0],
        },
        index=dates,
    )


def test_calculate_trailing_cid1_cross_section_happy_path() -> None:
    prices = _make_prices()
    result = calculate_trailing_cid1_cross_section(prices, [prices.index[-1]], window_days=4)

    up = result.loc[(prices.index[-1], "UP")]
    assert up["cost_basis_pain"] == 0.0
    assert up["total_return"] == pytest.approx(0.3)
    assert np.isinf(up["cid1_ratio"])
    assert up["cid1_angle"] == pytest.approx(np.pi / 2)

    dip = result.loc[(prices.index[-1], "DIP")]
    assert dip["cost_basis_pain"] == pytest.approx(0.3)
    assert dip["total_return"] == pytest.approx(0.2)
    assert dip["cid1_ratio"] == pytest.approx(0.2 / 0.3)
    assert dip["cid1_angle"] == pytest.approx(np.arctan2(0.2, 0.3))
    # Ordering: pain-free path ranks above the dipping path.
    assert up["cid1_angle"] > dip["cid1_angle"]


def test_calculate_trailing_cid1_cross_section_excludes_incomplete_windows() -> None:
    prices = _make_prices()
    result = calculate_trailing_cid1_cross_section(prices, [prices.index[-1]], window_days=4)
    symbols = result.index.get_level_values("symbol")
    assert "GAPPY" not in symbols
    assert set(symbols) == {"UP", "DIP"}


def test_calculate_trailing_cid1_cross_section_flat_path_is_zero() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4, tz="America/New_York")
    prices = pd.DataFrame({"FLAT": [100.0] * 4}, index=dates)
    result = calculate_trailing_cid1_cross_section(prices, [dates[-1]], window_days=4)
    row = result.loc[(dates[-1], "FLAT")]
    assert row["cid1_ratio"] == 0.0
    assert row["cid1_angle"] == pytest.approx(0.0)


def test_calculate_trailing_cid1_cross_section_types_and_index() -> None:
    prices = _make_prices()
    result = calculate_trailing_cid1_cross_section(prices, [prices.index[-1]], window_days=4)
    assert list(result.columns) == [
        "total_return",
        "cost_basis_pain",
        "cid1_ratio",
        "cid1_angle",
    ]
    assert result.index.names == ["date", "symbol"]
    assert result["total_return"].dtype == np.float64


def test_calculate_trailing_cid1_cross_section_bad_inputs() -> None:
    prices = _make_prices()
    with pytest.raises(DataSchemaError):
        calculate_trailing_cid1_cross_section(prices, [prices.index[-1]], window_days=1)
    with pytest.raises(DataSchemaError):
        outside = prices.index[-1] + pd.Timedelta(days=30)
        calculate_trailing_cid1_cross_section(prices, [outside], window_days=4)
    with pytest.raises(DataSchemaError):
        # No date has enough history for the window.
        calculate_trailing_cid1_cross_section(prices, [prices.index[2]], window_days=6)
