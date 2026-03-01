"""
Unit tests for core.signals.momentum and core.signals.factor_signals.
"""

import numpy as np
import pandas as pd
import pytest

from core.signals.momentum import (
    analyze_momentum_grid_search,
    bootstrap_significance_test,
    calculate_sortino_slopes,
    get_current_regime,
    prepare_ml_features,
)


@pytest.fixture()
def daily_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(
        rng.normal(0.0004, 0.012, 600),
        index=pd.bdate_range("2022-01-03", periods=600),
        name="returns",
    )


class TestSortinoSlopes:
    def test_output_length(self):
        s = pd.Series(range(100), dtype=float)
        slopes = calculate_sortino_slopes(s, 5)
        assert len(slopes) == len(s)

    def test_first_values_nan(self):
        s = pd.Series(range(20), dtype=float)
        slopes = calculate_sortino_slopes(s, 5)
        assert slopes.iloc[:5].isna().all()


class TestGridSearch:
    def test_returns_dataframe(self, daily_returns):
        result = analyze_momentum_grid_search(
            daily_returns, sortino_window=60, min_signals=5
        )
        assert isinstance(result, pd.DataFrame)

    def test_hit_rate_bounded(self, daily_returns):
        result = analyze_momentum_grid_search(
            daily_returns, sortino_window=60, min_signals=5
        )
        if not result.empty:
            assert (result["Z (hit_rate)"] >= 0).all()
            assert (result["Z (hit_rate)"] <= 100).all()


class TestSignificance:
    def test_returns_dict(self, daily_returns):
        result = bootstrap_significance_test(
            daily_returns, x=10, k=10, sortino_window=60, n_bootstraps=50
        )
        assert isinstance(result, dict)
        assert "p_value" in result
        assert "significant" in result


class TestPrepareMLFeatures:
    def test_has_target(self, daily_returns):
        features = prepare_ml_features(
            daily_returns, sortino_window=60, forecast_horizon=5
        )
        assert "target" in features.columns

    def test_no_nans(self, daily_returns):
        features = prepare_ml_features(
            daily_returns, sortino_window=60, forecast_horizon=5
        )
        assert not features.isna().any().any()


class TestCurrentRegime:
    def test_returns_dict_or_none(self, daily_returns):
        result = get_current_regime(daily_returns, x=10, k=5, sortino_window=60)
        if result is not None:
            assert "current_sortino" in result
            assert "strong_momentum" in result

    def test_insufficient_data(self):
        short_returns = pd.Series(
            [0.01] * 10,
            index=pd.bdate_range("2023-01-02", periods=10),
        )
        result = get_current_regime(short_returns, x=20, k=5, sortino_window=60)
        assert result is None
