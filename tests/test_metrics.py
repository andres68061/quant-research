"""
Unit tests for core.metrics.performance and core.metrics.risk.
"""

import numpy as np
import pandas as pd
import pytest

from core.metrics.performance import (
    calculate_calmar_ratio,
    calculate_cumulative_returns,
    calculate_drawdown,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_performance_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    format_performance_table,
)
from core.metrics.risk import (
    calculate_all_var,
    calculate_historical_var,
    calculate_monte_carlo_var,
    calculate_parametric_var,
)


@pytest.fixture()
def daily_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(
        rng.normal(0.0004, 0.012, 500),
        index=pd.bdate_range("2023-01-02", periods=500),
        name="returns",
    )


@pytest.fixture()
def benchmark_returns(daily_returns: pd.Series) -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(
        rng.normal(0.0003, 0.011, len(daily_returns)),
        index=daily_returns.index,
        name="benchmark",
    )


class TestCumulativeReturns:
    def test_monotonic_positive(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        cum = calculate_cumulative_returns(returns)
        assert cum.iloc[-1] > cum.iloc[0]

    def test_zero_returns(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        cum = calculate_cumulative_returns(returns)
        assert np.allclose(cum.values, [1.0, 1.0, 1.0])


class TestDrawdown:
    def test_always_non_positive(self, daily_returns):
        dd = calculate_drawdown(daily_returns)
        assert (dd <= 0).all()

    def test_max_drawdown_is_min_of_series(self, daily_returns):
        dd = calculate_drawdown(daily_returns)
        assert calculate_max_drawdown(daily_returns) == pytest.approx(dd.min())


class TestSharpeRatio:
    def test_zero_vol_returns_zero(self):
        returns = pd.Series([0.01, 0.01, 0.01])
        assert calculate_sharpe_ratio(returns) == 0.0

    def test_positive_mean_positive_sharpe(self, daily_returns):
        assert calculate_sharpe_ratio(daily_returns) > 0

    def test_risk_free_reduces_sharpe(self, daily_returns):
        sr_no_rf = calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0)
        sr_with_rf = calculate_sharpe_ratio(daily_returns, risk_free_rate=0.05)
        assert sr_with_rf < sr_no_rf


class TestSortinoRatio:
    def test_only_positive_returns_zero(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.005])
        assert calculate_sortino_ratio(returns) == 0.0

    def test_mixed_returns(self, daily_returns):
        sortino = calculate_sortino_ratio(daily_returns)
        assert isinstance(sortino, float)


class TestCalmarRatio:
    def test_no_drawdown_returns_zero(self):
        returns = pd.Series([0.01, 0.01, 0.01])
        assert calculate_calmar_ratio(returns) == 0.0


class TestInformationRatio:
    def test_identical_returns_zero(self, daily_returns):
        ir = calculate_information_ratio(daily_returns, daily_returns)
        assert ir == 0.0


class TestPerformanceMetrics:
    def test_all_keys_present(self, daily_returns):
        metrics = calculate_performance_metrics(daily_returns)
        expected = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "n_periods",
        }
        assert expected.issubset(metrics.keys())

    def test_with_benchmark(self, daily_returns, benchmark_returns):
        metrics = calculate_performance_metrics(
            daily_returns, benchmark_returns=benchmark_returns
        )
        assert "beta" in metrics
        assert "alpha" in metrics
        assert "information_ratio" in metrics

    def test_empty_returns(self):
        metrics = calculate_performance_metrics(pd.Series(dtype=float))
        assert metrics["sharpe_ratio"] == 0.0


class TestFormatTable:
    def test_returns_dataframe(self, daily_returns):
        metrics = calculate_performance_metrics(daily_returns)
        table = format_performance_table(metrics)
        assert isinstance(table, pd.DataFrame)
        assert len(table) > 0


class TestHistoricalVar:
    def test_positive_var(self, daily_returns):
        result = calculate_historical_var(daily_returns.values, confidence=95)
        assert result["var"] > 0
        assert result["cvar"] >= result["var"]


class TestParametricVar:
    def test_positive_var(self, daily_returns):
        result = calculate_parametric_var(daily_returns.values, confidence=95)
        assert result["var"] > 0


class TestMonteCarloVar:
    def test_reproducible(self, daily_returns):
        r1 = calculate_monte_carlo_var(daily_returns.values, seed=1)
        r2 = calculate_monte_carlo_var(daily_returns.values, seed=1)
        assert r1["var"] == pytest.approx(r2["var"])


class TestAllVar:
    def test_all_methods_present(self, daily_returns):
        result = calculate_all_var(daily_returns.values)
        assert set(result.keys()) == {"historical", "parametric", "monte_carlo"}
        for method in result.values():
            assert "var" in method
            assert "cvar" in method
