"""Tests for downside-focused performance metrics."""

import pandas as pd

from core.metrics.performance import (
    calculate_loss_probability,
    calculate_performance_metrics,
    calculate_time_underwater,
)


def test_calculate_time_underwater_counts_longest_drawdown_streak() -> None:
    returns = pd.Series([0.10, -0.05, 0.00, 0.06, -0.02, 0.03])

    assert calculate_time_underwater(returns) == 2


def test_calculate_loss_probability_is_bounded_and_reproducible() -> None:
    returns = pd.Series([0.01, -0.02, 0.005, -0.004, 0.003])

    first = calculate_loss_probability(returns, 21, n_bootstrap=500, seed=7)
    second = calculate_loss_probability(returns, 21, n_bootstrap=500, seed=7)

    assert 0.0 <= first <= 1.0
    assert first == second


def test_loss_probability_rises_with_horizon_for_negative_mean_returns() -> None:
    returns = pd.Series([-0.002, -0.001, 0.0005, -0.003, 0.001])

    short_horizon = calculate_loss_probability(returns, 5, n_bootstrap=1000, seed=42)
    long_horizon = calculate_loss_probability(returns, 63, n_bootstrap=1000, seed=42)

    assert long_horizon >= short_horizon


def test_calculate_performance_metrics_includes_downside_keys() -> None:
    returns = pd.Series([0.01, -0.02, 0.005, -0.004, 0.003] * 20)

    metrics = calculate_performance_metrics(
        returns,
        loss_probability_horizons=(21,),
        loss_probability_bootstraps=200,
        loss_probability_seed=1,
    )

    assert "cvar_95" in metrics
    assert "cvar_99" in metrics
    assert "time_underwater_days" in metrics
    assert "loss_probability_21d" in metrics
    assert metrics["cvar_99"] >= metrics["cvar_95"]
