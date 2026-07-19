"""Tests for downside-focused performance metrics."""

import pandas as pd
import pytest

from core.metrics.performance import (
    calculate_cid1_ratio,
    calculate_cid2_ratio,
    calculate_cost_basis_pain,
    calculate_loss_probability,
    calculate_martin_ratio,
    calculate_pain_index,
    calculate_pain_ratio,
    calculate_performance_metrics,
    calculate_time_underwater,
    calculate_typical_period_return,
    calculate_ulcer_index,
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
    assert "pain_index" in metrics
    assert "pain_ratio" in metrics
    assert "ulcer_index" in metrics
    assert "martin_ratio" in metrics


def test_calculate_performance_metrics_empty_input_includes_pain_keys() -> None:
    metrics = calculate_performance_metrics(pd.Series(dtype=float))

    assert metrics["pain_index"] == 0.0
    assert metrics["pain_ratio"] == 0.0
    assert metrics["ulcer_index"] == 0.0
    assert metrics["martin_ratio"] == 0.0


def test_calculate_pain_index_zero_when_no_drawdown() -> None:
    returns = pd.Series([0.01, 0.02, 0.0, 0.03])

    assert calculate_pain_index(returns) == 0.0
    assert calculate_ulcer_index(returns) == 0.0


def test_calculate_pain_ratio_zero_when_pain_index_zero() -> None:
    returns = pd.Series([0.01, 0.02, 0.0, 0.03])

    assert calculate_pain_ratio(returns) == 0.0
    assert calculate_martin_ratio(returns) == 0.0


def test_calculate_ulcer_index_at_least_pain_index() -> None:
    """RMS drawdown >= mean |drawdown| (quadratic-mean >= arithmetic-mean)."""
    returns = pd.Series([0.02, -0.05, 0.01, -0.08, 0.03, -0.01, 0.06])

    pain = calculate_pain_index(returns)
    ulcer = calculate_ulcer_index(returns)

    assert pain > 0.0
    assert ulcer >= pain


def test_calculate_pain_ratio_rewards_shallower_equal_return_path() -> None:
    """Same total drawdown days, shallower dip -> smaller pain index, larger pain ratio."""
    deep_dip = pd.Series([0.0, -0.10, 0.02, 0.02, 0.02, 0.02])
    shallow_dip = pd.Series([0.0, -0.02, 0.02, 0.02, 0.02, 0.02])

    assert calculate_pain_index(shallow_dip) < calculate_pain_index(deep_dip)
    assert calculate_pain_ratio(shallow_dip) > calculate_pain_ratio(deep_dip)


def test_calculate_cost_basis_pain_uses_initial_investment_not_running_peak() -> None:
    """A path that sets a new high then dips again should not double-count the
    peak-to-trough dip the way calculate_pain_index (peak-relative) would --
    it only measures shortfall below the ORIGINAL 1.0 cost basis."""
    # +50% to a new high, then down to exactly break-even (1.0), then flat.
    returns = pd.Series([0.50, -1 / 3, 0.0, 0.0])

    # Wealth path: 1.0 -> 1.5 -> 1.0 -> 1.0 -> 1.0. Never dips BELOW 1.0.
    assert calculate_cost_basis_pain(returns) == 0.0
    assert calculate_cid1_ratio(returns) == 0.0

    # Now push one more step below cost basis.
    returns_with_loss = pd.Series([0.50, -1 / 3, -0.10, 0.0])
    pain = calculate_cost_basis_pain(returns_with_loss)
    assert pain > 0.0
    # Shortfall on the last two days: day3 wealth = 1.0*0.9 = 0.9 -> shortfall 0.1;
    # day4 wealth unchanged at 0.9 -> shortfall 0.1. Sum = 0.2.
    assert pain == pytest.approx(0.2, abs=1e-9)


def test_calculate_cid1_ratio_is_total_return_not_annualized() -> None:
    """Uses total (compounded) return to date, not an annualized figure."""
    returns = pd.Series([-0.10, 0.30])  # wealth: 1.0 -> 0.9 -> 1.17
    total_return = 1.17 - 1.0
    pain = calculate_cost_basis_pain(returns)  # shortfall only on day 1: 0.1
    assert pain == pytest.approx(0.1, abs=1e-9)
    assert calculate_cid1_ratio(returns) == pytest.approx(total_return / pain, abs=1e-9)


def test_calculate_typical_period_return_matches_worked_example() -> None:
    """100 -> 110 (+10%) -> 132 (+20%): average 1-'year' (period_days=1 block)
    cumulative return is (10% + 20%) / 2 = 15%, NOT an annualized daily mean."""
    returns = pd.Series([0.10, 0.20])

    assert calculate_typical_period_return(returns, period_days=1) == pytest.approx(0.15, abs=1e-9)


def test_calculate_typical_period_return_compounds_within_block() -> None:
    # One block of 2 days: +10% then +10% compounds to 21%, not 20%.
    returns = pd.Series([0.10, 0.10])

    assert calculate_typical_period_return(returns, period_days=2) == pytest.approx(0.21, abs=1e-9)


def test_calculate_typical_period_return_drops_trailing_partial_block() -> None:
    # 5 days at period_days=2 -> 2 full blocks, 1 trailing day dropped.
    returns = pd.Series([0.10, 0.10, 0.05, 0.05, 0.99])

    block1 = 1.10 * 1.10 - 1.0
    block2 = 1.05 * 1.05 - 1.0
    expected = (block1 + block2) / 2
    assert calculate_typical_period_return(returns, period_days=2) == pytest.approx(
        expected, abs=1e-9
    )


def test_calculate_typical_period_return_zero_when_less_than_one_block() -> None:
    returns = pd.Series([0.01, 0.02])
    assert calculate_typical_period_return(returns, period_days=10) == 0.0


def test_calculate_cid2_ratio_zero_when_no_pain() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])
    assert calculate_cid2_ratio(returns, period_days=1) == 0.0


def test_calculate_performance_metrics_includes_new_cost_basis_keys() -> None:
    returns = pd.Series([0.01, -0.02, 0.005, -0.004, 0.003] * 60)
    metrics = calculate_performance_metrics(returns)

    assert "cid1_ratio" in metrics
    assert "typical_period_return" in metrics
    assert "cid2_ratio" in metrics


def test_calculate_performance_metrics_empty_input_includes_new_cost_basis_keys() -> None:
    metrics = calculate_performance_metrics(pd.Series(dtype=float))

    assert metrics["cid1_ratio"] == 0.0
    assert metrics["typical_period_return"] == 0.0
    assert metrics["cid2_ratio"] == 0.0
