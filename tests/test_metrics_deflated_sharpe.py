"""Tests for Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado)."""

import numpy as np
import pandas as pd
import pytest

from core.metrics.deflated_sharpe import (
    calculate_deflated_sharpe_ratio,
    calculate_probabilistic_sharpe_ratio,
    expected_max_sharpe_under_null,
)


def _normal_returns(n: int, mean: float, std: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def test_calculate_probabilistic_sharpe_ratio_high_for_strong_long_track_record() -> None:
    # Per-period SR ~ 0.1 over 2000 obs: overwhelming evidence of skill vs 0.
    returns = _normal_returns(2000, 0.001, 0.01)

    assert calculate_probabilistic_sharpe_ratio(returns) > 0.99


def test_calculate_probabilistic_sharpe_ratio_near_half_when_sr_equals_benchmark() -> None:
    returns = _normal_returns(1000, 0.001, 0.01, seed=3)
    sr = float(returns.mean() / returns.std(ddof=1))

    psr = calculate_probabilistic_sharpe_ratio(returns, benchmark_sharpe=sr)

    assert psr == pytest.approx(0.5, abs=1e-6)


def test_calculate_probabilistic_sharpe_ratio_increases_with_sample_length() -> None:
    # Mean is 0.5 sigma so the sample Sharpe is positive at any seed; more
    # observations of a positive-SR series must increase the PSR.
    short = _normal_returns(100, 0.005, 0.01, seed=7)
    assert float(short.mean()) > 0
    long = pd.Series(np.tile(short.to_numpy(), 10))

    assert calculate_probabilistic_sharpe_ratio(long) > calculate_probabilistic_sharpe_ratio(short)


def test_calculate_probabilistic_sharpe_ratio_edge_cases() -> None:
    assert calculate_probabilistic_sharpe_ratio(pd.Series(dtype=float)) == 0.0
    assert calculate_probabilistic_sharpe_ratio(pd.Series([0.01, 0.02])) == 0.0
    # Zero variance -> no information about SR.
    assert calculate_probabilistic_sharpe_ratio(pd.Series([0.01] * 50)) == 0.0


def test_expected_max_sharpe_under_null_grows_with_trials() -> None:
    v = 0.01
    e3 = expected_max_sharpe_under_null(3, v)
    e10 = expected_max_sharpe_under_null(10, v)
    e100 = expected_max_sharpe_under_null(100, v)

    assert 0.0 < e3 < e10 < e100


def test_expected_max_sharpe_under_null_zero_for_single_trial_or_no_variance() -> None:
    assert expected_max_sharpe_under_null(1, 0.5) == 0.0
    assert expected_max_sharpe_under_null(10, 0.0) == 0.0


def test_calculate_deflated_sharpe_ratio_deflation_bites_with_many_trials() -> None:
    """The same track record is less convincing when it was the best of many
    dispersed trials than when it was the only trial."""
    returns = _normal_returns(500, 0.0005, 0.01, seed=11)
    psr_alone = calculate_probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)

    sr = float(returns.mean() / returns.std(ddof=1))
    many_trials = [sr, -0.02, 0.01, -0.05, 0.03, -0.01, 0.02, -0.03]
    out = calculate_deflated_sharpe_ratio(returns, many_trials)

    assert out["n_trials"] == 8
    assert out["expected_max_sharpe"] > 0.0
    assert out["dsr"] < psr_alone


def test_calculate_deflated_sharpe_ratio_single_trial_equals_psr_vs_zero() -> None:
    returns = _normal_returns(500, 0.0005, 0.01, seed=13)
    out = calculate_deflated_sharpe_ratio(returns, [0.05])

    assert out["expected_max_sharpe"] == 0.0
    assert out["dsr"] == pytest.approx(
        calculate_probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0), abs=1e-12
    )
