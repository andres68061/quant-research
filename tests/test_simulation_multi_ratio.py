"""Tests for /simulation/sharpe-comparison multi-ratio calibration."""

import numpy as np

from api.routes.simulation import (
    _metrics_and_lists,
    _realized_sortino,
    sharpe_comparison,
)


def test_sharpe_comparison_three_blocks_match_target() -> None:
    r = sharpe_comparison(target_sharpe=1.2, n_days=900, seed=11)
    assert r.target == 1.2
    assert len(r.by_sharpe) == len(r.by_sortino) == len(r.by_calmar) == 5
    for inv in r.by_sharpe:
        assert abs(inv.metrics.sharpe - 1.2) < 1e-3
    for inv in r.by_sortino:
        assert abs(inv.metrics.sortino - 1.2) < 1e-3
    for inv in r.by_calmar:
        assert abs(inv.metrics.calmar - 1.2) < 1e-3


def test_same_names_and_colors_across_blocks() -> None:
    r = sharpe_comparison(target_sharpe=1.5, n_days=600, seed=3)
    names_s = [i.name for i in r.by_sharpe]
    assert names_s == [i.name for i in r.by_sortino] == [i.name for i in r.by_calmar]
    colors_s = [i.color for i in r.by_sharpe]
    assert colors_s == [i.color for i in r.by_sortino] == [i.color for i in r.by_calmar]


def test_sortino_single_downside_day_matches_metrics() -> None:
    """One negative day must use the same Sortino path in calibrate + report."""
    returns = np.array([0.01, -0.02, 0.015, 0.01, 0.005])
    rf = 0.0
    realized = _realized_sortino(returns, rf)
    reported = _metrics_and_lists(returns, rf).metrics.sortino
    assert realized == reported
    # Single downside obs → downside std (ddof=0) is 0 → shared sentinel path.
    assert abs(realized) >= 1e5


def test_sortino_multiple_downside_days_finite() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=252)
    rf = 0.0
    realized = _realized_sortino(returns, rf)
    reported = _metrics_and_lists(returns, rf).metrics.sortino
    assert round(realized, 4) == reported
    assert np.isfinite(realized)
    assert abs(realized) < 1e5
