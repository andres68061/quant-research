"""Tests for /simulation/sharpe-comparison multi-ratio calibration."""

from api.routes.simulation import sharpe_comparison


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
