"""Tests for the /measures-lab synthetic-data explorer endpoint."""

from __future__ import annotations

from api.routes.measures_lab import MeasuresLabRequest, measures_lab


def test_measures_lab_end_to_end() -> None:
    req = MeasuresLabRequest(n_days=400, n_relationship_draws=15, seed=1)
    resp = measures_lab(req)

    assert len(resp.single_stock_examples) == 5
    for s in resp.single_stock_examples:
        assert len(s.prices) == 400
        assert s.prices[0] > 0

    assert resp.portfolio_example.prices
    assert len(resp.portfolio_legs) == 2

    for key in (
        "sharpe_ratio",
        "pain_ratio",
        "martin_ratio",
        "cid1_ratio",
        "typical_period_return",
        "cid2_ratio",
    ):
        assert key in resp.relationship_scatter
        assert len(resp.relationship_scatter[key]) == 15


def test_measures_lab_portfolio_weight_changes_blend() -> None:
    req_a = MeasuresLabRequest(n_days=300, n_relationship_draws=10, seed=7, portfolio_weight_a=1.0)
    req_b = MeasuresLabRequest(n_days=300, n_relationship_draws=10, seed=7, portfolio_weight_a=0.0)

    resp_a = measures_lab(req_a)
    resp_b = measures_lab(req_b)

    # weight_a=1.0 should reproduce leg A's own prices; weight_a=0.0 -> leg B's.
    assert resp_a.portfolio_example.prices == resp_a.portfolio_legs[0].prices
    assert resp_b.portfolio_example.prices == resp_b.portfolio_legs[1].prices


def test_measures_lab_falls_back_on_unknown_leg_names() -> None:
    req = MeasuresLabRequest(
        n_days=300, n_relationship_draws=10, seed=3, portfolio_a="Nope", portfolio_b="AlsoNope"
    )
    resp = measures_lab(req)
    assert resp.portfolio_example.prices
