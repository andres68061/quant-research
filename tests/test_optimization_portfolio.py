"""Tests for the walk-forward tangency/min-variance portfolio backtest.

Regression-motivated: `/portfolio/optimize` fits weights on `[start, end]`
and the UI's "Simulate" evaluates those same weights over the identical
window -- in-sample look-ahead. `run_walk_forward_tangency` is the fix:
weights are always fit on a trailing window ending strictly before the
period whose returns they are then scored on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.optimization.portfolio import run_walk_forward_tangency


def _two_asset_panel(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n, tz="America/New_York")
    # HI: higher drift, moderate vol. LO: near-zero drift, low vol.
    hi = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.015, n)))
    lo = 100 * np.exp(np.cumsum(rng.normal(0.0000, 0.005, n)))
    return pd.DataFrame({"HI": hi, "LO": lo}, index=dates)


class TestRunWalkForwardTangency:
    def test_produces_non_overlapping_back_to_back_periods(self) -> None:
        prices = _two_asset_panel()
        out = run_walk_forward_tangency(
            prices,
            ["HI", "LO"],
            start=prices.index[300],
            end=prices.index[-1],
            lookback_months=12,
            rebalance_months=6,
        )
        assert len(out["periods"]) >= 3
        for prev, nxt in zip(out["periods"], out["periods"][1:], strict=False):
            assert prev["hold_end"] == nxt["hold_start"]
        assert out["net_returns"].index.is_monotonic_increasing
        assert not out["net_returns"].index.has_duplicates

    def test_weights_are_causal_not_affected_by_future_prices(self) -> None:
        """Perturbing prices strictly after a period's hold window must not
        change that period's chosen weights (no look-ahead into the future)."""
        prices = _two_asset_panel(seed=5)
        base = run_walk_forward_tangency(
            prices,
            ["HI", "LO"],
            start=prices.index[300],
            end=prices.index[-1],
            lookback_months=12,
            rebalance_months=6,
        )
        first_hold_end = pd.Timestamp(base["periods"][0]["hold_end"], tz=prices.index.tz)
        perturbed = prices.copy()
        after_mask = perturbed.index >= first_hold_end
        perturbed.loc[after_mask, "HI"] = perturbed.loc[after_mask, "HI"] * 3.0
        bumped = run_walk_forward_tangency(
            perturbed,
            ["HI", "LO"],
            start=prices.index[300],
            end=prices.index[-1],
            lookback_months=12,
            rebalance_months=6,
        )
        assert base["periods"][0]["weights"] == bumped["periods"][0]["weights"]

    def test_weights_sum_to_approximately_one(self) -> None:
        prices = _two_asset_panel()
        out = run_walk_forward_tangency(
            prices,
            ["HI", "LO"],
            start=prices.index[300],
            end=prices.index[-1],
            lookback_months=12,
            rebalance_months=6,
        )
        for p in out["periods"]:
            total = sum(p["weights"].values())
            assert abs(total - 1.0) < 1e-6

    def test_min_variance_kind_runs(self) -> None:
        prices = _two_asset_panel()
        out = run_walk_forward_tangency(
            prices,
            ["HI", "LO"],
            start=prices.index[300],
            end=prices.index[-1],
            lookback_months=12,
            rebalance_months=6,
            portfolio_kind="min_variance",
        )
        assert len(out["net_returns"]) > 0

    def test_rejects_bad_params(self) -> None:
        prices = _two_asset_panel(n=100)
        with pytest.raises(ValueError):
            run_walk_forward_tangency(
                prices,
                ["HI", "LO"],
                start=prices.index[0],
                end=prices.index[-1],
                lookback_months=1,
            )
        with pytest.raises(ValueError):
            run_walk_forward_tangency(
                prices,
                ["HI", "LO"],
                start=prices.index[-1],
                end=prices.index[0],
            )
        with pytest.raises(ValueError):
            run_walk_forward_tangency(
                prices,
                ["HI", "LO"],
                start=prices.index[0],
                end=prices.index[-1],
                portfolio_kind="bogus",
            )
