"""Tests for core.data.factors.fama_french.prepare_ff5_view."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.factors.fama_french import prepare_ff5_view


def _ff5_frame(n_days: int = 504, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n_days, name="date")
    columns = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    return pd.DataFrame(
        rng.normal(0.0003, 0.01, (n_days, len(columns))), index=index, columns=columns
    )


class TestPrepareFf5View:
    def test_growth_is_cumulative_product_at_month_ends(self) -> None:
        ff5 = _ff5_frame()
        growth, _ = prepare_ff5_view(ff5)
        expected_last = (1.0 + ff5["mkt_rf"]).prod()
        assert abs(growth["mkt_rf"].iloc[-1] - expected_last) < 1e-10
        assert growth.index.is_monotonic_increasing
        assert "rf" not in growth.columns

    def test_start_filter_drops_earlier_rows(self) -> None:
        ff5 = _ff5_frame()
        growth_full, _ = prepare_ff5_view(ff5)
        growth_late, _ = prepare_ff5_view(ff5, start="2021-01-01")
        assert growth_late.index.min() > growth_full.index.min()

    def test_stats_have_expected_shape_and_finite_values(self) -> None:
        _, stats = prepare_ff5_view(_ff5_frame())
        assert list(stats.index) == ["mkt_rf", "smb", "hml", "rmw", "cma"]
        assert list(stats.columns) == ["annualized_return", "annualized_volatility", "sharpe_ratio"]
        assert np.isfinite(stats.values).all()

    def test_all_nan_rows_dropped(self) -> None:
        ff5 = _ff5_frame(n_days=100)
        ff5.iloc[:10] = np.nan
        growth, _ = prepare_ff5_view(ff5)
        assert np.isfinite(growth.values).all()
