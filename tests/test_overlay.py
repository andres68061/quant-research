"""Tests for core.backtest.overlay."""

import numpy as np
import pandas as pd
import pytest

from core.backtest.overlay import apply_exposure_overlay


@pytest.fixture()
def book_returns() -> pd.Series:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=100, tz="America/New_York")
    return pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)


class TestApplyExposureOverlay:
    def test_full_exposure_reproduces_book_minus_entry_cost(self, book_returns) -> None:
        exposure = pd.Series(1.0, index=book_returns.index)
        out = apply_exposure_overlay(book_returns, exposure, transaction_cost=0.001)
        # Day 0: no lagged signal yet → flat, no cost. Day 1: enter (|Δe|=1).
        assert out["net_return"].iloc[0] == 0.0
        entry_cost = 1.0 * 2.0 / 2.0 * 0.001
        assert out["net_return"].iloc[1] == pytest.approx(book_returns.iloc[1] - entry_cost)
        pd.testing.assert_series_equal(
            out["net_return"].iloc[2:], book_returns.iloc[2:], check_names=False
        )

    def test_zero_exposure_is_flat_and_free(self, book_returns) -> None:
        exposure = pd.Series(0.0, index=book_returns.index)
        out = apply_exposure_overlay(book_returns, exposure)
        assert (out["net_return"] == 0.0).all()
        assert (out["scaling_cost"] == 0.0).all()

    def test_signal_is_lagged_one_day(self, book_returns) -> None:
        # Exposure drops to 0 on day k; the cut must take effect on day k+1.
        k = 50
        exposure = pd.Series(1.0, index=book_returns.index)
        exposure.iloc[k:] = 0.0
        out = apply_exposure_overlay(book_returns, exposure, transaction_cost=0.0)
        assert out["net_return"].iloc[k] == pytest.approx(book_returns.iloc[k])
        assert out["net_return"].iloc[k + 1] == 0.0

    def test_exposure_change_charged(self, book_returns) -> None:
        exposure = pd.Series(1.0, index=book_returns.index)
        exposure.iloc[50:] = 0.5
        out = apply_exposure_overlay(
            book_returns, exposure, transaction_cost=0.001, gross_leverage=2.0
        )
        assert out["scaling_cost"].iloc[51] == pytest.approx(0.5 * 2.0 / 2.0 * 0.001)

    def test_naive_exposure_index_aligned_to_tz_aware_book(self, book_returns) -> None:
        naive_idx = book_returns.index.tz_localize(None)
        exposure = pd.Series(0.75, index=naive_idx)
        out = apply_exposure_overlay(book_returns, exposure)
        assert out["exposure"].iloc[5] == 0.75

    def test_out_of_range_exposure_raises(self, book_returns) -> None:
        exposure = pd.Series(1.5, index=book_returns.index)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            apply_exposure_overlay(book_returns, exposure)

    def test_empty_book(self) -> None:
        idx = pd.DatetimeIndex([], tz="America/New_York")
        out = apply_exposure_overlay(
            pd.Series(dtype=float, index=idx), pd.Series(dtype=float, index=idx)
        )
        assert out.empty
