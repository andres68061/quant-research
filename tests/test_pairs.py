"""Unit tests for pairs cointegration signals and backtest runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.signals.pairs import (
    align_pair_log_prices,
    engle_granger_test,
    pairs_position_from_zscore,
    rolling_hedge_ratio,
    rolling_spread_zscore,
    spread_from_hedge,
)
from core.strategies.pairs_runner import run_pairs_cointegration_backtest


def _cointegrated_panel(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Synthetic pair with known hedge ratio ~1.5 plus stationary noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n, tz="America/New_York")
    x = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    noise = rng.normal(0.0, 0.005, n)
    y = np.exp(np.log(x) * 1.5 + noise)
    return pd.DataFrame({"AAA": y, "BBB": x}, index=dates)


class TestAlignPair:
    def test_missing_symbol_raises(self) -> None:
        prices = _cointegrated_panel(50)
        with pytest.raises(KeyError):
            align_pair_log_prices(prices, "AAA", "MISSING")


class TestEngleGranger:
    def test_cointegrated_pair_rejects_unit_root_at_5pct(self) -> None:
        prices = _cointegrated_panel()
        log_y, log_x = align_pair_log_prices(prices, "AAA", "BBB")
        result = engle_granger_test(log_y, log_x)
        assert result["adf_pvalue"] < 0.05
        assert 1.2 < result["hedge_ratio"] < 1.8


class TestRollingHedgeAndZ:
    def test_hedge_warm_up_nan(self) -> None:
        prices = _cointegrated_panel(300)
        log_y, log_x = align_pair_log_prices(prices, "AAA", "BBB")
        beta = rolling_hedge_ratio(log_y, log_x, window=100)
        assert beta.isna().sum() == 99
        assert beta.notna().sum() > 0

    def test_zscore_near_zero_mean_in_stable_window(self) -> None:
        prices = _cointegrated_panel(500)
        log_y, log_x = align_pair_log_prices(prices, "AAA", "BBB")
        beta = rolling_hedge_ratio(log_y, log_x, window=120)
        spread = spread_from_hedge(log_y, log_x, beta)
        z = rolling_spread_zscore(spread, window=40)
        tail = z.dropna().iloc[-100:]
        assert abs(float(tail.mean())) < 0.5


class TestPositions:
    def test_entry_and_exit_hysteresis(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=8)
        z = pd.Series([0.0, -2.1, -1.5, -0.4, 0.0, 2.2, 1.0, 0.3], index=idx)
        pos = pairs_position_from_zscore(z, entry_z=2.0, exit_z=0.5)
        assert pos.iloc[1] == 1.0
        assert pos.iloc[2] == 1.0  # still in trade
        assert pos.iloc[3] == 0.0  # exited
        assert pos.iloc[5] == -1.0
        assert pos.iloc[7] == 0.0

    def test_rejects_bad_thresholds(self) -> None:
        z = pd.Series([0.0, 1.0])
        with pytest.raises(ValueError):
            pairs_position_from_zscore(z, entry_z=0.5, exit_z=1.0)


class TestPairsRunner:
    def test_returns_finite_series(self) -> None:
        prices = _cointegrated_panel(700)
        out = run_pairs_cointegration_backtest(
            prices,
            symbol_y="AAA",
            symbol_x="BBB",
            hedge_window=120,
            zscore_window=40,
            entry_z=2.0,
            exit_z=0.5,
            transaction_cost=0.001,
        )
        assert len(out["net_returns"]) > 50
        assert np.isfinite(out["net_returns"]).all()
        assert "engle_granger" in out["diagnostics"]
        assert out["diagnostics"]["engle_granger"]["adf_pvalue"] < 0.05
