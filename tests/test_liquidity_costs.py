"""Tests for dollar-ADV cost schedule and liquidity-scaled portfolio costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtest.portfolio import calculate_portfolio_returns
from core.data.liquidity import cost_bps_from_dollar_adv


class TestCostSchedule:
    def test_buckets(self) -> None:
        assert cost_bps_from_dollar_adv(200e6) == 0.0005
        assert cost_bps_from_dollar_adv(50e6) == 0.0010
        assert cost_bps_from_dollar_adv(10e6) == 0.0020
        assert cost_bps_from_dollar_adv(1e6) == 0.0040
        assert cost_bps_from_dollar_adv(float("nan")) == 0.0040


def _signals_all_dates(
    dates: pd.DatetimeIndex,
    symbol_signals: dict[str, float],
) -> pd.DataFrame:
    """Signal panel on every date (zeros except the intended ranks).

    ``calculate_portfolio_returns`` intersects signal dates with returns; a
    single-date signal frame collapses the backtest to one row and never
    charges turnover. Real factor signals cover the full calendar.
    """
    symbols = list(symbol_signals)
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    out = pd.DataFrame({"signal": 0.0}, index=idx)
    for symbol, value in symbol_signals.items():
        out.loc[(slice(None), symbol), "signal"] = value
    return out


class TestLiquidityScaledCosts:
    def test_illiquid_names_cost_more_than_flat(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=60, tz="America/New_York")
        prices = pd.DataFrame(
            {
                "LIQUID": np.linspace(100, 110, len(dates)),
                "ILLIQ": np.linspace(20, 22, len(dates)),
            },
            index=dates,
        )
        # Long both equally on every day; ME rebalance enters from cash on first ME.
        signals = _signals_all_dates(dates, {"LIQUID": 1.0, "ILLIQ": 1.0})
        dollar_adv = pd.DataFrame(
            {"LIQUID": 200e6, "ILLIQ": 1e6},
            index=dates,
        )

        flat = calculate_portfolio_returns(signals, prices, transaction_cost=0.001)
        scaled = calculate_portfolio_returns(
            signals, prices, transaction_cost=0.001, dollar_adv=dollar_adv
        )
        cost_days_flat = flat.loc[flat["transaction_cost"] > 0, "transaction_cost"]
        cost_days_scaled = scaled.loc[scaled["transaction_cost"] > 0, "transaction_cost"]
        assert len(cost_days_flat) >= 1
        assert float(cost_days_scaled.sum()) > float(cost_days_flat.sum())

    def test_flat_path_unchanged_without_adv(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=40, tz="America/New_York")
        prices = pd.DataFrame({"A": np.linspace(100, 105, len(dates))}, index=dates)
        signals = _signals_all_dates(dates, {"A": 1.0})
        a = calculate_portfolio_returns(signals, prices, transaction_cost=0.001)
        b = calculate_portfolio_returns(signals, prices, transaction_cost=0.001, dollar_adv=None)
        pd.testing.assert_frame_equal(a, b)
