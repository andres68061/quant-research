"""Tests for core.strategies registry and factor runner."""

import numpy as np
import pandas as pd
import pytest

from core.backtest.portfolio import calculate_portfolio_returns, create_signals_from_factor
from core.strategies import (
    get_strategy,
    list_strategies,
    run_factor_cross_section_backtest,
)


@pytest.fixture()
def price_panel() -> pd.DataFrame:
    """Wide-format price panel (date x symbols)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=300)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    prices = pd.DataFrame(
        100 * np.exp(rng.normal(0.0004, 0.015, (300, 5)).cumsum(axis=0)),
        index=dates,
        columns=symbols,
    )
    return prices


@pytest.fixture()
def factors_df(price_panel: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex (date, symbol) factor DataFrame."""
    returns = price_panel.pct_change()
    mom = returns.rolling(60).mean()
    records = []
    for date in mom.index[60:]:
        for symbol in mom.columns:
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "mom_12_1": mom.loc[date, symbol],
                }
            )
    df = pd.DataFrame(records).set_index(["date", "symbol"])
    return df


class TestRegistry:
    def test_list_contains_expected_ids(self) -> None:
        ids = {m.id for m in list_strategies()}
        assert "factor_cross_section" in ids
        assert "ml_commodity_direction" in ids

    def test_get_strategy(self) -> None:
        m = get_strategy("factor_cross_section")
        assert m.id == "factor_cross_section"
        assert m.post_path == "/run-backtest"

    def test_get_strategy_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_strategy("not_a_real_strategy")


class TestFactorRunner:
    def test_matches_direct_pipeline(
        self, factors_df: pd.DataFrame, price_panel: pd.DataFrame
    ) -> None:
        """Regression: runner output equals manual slice + signals + portfolio."""
        factor_col = "mom_12_1"
        start = pd.Timestamp(factors_df.index.get_level_values("date").min())
        end = pd.Timestamp(factors_df.index.get_level_values("date").max())

        f_dates = factors_df.index.get_level_values("date")
        factors_slice = factors_df[(f_dates >= start) & (f_dates <= end)]
        prices_slice = price_panel[
            (price_panel.index >= start) & (price_panel.index <= end)
        ]

        signals = create_signals_from_factor(
            factors_slice, factor_col, min_stocks=3
        )
        results = calculate_portfolio_returns(
            signals,
            prices_slice,
            rebalance_freq="ME",
            transaction_cost=0.001,
        )
        direct = results["net_return"]

        runner = run_factor_cross_section_backtest(
            factors_df,
            price_panel,
            factor_col=factor_col,
            start=start,
            end=end,
            rebalance_freq="ME",
            transaction_cost=0.001,
            min_stocks=3,
        )

        pd.testing.assert_series_equal(
            direct,
            runner,
            check_names=False,
            rtol=1e-12,
            atol=1e-12,
        )
