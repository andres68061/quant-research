"""
Unit tests for core.backtest.portfolio and core.backtest.walkforward.
"""

import numpy as np
import pandas as pd
import pytest

from core.backtest.portfolio import (
    calculate_portfolio_returns,
    calculate_rolling_metrics,
    create_equal_weight_portfolio,
    create_signals_from_factor,
    create_weighted_portfolio,
)
from core.backtest.walkforward import WalkForwardValidator


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


class TestCreateSignals:
    def test_signal_values(self, factors_df):
        signals = create_signals_from_factor(
            factors_df, "mom_12_1", min_stocks=3
        )
        unique_signals = set(signals["signal"].unique())
        assert unique_signals.issubset({-1, 0, 1})

    def test_long_only(self, factors_df):
        signals = create_signals_from_factor(
            factors_df, "mom_12_1", long_only=True, min_stocks=3
        )
        assert (signals["signal"] >= 0).all()


class TestPortfolioReturns:
    def test_output_columns(self, factors_df, price_panel):
        signals = create_signals_from_factor(
            factors_df, "mom_12_1", min_stocks=3
        )
        result = calculate_portfolio_returns(signals, price_panel)
        expected_cols = {
            "gross_return",
            "transaction_cost",
            "net_return",
            "turnover",
            "n_long",
            "n_short",
            "cash",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_net_leq_gross(self, factors_df, price_panel):
        signals = create_signals_from_factor(
            factors_df, "mom_12_1", min_stocks=3
        )
        result = calculate_portfolio_returns(
            signals, price_panel, transaction_cost=0.001
        )
        cum_net = (1 + result["net_return"]).prod()
        cum_gross = (1 + result["gross_return"]).prod()
        assert cum_net <= cum_gross + 1e-10


class TestWeightedPortfolio:
    def test_equal_weight(self, price_panel):
        returns = create_weighted_portfolio(
            price_panel, ["AAPL", "MSFT"], "equal"
        )
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(price_panel)

    def test_manual_weight(self, price_panel):
        returns = create_weighted_portfolio(
            price_panel,
            ["AAPL", "MSFT"],
            "manual",
            manual_weights={"AAPL": 0.7, "MSFT": 0.3},
        )
        assert isinstance(returns, pd.Series)

    def test_unknown_scheme_raises(self, price_panel):
        with pytest.raises(ValueError, match="Unknown weighting"):
            create_weighted_portfolio(price_panel, ["AAPL"], "magic")


class TestEqualWeightPortfolio:
    def test_returns_series(self, price_panel):
        returns = create_equal_weight_portfolio(price_panel)
        assert isinstance(returns, pd.Series)


class TestRollingMetrics:
    def test_output_columns(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(
            rng.normal(0.0004, 0.01, 300),
            index=pd.bdate_range("2023-01-02", periods=300),
        )
        metrics = calculate_rolling_metrics(returns, window=60)
        expected = {
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "drawdown",
        }
        assert expected == set(metrics.columns)


class TestWalkForwardValidator:
    def test_splits_count(self):
        df = pd.DataFrame(
            {"feature": range(200), "target": [0, 1] * 100},
            index=pd.bdate_range("2023-01-02", periods=200),
        )
        wfv = WalkForwardValidator(
            initial_train_days=63, test_days=5, max_splits=100
        )
        splits = wfv.create_splits(df)
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(test_idx) == 5
            assert train_idx[-1] < test_idx[0]

    def test_insufficient_data(self):
        df = pd.DataFrame(
            {"feature": range(10)},
            index=pd.bdate_range("2023-01-02", periods=10),
        )
        wfv = WalkForwardValidator(initial_train_days=63, test_days=5)
        splits = wfv.create_splits(df)
        assert len(splits) == 0

    def test_max_splits_limit(self):
        df = pd.DataFrame(
            {"feature": range(1000)},
            index=pd.bdate_range("2022-01-03", periods=1000),
        )
        wfv = WalkForwardValidator(
            initial_train_days=63, test_days=5, max_splits=10
        )
        splits = wfv.create_splits(df)
        assert len(splits) == 10
