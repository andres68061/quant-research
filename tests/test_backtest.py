"""
Unit tests for core.backtest.portfolio and core.backtest.walkforward.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from core.backtest.portfolio import (
    _infer_abs_bound,
    _normalize_rebalance_freq,
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


class TestInferAbsBound:
    """Factor-specific default bounds for outlier filtering."""

    def test_prefix_and_special_cases(self) -> None:
        assert _infer_abs_bound("mom_12_1") == 10.0
        assert _infer_abs_bound("mom_6_1") == 10.0
        assert _infer_abs_bound("vol_60d") == 5.0
        assert _infer_abs_bound("beta_60d") == float("inf")
        assert _infer_abs_bound("log_market_cap") == float("inf")
        assert _infer_abs_bound("unknown_factor") == 10.0


class TestCreateSignals:
    def test_signal_values(self, factors_df):
        signals = create_signals_from_factor(factors_df, "mom_12_1", min_stocks=3)
        unique_signals = set(signals["signal"].unique())
        assert unique_signals.issubset({-1, 0, 1})

    def test_long_only(self, factors_df):
        signals = create_signals_from_factor(factors_df, "mom_12_1", long_only=True, min_stocks=3)
        assert (signals["signal"] >= 0).all()

    def test_beta_extreme_values_not_clipped_by_default_bound(self) -> None:
        """Beta uses no absolute cap; large |beta| rows remain valid for ranking."""
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2023-01-03"), "AAA"),
                (pd.Timestamp("2023-01-03"), "BBB"),
            ],
            names=["date", "symbol"],
        )
        factors = pd.DataFrame({"beta_60d": [15.0, 1.0]}, index=idx)
        signals = create_signals_from_factor(
            factors,
            "beta_60d",
            min_stocks=2,
            top_pct=0.5,
            bottom_pct=0.5,
        )
        assert set(signals["signal"].unique()) == {-1, 1}
        assert (signals["signal"] != 0).all()

    def test_unknown_factor_uses_default_bound_ten(self) -> None:
        """Unknown columns use abs < 10; extreme values are dropped."""
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2023-01-03"), "A"),
                (pd.Timestamp("2023-01-03"), "B"),
                (pd.Timestamp("2023-01-03"), "C"),
            ],
            names=["date", "symbol"],
        )
        factors = pd.DataFrame({"custom_x": [0.1, 0.2, 15.0]}, index=idx)
        signals_default = create_signals_from_factor(
            factors, "custom_x", min_stocks=2, top_pct=0.33, bottom_pct=0.33
        )
        assert (signals_default.loc[(slice(None), "C"), "signal"] == 0).all()

    def test_max_abs_value_overrides_inferred_bound(self) -> None:
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2023-01-03"), "A"),
                (pd.Timestamp("2023-01-03"), "B"),
                (pd.Timestamp("2023-01-03"), "C"),
            ],
            names=["date", "symbol"],
        )
        factors = pd.DataFrame({"custom_x": [0.1, 0.2, 15.0]}, index=idx)
        signals = create_signals_from_factor(
            factors,
            "custom_x",
            min_stocks=2,
            top_pct=0.33,
            bottom_pct=0.33,
            max_abs_value=20.0,
        )
        assert (signals.loc[(slice(None), "C"), "signal"] != 0).any()


class TestPortfolioReturns:
    def test_output_columns(self, factors_df, price_panel):
        signals = create_signals_from_factor(factors_df, "mom_12_1", min_stocks=3)
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
        signals = create_signals_from_factor(factors_df, "mom_12_1", min_stocks=3)
        result = calculate_portfolio_returns(signals, price_panel, transaction_cost=0.001)
        cum_net = (1 + result["net_return"]).prod()
        cum_gross = (1 + result["gross_return"]).prod()
        assert cum_net <= cum_gross + 1e-10


class TestWeightedPortfolio:
    def test_equal_weight(self, price_panel):
        returns = create_weighted_portfolio(price_panel, ["AAPL", "MSFT"], "equal")
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
        wfv = WalkForwardValidator(initial_train_days=63, test_days=5, max_splits=100)
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
        wfv = WalkForwardValidator(initial_train_days=63, test_days=5, max_splits=10)
        splits = wfv.create_splits(df)
        assert len(splits) == 10


class TestRebalanceFreqNormalization:
    """Pandas 2.2+ resample alias handling (M -> ME, etc.)."""

    def test_legacy_aliases_map(self) -> None:
        assert _normalize_rebalance_freq("M") == "ME"
        assert _normalize_rebalance_freq("Q") == "QE"
        assert _normalize_rebalance_freq("Y") == "YE"
        assert _normalize_rebalance_freq("A") == "YE"
        assert _normalize_rebalance_freq("ME") == "ME"
        assert _normalize_rebalance_freq("D") == "D"

    def test_calculate_portfolio_legacy_m_emits_no_future_warning(
        self, factors_df: pd.DataFrame, price_panel: pd.DataFrame
    ) -> None:
        signals = create_signals_from_factor(factors_df, "mom_12_1", min_stocks=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculate_portfolio_returns(
                signals,
                price_panel,
                rebalance_freq="M",
                transaction_cost=0.001,
            )
        future = [x for x in w if issubclass(x.category, FutureWarning)]
        assert not future, [str(x.message) for x in future]
