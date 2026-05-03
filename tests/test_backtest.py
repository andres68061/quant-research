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
            # This test checks the `_infer_abs_bound` filter for beta, not the
            # 1-day execution lag; use lag=0 so the single-date fixture produces
            # a signal row to assert on.
            signal_lag_days=0,
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
            factors,
            "custom_x",
            min_stocks=2,
            top_pct=0.33,
            bottom_pct=0.33,
            signal_lag_days=0,
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
            signal_lag_days=0,
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


class TestDelistingRealization:
    """
    Guards for Bug 4 + Bug 8 in docs/FACTOR_BACKTEST_AUDIT.md.

    A long position whose price goes to NaN (delisting / suspension) must be
    realized as a -100% return on the position weight on the first NaN day.
    The prior implementation moved the weight to a "cash" column and let the
    NaN silently ffill to zero-return, which understated long-only drawdowns
    substantially in historical backtests.
    """

    def test_long_position_delisting_realizes_minus_100_percent(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=10)
        prices = pd.DataFrame(
            {
                "AAA": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                # BBB trades for 4 days, then goes to NaN (bankruptcy)
                "BBB": [100, 101, 102, 103, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            },
            index=dates,
        )
        # Long BBB continuously until delisting. Use daily rebalance so the
        # position is maintained until BBB drops out of the available universe.
        sig_idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "symbol"])
        signals = pd.DataFrame(0, index=sig_idx, columns=["signal"])
        for d in dates[:4]:
            signals.loc[(d, "BBB"), "signal"] = 1

        result = calculate_portfolio_returns(
            signals,
            prices,
            rebalance_freq="D",
            transaction_cost=0.0,
            long_only=True,
        )

        # Day 4 is the first day BBB is NaN. Position carried from day 3 (weight
        # 1.0) must realize -100% on day 4.
        day4 = dates[4]
        assert np.isclose(result.loc[day4, "gross_return"], -1.0, atol=1e-10), (
            f"Expected -100% on delisting day, got {result.loc[day4, 'gross_return']}"
        )
        # Subsequent days must not double-count (position zeroed out)
        for later in dates[5:]:
            assert np.isclose(result.loc[later, "gross_return"], 0.0, atol=1e-10)

    def test_pct_change_does_not_forward_fill_nans(self) -> None:
        """
        Regression: pct_change previously used the default fill_method='pad'
        which silently forward-filled NaN prices. This test asserts NaN prices
        produce NaN returns, which are then handled by the delisting path.
        """
        dates = pd.bdate_range("2023-01-02", periods=6)
        prices = pd.DataFrame(
            {"X": [100.0, 101.0, np.nan, np.nan, 103.0, 104.0]},
            index=dates,
        )
        returns = prices.pct_change(fill_method=None)
        # Day 2 and 3 must be NaN. (Not 0% as under ffill.)
        assert pd.isna(returns.iloc[2, 0])
        assert pd.isna(returns.iloc[3, 0])


class TestSignalLag:
    """
    Guards for Bug 3 in docs/FACTOR_BACKTEST_AUDIT.md.

    With default `signal_lag_days=1`, the signal emitted on date t must be
    derived from the factor value observed on date (t-1). With
    `signal_lag_days=0`, the factor value on t is used (same-close / MOC
    execution — retained only for legacy parity, NOT realistic).
    """

    def test_default_lag_shifts_factor_by_one_day(self) -> None:
        dates = pd.date_range("2023-01-03", periods=5, freq="D")
        symbols = ["A", "B", "C", "D"]
        # Factor = rank(symbol) increasing each day so shift is easy to detect.
        records = []
        for t, d in enumerate(dates):
            for i, s in enumerate(symbols):
                # On day t: A=t, B=t+1, C=t+2, D=t+3  (D is highest every day)
                records.append({"date": d, "symbol": s, "f": float(t + i)})
        factors = pd.DataFrame(records).set_index(["date", "symbol"])

        # With lag=1, date t uses factor at date (t-1). Day 0 has no prior
        # factor so all signals should be 0. Day 1 uses day 0's factor.
        sigs = create_signals_from_factor(
            factors, "f", top_pct=0.25, bottom_pct=0.25, min_stocks=2, signal_lag_days=1
        )
        day0_signals = sigs.loc[dates[0], "signal"]
        assert (day0_signals == 0).all(), (
            "signal_lag_days=1: day 0 must have no signals (no prior factor)"
        )

        # With lag=0, day 0 produces signals immediately.
        sigs0 = create_signals_from_factor(
            factors, "f", top_pct=0.25, bottom_pct=0.25, min_stocks=2, signal_lag_days=0
        )
        day0_signals_nolag = sigs0.loc[dates[0], "signal"]
        assert (day0_signals_nolag != 0).any(), (
            "signal_lag_days=0: day 0 should produce signals"
        )

    def test_lag_zero_reproduces_legacy_behavior(self) -> None:
        """signal_lag_days=0 should produce identical signals to the old
        factor-on-t approach (same-close MOC)."""
        dates = pd.date_range("2023-01-03", periods=3, freq="D")
        symbols = ["A", "B", "C"]
        records = []
        for t, d in enumerate(dates):
            for i, s in enumerate(symbols):
                records.append({"date": d, "symbol": s, "f": float(t + i)})
        factors = pd.DataFrame(records).set_index(["date", "symbol"])

        sigs = create_signals_from_factor(
            factors, "f", top_pct=0.34, bottom_pct=0.34, min_stocks=2, signal_lag_days=0
        )
        # On every day, C has highest factor -> long; A has lowest -> short.
        for d in dates:
            assert sigs.loc[(d, "C"), "signal"] == 1
            assert sigs.loc[(d, "A"), "signal"] == -1

    def test_negative_lag_rejected(self) -> None:
        dates = pd.date_range("2023-01-03", periods=2, freq="D")
        factors = pd.DataFrame(
            {"f": [1.0, 2.0]},
            index=pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"]),
        )
        with pytest.raises(ValueError, match="signal_lag_days"):
            create_signals_from_factor(factors, "f", signal_lag_days=-1, min_stocks=1)


class TestEndOfDayTzBoundary:
    """
    Guards for Bug 5 in docs/FACTOR_BACKTEST_AUDIT.md.

    When the factor index is tz-aware UTC, a tz-naive request date like
    '2024-06-28' localizes to 00:00 UTC — which is BEFORE the ~20:00 UTC
    US equity close. Without end-of-day normalization, the last trading bar
    is silently dropped from the backtest window.
    """

    def test_factor_runner_includes_last_bar_under_utc_index(self) -> None:
        from core.strategies.factor_runner import run_factor_cross_section_backtest

        # Build a tz-aware UTC factor and price panel spanning 3 days.
        dates = pd.to_datetime(
            ["2024-06-26 20:00", "2024-06-27 20:00", "2024-06-28 20:00"], utc=True
        )
        symbols = ["AAA", "BBB", "CCC"]
        price_df = pd.DataFrame(
            {"AAA": [100, 101, 102], "BBB": [50, 51, 52], "CCC": [200, 198, 199]},
            index=dates,
        )
        records = []
        for t, d in enumerate(dates):
            for i, s in enumerate(symbols):
                records.append({"date": d, "symbol": s, "f": float(t * 10 + i)})
        factors = pd.DataFrame(records).set_index(["date", "symbol"])

        # Pass a tz-naive end date at midnight; the runner must push to EOD
        # so the 20:00 UTC bar on 2024-06-28 is included.
        net = run_factor_cross_section_backtest(
            factors,
            price_df,
            factor_col="f",
            start=pd.Timestamp("2024-06-26"),
            end=pd.Timestamp("2024-06-28"),  # midnight => would exclude 20:00 bar
            top_pct=0.34,
            bottom_pct=0.34,
            rebalance_freq="D",
            transaction_cost=0.0,
            min_stocks=2,
            signal_lag_days=0,
        )
        # If the tz bug returned, `net` would miss the last day and be shorter.
        assert len(net) >= 2, (
            f"Got {len(net)} return rows from 3-day panel; "
            "last-bar drop (Bug 5) likely regressed."
        )
        assert dates[-1] in net.index, (
            "The 2024-06-28 20:00 UTC bar must be in the net return index."
        )
