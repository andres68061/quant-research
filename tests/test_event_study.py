"""Tests for the event-time study engine (PEAD design correctness)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.event_study import (
    assign_quarterly_quantiles,
    extract_events_from_surprise_panel,
    run_event_study,
)
from core.exceptions import DataSchemaError

TZ = "America/New_York"


def _flat_prices(n_symbols: int = 30, n_days: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-02", periods=n_days, tz=TZ, name="date")
    returns = rng.normal(0.0, 0.005, size=(n_days, n_symbols))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"S{i:02d}" for i in range(n_symbols)])


def _events(prices: pd.DataFrame, n_per_symbol: int = 2) -> pd.DataFrame:
    """One event per symbol per half-year, signal = symbol number (deterministic ranks)."""
    rows = []
    for i, symbol in enumerate(prices.columns):
        for k in range(n_per_symbol):
            rows.append(
                {
                    "symbol": symbol,
                    "event_date": prices.index[40 + 150 * k + (i % 5)],
                    "signal": float(i),
                }
            )
    return pd.DataFrame(rows)


class TestRunEventStudy:
    def test_recovers_an_injected_drift(self) -> None:
        """Stocks with the top signal get +10bps/day injected after their event;
        the study must show that drift in the top quantile and not the bottom."""
        prices = _flat_prices()
        events = _events(prices)
        top_symbols = {f"S{i:02d}" for i in range(24, 30)}  # top signal quintile

        boosted = prices.copy()
        for _, event in events.iterrows():
            if event["symbol"] in top_symbols:
                position = boosted.index.get_loc(event["event_date"])
                drift = np.ones(len(boosted))
                drift[position + 1 : position + 41] = 1.001
                boosted[event["symbol"]] = boosted[event["symbol"]] * np.cumprod(drift)

        result = run_event_study(events, boosted, horizon_days=40, n_quantiles=5)
        car = result["car_paths"]
        # ~40 days x 10bps = ~4% drift, minus the leak into the universe mean.
        assert car["Q5"].iloc[-1] > 0.02
        assert abs(car["Q1"].iloc[-1]) < 0.02
        assert result["spread_path"].iloc[-1] > 0.02
        assert result["spread_t_stat"] > 2

    def test_no_drift_means_no_spread(self) -> None:
        prices = _flat_prices(seed=11)
        result = run_event_study(_events(prices), prices, horizon_days=40, n_quantiles=5)
        assert abs(result["spread_path"].iloc[-1]) < 0.02
        assert abs(result["spread_t_stat"]) < 2.5

    def test_day_zero_jump_is_excluded(self) -> None:
        """A one-day announcement jump ON day 0 must not appear in the drift path."""
        prices = _flat_prices(seed=5)
        events = _events(prices).iloc[:30]
        jumped = prices.copy()
        for _, event in events.iterrows():
            position = jumped.index.get_loc(event["event_date"])
            jumped.iloc[position:, jumped.columns.get_loc(event["symbol"])] *= 1.10

        result = run_event_study(events, jumped, horizon_days=20, n_quantiles=3)
        # The 10% jump happens at day 0; the path starting at day +1 must not carry it.
        for quantile in result["car_paths"].columns:
            assert abs(result["car_paths"][quantile].iloc[0]) < 0.02

    def test_event_near_end_of_history_truncates_gracefully(self) -> None:
        prices = _flat_prices()
        events = pd.DataFrame(
            [
                {"symbol": "S00", "event_date": prices.index[-5], "signal": 1.0},
                {"symbol": "S01", "event_date": prices.index[-5], "signal": 2.0},
            ]
            * 6
        )
        result = run_event_study(events, prices, horizon_days=40, n_quantiles=2)
        assert result["car_paths"].shape[0] == 40  # full horizon frame, NaN-safe

    def test_malformed_events_raise(self) -> None:
        prices = _flat_prices()
        with pytest.raises(DataSchemaError):
            run_event_study(pd.DataFrame({"symbol": ["A"]}), prices)
        with pytest.raises(DataSchemaError):
            run_event_study(pd.DataFrame(columns=["symbol", "event_date", "signal"]), prices)


class TestQuarterlyQuantiles:
    def test_breakpoints_are_within_quarter(self) -> None:
        """A signal huge in Q1-2020 terms but average in full-sample terms must
        still rank at the top of its own quarter."""
        events = pd.DataFrame(
            {
                "event_date": (
                    list(pd.date_range("2020-01-15", periods=12, freq="D", tz=TZ))
                    + list(pd.date_range("2021-01-15", periods=12, freq="D", tz=TZ))
                ),
                "signal": list(range(12)) + [x + 100 for x in range(12)],
            }
        )
        quantiles = assign_quarterly_quantiles(events, n_quantiles=3)
        # Highest signal WITHIN each quarter gets the top bucket.
        assert quantiles.iloc[11] == 3
        assert quantiles.iloc[23] == 3
        # Lowest of the 2021 quarter is bucket 1 even though its raw signal (100)
        # dwarfs everything in 2020.
        assert quantiles.iloc[12] == 1

    def test_thin_quarters_are_dropped(self) -> None:
        events = pd.DataFrame(
            {
                "event_date": pd.date_range("2020-01-15", periods=4, freq="D", tz=TZ),
                "signal": [1.0, 2.0, 3.0, 4.0],
            }
        )
        assert assign_quarterly_quantiles(events, n_quantiles=2).isna().all()


class TestExtractEvents:
    def test_counter_reset_marks_new_announcement(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.bdate_range("2024-01-02", periods=6, tz=TZ, name="date"), ["AAA"]],
            names=["date", "symbol"],
        )
        panel = pd.DataFrame(
            {
                "sue_price_scaled": [0.01] * 3 + [0.02] * 3,
                "days_since_earnings": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            },
            index=index,
        )
        events = extract_events_from_surprise_panel(panel)
        assert len(events) == 2
        assert list(events["signal"]) == [0.01, 0.02]

    def test_missing_counter_raises(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.bdate_range("2024-01-02", periods=2, tz=TZ, name="date"), ["AAA"]],
            names=["date", "symbol"],
        )
        panel = pd.DataFrame({"sue_price_scaled": [0.01, 0.01]}, index=index)
        with pytest.raises(DataSchemaError):
            extract_events_from_surprise_panel(panel)
