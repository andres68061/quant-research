"""Tests for core.backtest.events (v0 event log and simulator)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from core.backtest.events import (
    Event,
    EventLog,
    EventType,
    simulate_equal_weight_rebalances,
    validate_and_sort_events,
)
from core.backtest.portfolio import calculate_portfolio_returns, create_signals_from_factor
from core.exceptions import DataSchemaError


def _utc_ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


class TestEventLogValidation:
    def test_rejects_naive_timestamp(self) -> None:
        ev = Event(
            ts=pd.Timestamp("2024-01-02"),
            event_type=EventType.REBALANCE,
            symbol=None,
            payload={"symbols": ["A"]},
        )
        with pytest.raises(DataSchemaError, match="timezone-aware"):
            EventLog([ev])

    def test_rejects_duplicate_timestamp(self) -> None:
        t = _utc_ts("2024-01-02")
        a = Event(
            ts=t,
            event_type=EventType.REBALANCE,
            symbol=None,
            payload={"symbols": ["A"]},
        )
        b = Event(
            ts=t,
            event_type=EventType.REBALANCE,
            symbol=None,
            payload={"symbols": ["B"]},
        )
        with pytest.raises(DataSchemaError, match="strictly increasing"):
            EventLog([a, b])

    def test_sorts_unsorted_input(self) -> None:
        a = Event(
            ts=_utc_ts("2024-01-02"),
            event_type=EventType.REBALANCE,
            symbol=None,
            payload={"symbols": ["A"]},
        )
        b = Event(
            ts=_utc_ts("2024-01-05"),
            event_type=EventType.REBALANCE,
            symbol=None,
            payload={"symbols": ["A", "B"]},
        )
        log = EventLog([b, a])
        assert [e.ts for e in log.events] == [a.ts, b.ts]


class TestSimulator:
    @pytest.fixture()
    def price_panel(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2024-01-02", periods=20, tz="UTC")
        a = 100 * np.exp(rng.normal(0, 0.01, len(idx)).cumsum())
        b = 100 * np.exp(rng.normal(0, 0.01, len(idx)).cumsum())
        return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)

    def test_matches_manual_equal_weight(
        self, price_panel: pd.DataFrame
    ) -> None:
        """Portfolio return equals manual 0.5 * r_a + 0.5 * r_b after rebalance."""
        ev = Event(
            ts=_utc_ts("2024-01-01"),
            event_type=EventType.REBALANCE,
            symbol=None,
            payload={"symbols": ["AAA", "BBB"]},
        )
        sim = simulate_equal_weight_rebalances(price_panel, [ev])
        r = price_panel.pct_change().fillna(0.0)
        manual = 0.5 * r["AAA"] + 0.5 * r["BBB"]
        # First day after event: searchsorted right from 2024-01-01 -> first bdate 2024-01-02
        pd.testing.assert_series_equal(
            sim.fillna(0.0),
            manual.fillna(0.0),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_no_future_warning_legacy_freq_portfolio(
        self, price_panel: pd.DataFrame
    ) -> None:
        """Cross-check: factor pipeline with ME avoids resample FutureWarning."""
        # Build trivial long-only factor panel MultiIndex
        records = []
        for dt in price_panel.index:
            for sym in price_panel.columns:
                records.append({"date": dt, "symbol": sym, "f": 1.0})
        factors = (
            pd.DataFrame(records).set_index(["date", "symbol"]).sort_index()
        )
        signals = create_signals_from_factor(
            factors, "f", long_only=True, min_stocks=1
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculate_portfolio_returns(
                signals,
                price_panel,
                rebalance_freq="ME",
                transaction_cost=0.0,
                long_only=True,
            )
        future = [x for x in w if issubclass(x.category, FutureWarning)]
        assert not future


class TestValidateAndSort:
    def test_strictly_increasing_preserved(self) -> None:
        t0 = _utc_ts("2020-06-01")
        t1 = _utc_ts("2020-06-15")
        events = [
            Event(
                ts=t0,
                event_type=EventType.REBALANCE,
                symbol=None,
                payload={"symbols": ["X"]},
            ),
            Event(
                ts=t1,
                event_type=EventType.REBALANCE,
                symbol=None,
                payload={"symbols": ["X", "Y"]},
            ),
        ]
        out = validate_and_sort_events(list(reversed(events)))
        assert [e.ts for e in out] == [t0, t1]
