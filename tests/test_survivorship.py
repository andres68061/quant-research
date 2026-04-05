"""Tests for survivorship-bias-free universe filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.portfolio import create_signals_from_factor, sp500_universe_filter


@pytest.fixture()
def toy_factor_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=3, tz="America/New_York")
    rows = []
    for d in dates:
        for sym in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
            rows.append({"date": d, "symbol": sym, "mom_12_1": np.random.default_rng(0).random()})
    df = pd.DataFrame(rows).set_index(["date", "symbol"])
    df["mom_12_1"] = np.arange(len(df), dtype=float) * 0.01
    return df


def _mock_universe_filter(eligible: set[str]):
    """Return a callable that always returns the same set of symbols."""

    def _filter(date: pd.Timestamp) -> set[str]:
        return eligible

    return _filter


class TestCreateSignalsWithFilter:
    def test_filter_excludes_symbols(self, toy_factor_panel: pd.DataFrame) -> None:
        eligible = {"AAA", "BBB", "CCC"}
        signals = create_signals_from_factor(
            toy_factor_panel,
            "mom_12_1",
            top_pct=0.5,
            bottom_pct=0.0,
            long_only=True,
            min_stocks=2,
            universe_filter=_mock_universe_filter(eligible),
        )
        nonzero = signals[signals["signal"] != 0]
        symbols_with_signal = nonzero.index.get_level_values("symbol").unique()
        assert all(s in eligible for s in symbols_with_signal)
        assert "DDD" not in symbols_with_signal
        assert "EEE" not in symbols_with_signal

    def test_no_filter_includes_all(self, toy_factor_panel: pd.DataFrame) -> None:
        signals = create_signals_from_factor(
            toy_factor_panel,
            "mom_12_1",
            top_pct=0.5,
            bottom_pct=0.0,
            long_only=True,
            min_stocks=2,
            universe_filter=None,
        )
        nonzero = signals[signals["signal"] != 0]
        symbols_with_signal = nonzero.index.get_level_values("symbol").unique()
        assert len(symbols_with_signal) > 0

    def test_empty_filter_yields_no_signals(self, toy_factor_panel: pd.DataFrame) -> None:
        signals = create_signals_from_factor(
            toy_factor_panel,
            "mom_12_1",
            top_pct=0.5,
            long_only=True,
            min_stocks=2,
            universe_filter=_mock_universe_filter(set()),
        )
        assert (signals["signal"] == 0).all()


class TestSP500UniverseFilter:
    def test_loads_and_returns_callable(self) -> None:
        uf = sp500_universe_filter()
        assert callable(uf)

    def test_returns_set_of_strings(self) -> None:
        uf = sp500_universe_filter()
        members = uf(pd.Timestamp("2020-06-15"))
        assert isinstance(members, set)
        assert all(isinstance(s, str) for s in members)
        assert len(members) > 400

    def test_different_dates_may_differ(self) -> None:
        uf = sp500_universe_filter()
        early = uf(pd.Timestamp("2000-01-03"))
        late = uf(pd.Timestamp("2025-01-02"))
        assert early != late

    def test_pre_csv_date_returns_empty(self) -> None:
        uf = sp500_universe_filter()
        members = uf(pd.Timestamp("1990-01-02"))
        assert isinstance(members, set)
        assert len(members) == 0
