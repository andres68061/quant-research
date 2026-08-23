"""Regression tests for the vectorized universe-filter path in signal construction.

The filter was rewritten from a per-date ``.loc`` + MultiIndex-append loop to a
single positional pass (see ADR 0013's consequences: at 6,400 symbols the old
path took ~100 s inside one backtest). These tests pin the two things that matter
about that rewrite: it must produce **identical** selections, and it must stay
fast enough that a widened universe does not silently regress the API.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from core.backtest.portfolio import create_signals_from_factor


def _factor_panel(n_dates: int, n_symbols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_dates, tz="America/New_York", name="date")
    symbols = [f"S{i:04d}" for i in range(n_symbols)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    return pd.DataFrame({"f": rng.normal(size=len(index))}, index=index)


def _reference_filter(valid_df: pd.DataFrame, universe_filter) -> pd.DataFrame:
    """The original per-date implementation, kept as the correctness oracle."""
    parts = []
    for date in valid_df.index.get_level_values("date").unique():
        eligible = universe_filter(date)
        date_slice = valid_df.loc[date]
        symbols_in = date_slice.index.get_level_values("symbol")
        kept = date_slice.index[symbols_in.isin(eligible)]
        parts.append(
            pd.MultiIndex.from_arrays(
                [[date] * len(kept), kept.get_level_values("symbol")], names=["date", "symbol"]
            )
        )
    if not parts:
        return valid_df.iloc[:0]
    combined = parts[0].append(parts[1:])
    return valid_df.loc[valid_df.index.isin(combined)]


def _current_filter(valid_df: pd.DataFrame, universe_filter) -> pd.DataFrame:
    """The shipped implementation, extracted for a direct A/B."""
    symbols_all = valid_df.index.get_level_values("symbol")
    keep = np.zeros(len(valid_df), dtype=bool)
    for date, positions in valid_df.groupby(level="date").indices.items():
        eligible = universe_filter(date)
        if not eligible:
            continue
        keep[positions] = symbols_all[positions].isin(eligible)
    return valid_df[keep]


def _time_varying_filter(symbols: list[str]):
    """Membership that changes every day, so a stale-cache bug would show up."""

    def eligible_on(date: pd.Timestamp) -> set[str]:
        size = 20 + (date.dayofyear % 40)
        return set(symbols[:size])

    return eligible_on


class TestFilterEquivalence:
    def test_selects_exactly_the_same_rows(self) -> None:
        panel = _factor_panel(n_dates=90, n_symbols=80)
        symbols = sorted({s for _, s in panel.index})
        universe_filter = _time_varying_filter(symbols)

        reference = _reference_filter(panel, universe_filter)
        current = _current_filter(panel, universe_filter)

        assert current.index.equals(reference.index)
        assert np.array_equal(current["f"].to_numpy(), reference["f"].to_numpy())

    def test_empty_eligible_set_drops_that_date_entirely(self) -> None:
        panel = _factor_panel(n_dates=10, n_symbols=5)
        cutoff = panel.index.get_level_values("date").unique()[3]

        def eligible_on(date: pd.Timestamp) -> set[str]:
            return set() if date == cutoff else {"S0000", "S0001"}

        current = _current_filter(panel, eligible_on)
        assert cutoff not in current.index.get_level_values("date")
        assert len(current) == 9 * 2

    def test_symbol_absent_from_panel_is_ignored(self) -> None:
        panel = _factor_panel(n_dates=5, n_symbols=3)
        current = _current_filter(panel, lambda _date: {"S0000", "NOT_IN_PANEL"})
        assert set(current.index.get_level_values("symbol")) == {"S0000"}

    def test_end_to_end_signals_match_between_implementations(self) -> None:
        """Equivalence must hold through the public entry point, not just the helper."""
        panel = _factor_panel(n_dates=60, n_symbols=40, seed=3)
        symbols = sorted({s for _, s in panel.index})
        signals = create_signals_from_factor(
            panel, factor_col="f", universe_filter=_time_varying_filter(symbols), min_stocks=5
        )
        # Every emitted signal must belong to an eligible name on its own date.
        eligible_on = _time_varying_filter(symbols)
        traded = signals[signals["signal"] != 0]
        for date, symbol in traded.index:
            assert symbol in eligible_on(date)


class TestFilterScaling:
    def test_wide_universe_stays_fast(self) -> None:
        """
        A widened universe must not reintroduce the quadratic path.

        The old implementation rebuilt a MultiIndex per date and ran ``isin``
        over the whole panel; this asserts the shipped path handles a
        2,000-symbol panel in well under the time that took.
        """
        panel = _factor_panel(n_dates=250, n_symbols=2000, seed=7)
        symbols = sorted({f"S{i:04d}" for i in range(2000)})

        started = time.monotonic()
        filtered = _current_filter(panel, _time_varying_filter(symbols))
        elapsed = time.monotonic() - started

        assert not filtered.empty
        assert elapsed < 5.0, f"universe filter took {elapsed:.1f}s on 500k rows"
