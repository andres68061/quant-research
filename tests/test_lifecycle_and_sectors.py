"""Tests for sector FMP fetch helpers and symbol lifecycle truncation."""

from __future__ import annotations

import pandas as pd

from core.data.lifecycle import (
    _earliest_segment_end,
    apply_lifecycle_to_panel,
    build_lifecycle_windows,
)
from core.data.sector_classification import _quote_type_from_profile, _slugify


class TestSectorHelpers:
    def test_slugify(self) -> None:
        assert _slugify("Consumer Electronics") == "consumer-electronics"
        assert _slugify("Unknown") == "Unknown"

    def test_quote_type_flags(self) -> None:
        assert _quote_type_from_profile({"isEtf": True}, "SPY") == "ETF"
        assert _quote_type_from_profile({"isFund": True}, "VTSAX") == "MUTUALFUND"
        assert _quote_type_from_profile({}, "AAPL") == "EQUITY"
        assert _quote_type_from_profile({}, "^GSPC") == "INDEX"


class TestLifecycle:
    def test_gap_detection_keeps_earliest_segment(self) -> None:
        idx = pd.DatetimeIndex(
            list(pd.bdate_range("2010-01-01", periods=50))
            + list(pd.bdate_range("2022-01-01", periods=20)),
            tz="America/New_York",
        )
        series = pd.Series(100.0, index=idx)
        end = _earliest_segment_end(series, max_gap_days=252)
        assert end == idx[49]

    def test_apply_truncates_post_gap_junk(self) -> None:
        idx = pd.DatetimeIndex(
            list(pd.bdate_range("2010-01-01", periods=40))
            + list(pd.bdate_range("2022-01-01", periods=10)),
            tz="America/New_York",
        )
        prices = pd.DataFrame({"STI": 50.0}, index=idx)
        windows = build_lifecycle_windows(prices)
        assert "price_gap" in windows.loc[0, "source_notes"]
        truncated, n_cleared = apply_lifecycle_to_panel(prices, windows)
        assert n_cleared == 10
        assert truncated["STI"].dropna().index.max() == idx[39]
        assert truncated.loc[idx[40] :, "STI"].isna().all()

    def test_symbol_change_ends_window(self) -> None:
        # Series ends within 90 days of the rename — trust it.
        idx = pd.bdate_range("2018-01-01", periods=50, tz="America/New_York")
        prices = pd.DataFrame({"FB": 100.0}, index=idx)
        changes = pd.DataFrame(
            {
                "date": [idx[40]],
                "old_symbol": ["FB"],
                "new_symbol": ["META"],
                "company_name": ["Meta"],
            }
        )
        windows = build_lifecycle_windows(prices, symbol_changes=changes)
        assert windows.loc[0, "valid_to"] == idx[40]
        assert "symbol_change" in windows.loc[0, "source_notes"]

    def test_price_span_only_window_never_truncates_fresh_data(self) -> None:
        # Windows built when the panel ended earlier must not delete data
        # fetched after the build: a price_span-only valid_to is not evidence
        # of delisting, just the panel's extent on build day.
        old_idx = pd.bdate_range("2024-01-01", periods=100, tz="America/New_York")
        windows = build_lifecycle_windows(pd.DataFrame({"AAPL": 100.0}, index=old_idx))
        assert windows.loc[0, "source_notes"] == "price_span"
        new_idx = pd.bdate_range("2024-01-01", periods=110, tz="America/New_York")
        fresh_prices = pd.DataFrame({"AAPL": 100.0}, index=new_idx)
        truncated, n_cleared = apply_lifecycle_to_panel(fresh_prices, windows)
        assert n_cleared == 0
        assert truncated["AAPL"].notna().all()

    def test_symbol_change_ignored_if_series_continues(self) -> None:
        idx = pd.bdate_range("2018-01-01", periods=500, tz="America/New_York")
        prices = pd.DataFrame({"BK": 100.0}, index=idx)
        changes = pd.DataFrame(
            {
                "date": [idx[50]],
                "old_symbol": ["BK"],
                "new_symbol": ["OTHER"],
                "company_name": ["noise"],
            }
        )
        windows = build_lifecycle_windows(prices, symbol_changes=changes)
        assert "symbol_change" not in windows.loc[0, "source_notes"]
        assert windows.loc[0, "valid_to"] == idx[-1]
