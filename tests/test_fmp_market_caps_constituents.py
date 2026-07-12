"""Tests for FMP market-cap parsing and S&P membership reconstruction."""

from __future__ import annotations

import pandas as pd

from core.data.fmp.constituents import (
    build_membership_snapshots,
    normalize_equity_ticker,
    reconcile_membership,
)
from core.data.fmp.market_caps import parse_market_cap_rows


class TestParseMarketCapRows:
    def test_parses_and_sorts_ascending(self) -> None:
        rows = [
            {"symbol": "AAPL", "date": "2024-01-03", "marketCap": 3000.0},
            {"symbol": "AAPL", "date": "2024-01-02", "marketCap": 2900.0},
        ]
        frame = parse_market_cap_rows(rows)
        assert list(frame["market_cap"]) == [2900.0, 3000.0]
        assert frame.index.tz is not None
        assert frame.index.is_monotonic_increasing

    def test_empty_payload(self) -> None:
        frame = parse_market_cap_rows([])
        assert frame.empty
        assert list(frame.columns) == ["market_cap"]


class TestBuildMembershipSnapshots:
    def test_undoes_add_and_remove(self) -> None:
        # Today: {A, B, C}. On 2020-01-15 C was added and D was removed.
        # Before that event membership was {A, B, D}.
        events = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-15")],
                "added_symbol": ["C"],
                "removed_symbol": ["D"],
                "reason": ["test"],
            }
        )
        snapshots = build_membership_snapshots(events, current_members={"A", "B", "C"})
        assert set(snapshots.loc[pd.Timestamp("2020-01-15"), "tickers"]) == {"A", "B", "C"}
        pre = snapshots.index.min()
        assert set(snapshots.loc[pre, "tickers"]) == {"A", "B", "D"}

    def test_reconcile_identical_sources(self) -> None:
        tickers = [["A", "B"], ["A", "B", "C"]]
        dates = pd.to_datetime(["2019-01-01", "2020-01-15"])
        snaps = pd.DataFrame({"tickers": tickers}, index=dates)
        report = reconcile_membership(snaps, snaps)
        assert report.mean_jaccard == 1.0
        assert report.min_jaccard == 1.0

    def test_normalize_equity_ticker_share_class(self) -> None:
        assert normalize_equity_ticker("brk.b") == "BRK-B"
        assert normalize_equity_ticker("BF-B") == "BF-B"

    def test_reconcile_notation_difference(self) -> None:
        dates = pd.to_datetime(["2020-01-15"])
        fmp = pd.DataFrame({"tickers": [["AAPL", "BRK-B", "BF-B"]]}, index=dates)
        csv = pd.DataFrame({"tickers": [["AAPL", "BRK.B", "BF.B"]]}, index=dates)
        raw = reconcile_membership(fmp, csv, normalize_notation=False)
        norm = reconcile_membership(fmp, csv, normalize_notation=True)
        assert raw.mean_jaccard < 1.0
        assert norm.mean_jaccard == 1.0
        assert norm.notation_normalized is True
