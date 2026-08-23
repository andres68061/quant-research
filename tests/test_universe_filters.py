"""Tests for universe eligibility filters (shells/SPACs, index membership)."""

from __future__ import annotations

import pandas as pd
import pytest

from core.data.universe_filters import (
    NON_OPERATING_INDUSTRIES,
    build_universe_filter,
    load_membership_filter,
    load_non_operating_symbols,
)


def _sectors_file(tmp_path) -> "pd.Path":  # type: ignore[name-defined]
    frame = pd.DataFrame(
        {
            "symbol": ["REAL", "SPAC1", "SPAC2", "FUND", "ALSOREAL"],
            "sector": [
                "Technology",
                "Financial Services",
                "Financial Services",
                "Financial Services",
                "Healthcare",
            ],
            "industry": [
                "Software",
                "Shell Companies",
                "Shell Companies",
                "Asset Management - Bonds",
                "Biotechnology",
            ],
        }
    )
    path = tmp_path / "sectors.parquet"
    frame.to_parquet(path, index=False)
    return path


def _membership_file(tmp_path) -> "pd.Path":  # type: ignore[name-defined]
    frame = pd.DataFrame(
        {
            "symbol": ["REAL", "REAL", "ALSOREAL", "SPAC1"],
            "index_name": ["sp500"] * 4,
            "valid_from": pd.to_datetime(["2000-01-01", "2015-01-01", "2010-01-01", "2021-01-01"]),
            "valid_to": pd.to_datetime(["2005-12-31", None, None, "2022-12-31"]),
        }
    )
    path = tmp_path / "membership.parquet"
    frame.to_parquet(path, index=False)
    return path


class TestNonOperatingSymbols:
    def test_shells_and_fund_vehicles_excluded(self, tmp_path) -> None:
        excluded = load_non_operating_symbols(_sectors_file(tmp_path))
        assert excluded == {"SPAC1", "SPAC2", "FUND"}

    def test_operating_companies_kept(self, tmp_path) -> None:
        excluded = load_non_operating_symbols(_sectors_file(tmp_path))
        assert "REAL" not in excluded and "ALSOREAL" not in excluded

    def test_missing_file_returns_empty_not_crash(self, tmp_path) -> None:
        assert load_non_operating_symbols(tmp_path / "nope.parquet") == set()

    def test_shell_companies_is_in_the_default_list(self) -> None:
        assert "Shell Companies" in NON_OPERATING_INDUSTRIES


class TestMembershipFilter:
    def test_member_only_inside_its_interval(self, tmp_path) -> None:
        members_on = load_membership_filter("sp500", _membership_file(tmp_path))
        assert "REAL" in members_on(pd.Timestamp("2003-06-01"))
        # Gap between REAL's two intervals.
        assert "REAL" not in members_on(pd.Timestamp("2010-06-01"))
        assert "REAL" in members_on(pd.Timestamp("2020-06-01"))

    def test_open_interval_still_a_member_today(self, tmp_path) -> None:
        members_on = load_membership_filter("sp500", _membership_file(tmp_path))
        assert "ALSOREAL" in members_on(pd.Timestamp("2026-01-01"))

    def test_closed_interval_not_a_member_after(self, tmp_path) -> None:
        members_on = load_membership_filter("sp500", _membership_file(tmp_path))
        assert "SPAC1" in members_on(pd.Timestamp("2021-06-01"))
        assert "SPAC1" not in members_on(pd.Timestamp("2024-01-01"))

    def test_tz_aware_dates_accepted(self, tmp_path) -> None:
        members_on = load_membership_filter("sp500", _membership_file(tmp_path))
        assert "REAL" in members_on(pd.Timestamp("2020-06-01", tz="America/New_York"))

    def test_missing_table_raises(self, tmp_path) -> None:
        """Returning 'everyone is a member' would be a silent survivorship bug."""
        with pytest.raises(FileNotFoundError):
            load_membership_filter("sp500", tmp_path / "nope.parquet")


class TestComposedFilter:
    def test_membership_and_exclusion_compose(self, tmp_path) -> None:
        panel = pd.Index(["REAL", "ALSOREAL", "SPAC1", "SPAC2", "FUND"])
        eligible = build_universe_filter(
            panel,
            exclude_non_operating=True,
            index_name="sp500",
            sectors_path=_sectors_file(tmp_path),
            membership_path=_membership_file(tmp_path),
        )
        # SPAC1 is an index member on this date but is a shell -> excluded anyway.
        assert eligible(pd.Timestamp("2021-06-01")) == {"REAL", "ALSOREAL"}

    def test_whole_universe_mode_returns_panel_minus_shells(self, tmp_path) -> None:
        panel = pd.Index(["REAL", "ALSOREAL", "SPAC1", "SPAC2", "FUND"])
        eligible = build_universe_filter(
            panel,
            exclude_non_operating=True,
            index_name=None,
            sectors_path=_sectors_file(tmp_path),
        )
        assert eligible(pd.Timestamp("2021-06-01")) == {"REAL", "ALSOREAL"}

    def test_symbols_outside_the_panel_are_never_eligible(self, tmp_path) -> None:
        panel = pd.Index(["REAL"])
        eligible = build_universe_filter(
            panel,
            exclude_non_operating=True,
            index_name="sp500",
            sectors_path=_sectors_file(tmp_path),
            membership_path=_membership_file(tmp_path),
        )
        assert eligible(pd.Timestamp("2020-06-01")) == {"REAL"}

    def test_no_filter_requested_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            build_universe_filter(pd.Index(["A"]), exclude_non_operating=False, index_name=None)
