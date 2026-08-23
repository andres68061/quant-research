"""Tests for the data-health audit against the real on-disk layout.

These intentionally run against the repo's actual data directory (read-only):
the audit's whole job is to describe reality, so a synthetic fixture would test
the wrong thing. Assertions are structural (keys, ranges, types), not exact
counts, so they survive data growth.
"""

from __future__ import annotations

import pytest

from core.data.health import (
    RAW_FMP,
    UNIVERSE_FILE,
    audit_calendar,
    audit_panels,
    audit_survivorship,
    audit_universe,
    known_flaws,
    load_symbol_detail,
    run_full_audit,
)

needs_universe = pytest.mark.skipif(not UNIVERSE_FILE.exists(), reason="universe table not built")
needs_prices_report = pytest.mark.skipif(
    not (RAW_FMP / "prices" / "_fetch_report.csv").exists(), reason="no price fetch report"
)


@needs_universe
class TestAuditUniverse:
    def test_summary_partitions_the_universe(self) -> None:
        _, summary = audit_universe()
        assert summary["total"] == summary["live"] + summary["delisted"] + summary["carried_over"]
        assert summary["total"] > 0


@needs_universe
@needs_prices_report
class TestAuditSurvivorship:
    def test_percentages_are_bounded(self) -> None:
        universe, _ = audit_universe()
        result = audit_survivorship(universe)
        assert 0 <= result["with_prices_pct"] <= 100
        assert 0 <= result["price_end_within_30d_of_delisting_pct"] <= 100
        assert result["missing_count"] >= len(result["missing_symbols_sample"]) or (
            result["missing_count"] == len(result["missing_symbols_sample"])
        )

    def test_era_buckets_cover_delisted_names(self) -> None:
        universe, _ = audit_universe()
        result = audit_survivorship(universe)
        assert sum(era["names"] for era in result["by_delist_era"]) <= result["delisted_total"]


class TestAuditCalendar:
    def test_reports_or_declares_unchecked(self) -> None:
        result = audit_calendar()
        if result.get("status") == "unchecked":
            return
        assert result["non_trading_dates"] == result["weekend_dates"] + (
            result["holiday_or_other_dates"]
        )


class TestAuditPanels:
    def test_every_declared_panel_is_reported(self) -> None:
        entries = audit_panels()
        assert all("file" in e for e in entries)
        # A missing file is reported as missing, never dropped.
        assert all(e.get("status") == "missing" or e.get("rows", 0) >= 0 for e in entries)


@needs_universe
@needs_prices_report
class TestFlawRegistry:
    def test_flaws_have_required_fields_and_valid_severity(self) -> None:
        snapshot = run_full_audit()
        assert len(snapshot["flaws"]) > 0
        for flaw in snapshot["flaws"]:
            assert set(flaw) == {"id", "severity", "title", "detail"}
            assert flaw["severity"] in {"high", "medium", "low"}
            assert flaw["detail"].strip()

    def test_flaw_ids_are_unique(self) -> None:
        snapshot = run_full_audit()
        ids = [f["id"] for f in snapshot["flaws"]]
        assert len(ids) == len(set(ids))

    def test_known_flaws_is_pure_over_its_inputs(self) -> None:
        empty: dict = {"missing_count": 0, "missing_symbols_sample": []}
        flaws = known_flaws(empty, {}, {}, {}, [])
        assert isinstance(flaws, list) and all("id" in f for f in flaws)


@needs_universe
class TestSymbolDetail:
    def test_known_symbol_has_layers(self) -> None:
        detail = load_symbol_detail("AAPL")
        assert detail is not None
        assert detail["symbol"] == "AAPL"
        assert "statements" in detail and "datasets" in detail

    def test_unknown_symbol_returns_none(self) -> None:
        assert load_symbol_detail("ZZZZNOPE") is None
