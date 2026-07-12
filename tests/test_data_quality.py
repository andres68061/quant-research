"""Tests for core.data.quality quarantine scanning (real small DataFrames, no I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.quality import (
    merge_with_existing,
    repair_isolated_bad_prints,
    scan_entity_mismatch,
    scan_extreme_returns,
    scan_price_panel,
    scan_spike_reversals,
    scan_stale_prices,
)


@pytest.fixture
def business_days() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=400, tz="America/New_York")


def _random_walk(index: pd.DatetimeIndex, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))), index=index)


class TestScanSpikeReversals:
    def test_bad_print_pattern_is_caught(self, business_days: pd.DatetimeIndex) -> None:
        clean = _random_walk(business_days, seed=1)
        corrupted = clean.copy()
        corrupted.iloc[200] = corrupted.iloc[199] * 100  # spike up 100x...
        corrupted.iloc[201] = corrupted.iloc[199]  # ...and right back down
        panel = pd.DataFrame({"CLEAN": clean, "BAD": corrupted})
        findings = scan_spike_reversals(panel)
        assert list(findings["symbol"]) == ["BAD"]

    def test_real_crash_is_not_flagged(self, business_days: pd.DatetimeIndex) -> None:
        crashed = _random_walk(business_days, seed=2)
        crashed.iloc[300:] = crashed.iloc[300:] * 0.02  # falls 98% and STAYS down
        findings = scan_spike_reversals(pd.DataFrame({"CRASH": crashed}))
        assert findings.empty


class TestScanExtremeReturns:
    def test_repeat_offender_flagged_single_event_not(
        self, business_days: pd.DatetimeIndex
    ) -> None:
        one_event = _random_walk(business_days, seed=3)
        one_event.iloc[100] *= 3.0  # single legitimate-looking jump

        repeat = _random_walk(business_days, seed=4)
        for i in (50, 150, 250):
            repeat.iloc[i] *= 5.0

        findings = scan_extreme_returns(pd.DataFrame({"ONE": one_event, "MANY": repeat}))
        assert list(findings["symbol"]) == ["MANY"]


class TestScanStalePrices:
    def test_frozen_feed_flagged(self, business_days: pd.DatetimeIndex) -> None:
        frozen = _random_walk(business_days, seed=5)
        frozen.iloc[100:130] = frozen.iloc[100]  # 30 identical closes
        findings = scan_stale_prices(pd.DataFrame({"FROZEN": frozen}))
        assert list(findings["symbol"]) == ["FROZEN"]
        assert findings["value"].iloc[0] >= 30


class TestScanEntityMismatch:
    def test_different_company_same_ticker(self, business_days: pd.DatetimeIndex) -> None:
        company_a = _random_walk(business_days, seed=6)
        company_b = _random_walk(business_days, seed=7)  # unrelated series
        panel = pd.DataFrame({"XYZ": company_a, "SAME": company_a * 1.5})
        reference = pd.DataFrame({"XYZ": company_b, "SAME": company_a})
        findings = scan_entity_mismatch(panel, reference)
        assert list(findings["symbol"]) == ["XYZ"]

    def test_short_overlap_not_judged(self) -> None:
        short_index = pd.bdate_range("2024-01-01", periods=50, tz="America/New_York")
        a = _random_walk(short_index, seed=8)
        b = _random_walk(short_index, seed=9)
        findings = scan_entity_mismatch(pd.DataFrame({"X": a}), pd.DataFrame({"X": b}))
        assert findings.empty


class TestScanPricePanel:
    def test_statuses_and_index_symbols_skipped(self, business_days: pd.DatetimeIndex) -> None:
        corrupted = _random_walk(business_days, seed=10)
        corrupted.iloc[200] = corrupted.iloc[199] * 100
        corrupted.iloc[201] = corrupted.iloc[199]
        index_series = corrupted.copy()  # same corruption, but it's a benchmark symbol
        panel = pd.DataFrame({"BAD": corrupted, "^GSPC": index_series})
        findings = scan_price_panel(panel)
        assert "^GSPC" not in set(findings["symbol"])
        bad_rows = findings[findings["symbol"] == "BAD"]
        assert (
            bad_rows.loc[bad_rows["check"] == "spike_reversal", "status"] == "quarantined"
        ).all()

    def test_clean_panel_returns_empty_with_schema(self, business_days: pd.DatetimeIndex) -> None:
        panel = pd.DataFrame({"A": _random_walk(business_days, seed=11)})
        findings = scan_price_panel(panel)
        assert findings.empty
        assert "status" in findings.columns


class TestRepairIsolatedBadPrints:
    def test_single_day_print_removed_and_logged(self, business_days: pd.DatetimeIndex) -> None:
        clean = _random_walk(business_days, seed=20)
        corrupted = clean.copy()
        corrupted.iloc[100] = corrupted.iloc[99] * 2.1  # LEN-style doubled quote for one day
        panel = pd.DataFrame({"BAD": corrupted, "OK": clean})
        repaired, log = repair_isolated_bad_prints(panel)
        assert pd.isna(repaired["BAD"].iloc[100])
        assert repaired["OK"].equals(clean)
        assert len(log) == 1
        assert log.iloc[0]["symbol"] == "BAD"

    def test_real_crash_untouched(self, business_days: pd.DatetimeIndex) -> None:
        crashed = _random_walk(business_days, seed=21)
        crashed.iloc[200:] = crashed.iloc[200:] * 0.05  # permanent 95% loss
        repaired, log = repair_isolated_bad_prints(pd.DataFrame({"CRASH": crashed}))
        assert log.empty
        assert repaired["CRASH"].equals(crashed)

    def test_no_false_positive_on_clean_panel(self, business_days: pd.DatetimeIndex) -> None:
        panel = pd.DataFrame({"A": _random_walk(business_days, seed=22)})
        repaired, log = repair_isolated_bad_prints(panel)
        assert log.empty
        assert repaired.equals(panel)


class TestMergeWithExisting:
    def test_manual_cleared_decision_survives_rescan(self) -> None:
        fresh = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "check": ["spike_reversal", "spike_reversal"],
                "value": [1.0, 2.0],
                "detail": ["x", "y"],
                "status": ["quarantined", "quarantined"],
                "scanned_at": ["t1", "t1"],
            }
        )
        existing = fresh.copy()
        existing.loc[existing["symbol"] == "AAA", "status"] = "cleared"
        merged = merge_with_existing(fresh, existing)
        assert merged.set_index("symbol")["status"].to_dict() == {
            "AAA": "cleared",
            "BBB": "quarantined",
        }

    def test_manual_escalation_with_note_survives_rescan(self) -> None:
        fresh = pd.DataFrame(
            {
                "symbol": ["CCC"],
                "check": ["extreme_returns"],
                "value": [4.0],
                "detail": ["4 extreme days"],
                "status": ["flagged"],  # scanner default for soft checks
                "scanned_at": ["t2"],
            }
        )
        existing = fresh.copy()
        existing["status"] = "quarantined"
        existing["review_note"] = "confirmed bad prints by hand"
        merged = merge_with_existing(fresh, existing)
        assert merged.loc[0, "status"] == "quarantined"
        assert merged.loc[0, "review_note"] == "confirmed bad prints by hand"
