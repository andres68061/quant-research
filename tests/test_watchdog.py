"""Tests for the unattended data watchdog.

The checks are file-system facts, so each test builds a small fake data tree and
points the module at it. What matters is the verdict logic: a silent cron job and
a collapsed panel must both come back as errors, and a clean tree must not
produce noise that trains the user to ignore the banner.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.data.quality import watchdog as wd

TZ = "America/New_York"
NOW = datetime(2026, 8, 15, 18, 0, tzinfo=timezone.utc)


def write_log(path: Path, text: str, when: datetime = NOW) -> None:
    """Write a job log and pin its mtime.

    Job staleness is measured against the file's mtime, so a log left at the
    real wall-clock time would make these tests pass or fail depending on the
    date they are run on.
    """
    path.write_text(text)
    stamp = when.timestamp()
    os.utime(path, (stamp, stamp))


@pytest.fixture
def fake_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal data tree with every critical panel present and fresh."""
    factors = tmp_path / "factors"
    quality = tmp_path / "quality"
    logs = tmp_path / "logs"
    for directory in (factors, quality, logs):
        directory.mkdir(parents=True)

    index = pd.bdate_range(end="2026-08-14", periods=50, tz=TZ, name="date")
    prices = pd.DataFrame(
        {f"S{i:02d}": range(100, 150) for i in range(10)},
        index=index,
        dtype="float64",
    )
    prices.to_parquet(factors / "prices.parquet")
    for name in wd.CRITICAL_PANELS:
        if name != "prices.parquet":
            pd.DataFrame({"value": range(1000)}).to_parquet(factors / name)

    for log_name, _, _ in wd.SCHEDULED_JOBS + wd.FMP_SCHEDULED_JOBS:
        write_log(logs / log_name, "finished cleanly\n")

    # Every catalogued commodity and FRED series, one fresh observation each,
    # so series_freshness has something real to judge without the repo's data.
    from core.data.factors.macro_catalog import FRED_SERIES_CATALOG
    from core.data.vendors.commodities import COMMODITIES_CONFIG

    stamp = pd.Timestamp(NOW.date())
    commodities = tmp_path / "commodities" / "prices.parquet"
    commodities.parent.mkdir()
    pd.DataFrame({sym: [1.0] for sym in COMMODITIES_CONFIG}, index=[stamp]).to_parquet(commodities)
    raw_macro = tmp_path / "raw" / "macro_fred.parquet"
    raw_macro.parent.mkdir()
    pd.DataFrame(
        {
            "reference_date": [stamp] * len(FRED_SERIES_CATALOG),
            "series_id": list(FRED_SERIES_CATALOG),
            "value": 1.0,
        }
    ).to_parquet(raw_macro)
    monkeypatch.setattr(wd, "COMMODITIES_PANEL", commodities)
    monkeypatch.setattr(wd, "RAW_MACRO_PANEL", raw_macro)
    # Independent of the developer's .env: vendor live, no snapshot.
    monkeypatch.setattr(wd, "FMP_ENABLED", True)
    monkeypatch.setattr(wd, "FMP_SNAPSHOT_AS_OF", None)
    monkeypatch.setattr(wd, "FACTORS_DIR", factors)
    monkeypatch.setattr(wd, "QUALITY_DIR", quality)
    monkeypatch.setattr(wd, "LOGS_DIR", logs)
    monkeypatch.setattr(wd, "STATUS_FILE", quality / "watchdog_status.json")
    monkeypatch.setattr(wd, "BASELINE_FILE", quality / "artifact_baseline.json")
    return tmp_path


class TestCleanTree:
    def test_healthy_tree_reports_ok(self, fake_tree: Path) -> None:
        snapshot = wd.run_watchdog(now=NOW)
        assert snapshot["status"] == "ok"
        assert snapshot["n_errors"] == 0
        assert snapshot["summary"] == "All data checks passed."

    def test_no_check_is_silently_skipped(self, fake_tree: Path) -> None:
        snapshot = wd.run_watchdog(now=NOW)
        names = {c["name"] for c in snapshot["checks"]}
        assert "panel_freshness" in names
        for panel in wd.CRITICAL_PANELS:
            assert f"panel_present:{panel}" in names
        for log_name, _, _ in wd.SCHEDULED_JOBS + wd.FMP_SCHEDULED_JOBS:
            assert f"job:{log_name}" in names


class TestPanelPresence:
    def test_missing_panel_is_an_error(self, fake_tree: Path) -> None:
        (wd.FACTORS_DIR / "factors_all.parquet").unlink()
        snapshot = wd.run_watchdog(now=NOW)
        assert snapshot["status"] == "error"
        assert any(
            c["name"] == "panel_present:factors_all.parquet" and c["status"] == "error"
            for c in snapshot["checks"]
        )

    def test_unreadable_panel_is_an_error(self, fake_tree: Path) -> None:
        """A truncated write leaves a file that exists but has no parquet footer."""
        (wd.FACTORS_DIR / "factors_all.parquet").write_bytes(b"not parquet")
        snapshot = wd.run_watchdog(now=NOW)
        assert snapshot["status"] == "error"


class TestCollapseDetection:
    def test_collapsed_panel_is_an_error(self, fake_tree: Path) -> None:
        """The between-runs counterpart to the write guard."""
        wd.write_baseline(wd.current_row_counts())
        pd.DataFrame({"value": range(3)}).to_parquet(wd.FACTORS_DIR / "factors_all.parquet")

        snapshot = wd.run_watchdog(now=NOW)
        assert snapshot["status"] == "error"
        collapse = next(
            c for c in snapshot["checks"] if c["name"] == "panel_size:factors_all.parquet"
        )
        assert collapse["status"] == "error"
        assert collapse["facts"]["rows"] == 3

    def test_growth_does_not_trigger_collapse(self, fake_tree: Path) -> None:
        wd.write_baseline(wd.current_row_counts())
        pd.DataFrame({"value": range(50_000)}).to_parquet(wd.FACTORS_DIR / "factors_all.parquet")
        assert wd.run_watchdog(now=NOW)["status"] == "ok"

    def test_no_baseline_means_no_collapse_check(self, fake_tree: Path) -> None:
        """First-ever run must not fail merely because it has nothing to compare to."""
        snapshot = wd.run_watchdog(now=NOW)
        assert not any(c["name"].startswith("panel_size:") for c in snapshot["checks"])


class TestFreshness:
    def test_stale_panel_is_an_error(self, fake_tree: Path) -> None:
        late = NOW + timedelta(days=wd.MAX_PANEL_STALENESS_DAYS + 3)
        snapshot = wd.run_watchdog(now=late)
        assert snapshot["status"] == "error"
        freshness = next(c for c in snapshot["checks"] if c["name"] == "panel_freshness")
        assert freshness["status"] == "error"

    def test_weekend_gap_is_tolerated(self, fake_tree: Path) -> None:
        snapshot = wd.run_watchdog(now=NOW + timedelta(days=3))
        freshness = next(c for c in snapshot["checks"] if c["name"] == "panel_freshness")
        assert freshness["status"] == "ok"


class TestSnapshot:
    def test_frozen_panel_is_ok_when_it_reaches_the_snapshot(
        self, fake_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A panel that stops on the declared snapshot date is healthy, however old."""
        last = pd.read_parquet(wd.FACTORS_DIR / "prices.parquet").index.max()
        monkeypatch.setattr(wd, "FMP_SNAPSHOT_AS_OF", str(pd.Timestamp(last).date()))
        snapshot = wd.run_watchdog(now=NOW + timedelta(days=400))
        panel = next(c for c in snapshot["checks"] if c["name"] == "panel_freshness")
        assert panel["status"] == "ok" and "snapshot" in panel["detail"]

    def test_frozen_panel_short_of_snapshot_is_an_error(
        self, fake_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        last = pd.read_parquet(wd.FACTORS_DIR / "prices.parquet").index.max()
        declared = pd.Timestamp(last) + timedelta(days=30)
        monkeypatch.setattr(wd, "FMP_SNAPSHOT_AS_OF", str(declared.date()))
        snapshot = wd.run_watchdog(now=NOW)
        panel = next(c for c in snapshot["checks"] if c["name"] == "panel_freshness")
        assert panel["status"] == "error"


class TestScheduledJobs:
    def test_silent_job_is_an_error(self, fake_tree: Path) -> None:
        """A cron job that stops firing emits no error anywhere; its log just stops."""
        snapshot = wd.run_watchdog(now=NOW + timedelta(days=10))
        job = next(c for c in snapshot["checks"] if c["name"] == "job:update.log")
        assert job["status"] == "error"
        assert "stopped running" in job["detail"]

    def test_traceback_in_log_is_an_error(self, fake_tree: Path) -> None:
        write_log(wd.LOGS_DIR / "update.log", "fetching...\nTraceback (most recent call last):\n")
        snapshot = wd.run_watchdog(now=NOW)
        job = next(c for c in snapshot["checks"] if c["name"] == "job:update.log")
        assert job["status"] == "error"

    def test_traceback_before_a_later_success_is_not_an_error(self, fake_tree: Path) -> None:
        """A job that failed once and then ran clean must clear its own alarm."""
        write_log(
            wd.LOGS_DIR / "update.log",
            "fetching...\nTraceback (most recent call last):\n  boom\n"
            "fetching...\n🏁 Incremental update finished\n",
        )
        snapshot = wd.run_watchdog(now=NOW)
        job = next(c for c in snapshot["checks"] if c["name"] == "job:update.log")
        assert job["status"] == "ok"

    def test_traceback_after_the_last_success_is_an_error(self, fake_tree: Path) -> None:
        write_log(
            wd.LOGS_DIR / "update.log",
            "🏁 Incremental update finished\nfetching...\nTraceback (most recent call last):\n",
        )
        snapshot = wd.run_watchdog(now=NOW)
        job = next(c for c in snapshot["checks"] if c["name"] == "job:update.log")
        assert job["status"] == "error"

    def test_success_line_quoting_a_marker_does_not_self_incriminate(self, fake_tree: Path) -> None:
        """The watchdog's own verdict line mentions other jobs' tracebacks."""
        write_log(
            wd.LOGS_DIR / "watchdog.log",
            "INFO watchdog: Watchdog verdict: error — job:update.log log ends with Traceback\n",
        )
        snapshot = wd.run_watchdog(now=NOW)
        job = next(c for c in snapshot["checks"] if c["name"] == "job:watchdog.log")
        assert job["status"] == "ok"

    def test_missing_log_is_a_warning_not_an_error(self, fake_tree: Path) -> None:
        """A job that has never run is worth noting but does not invalidate data."""
        (wd.LOGS_DIR / "update.log").unlink()
        snapshot = wd.run_watchdog(now=NOW)
        job = next(c for c in snapshot["checks"] if c["name"] == "job:update.log")
        assert job["status"] == "warning"


class TestAcknowledgedInvariants:
    def test_by_design_violation_does_not_raise_a_warning(self, fake_tree: Path) -> None:
        """
        A monitor that is permanently yellow gets ignored.

        Extreme returns are always present in the raw panel by the ADR-0012
        fidelity rule, so they must report ok — with the reason attached, not
        hidden.
        """
        prices = pd.read_parquet(wd.FACTORS_DIR / "prices.parquet")
        prices.iloc[20:, 0] = prices.iloc[19, 0] * 1_000_000
        prices.to_parquet(wd.FACTORS_DIR / "prices.parquet")

        snapshot = wd.run_watchdog(deep=True, now=NOW)
        check = next(c for c in snapshot["checks"] if c["name"] == "invariant:extreme_return")
        assert check["status"] == "ok"
        assert check["facts"]["acknowledged"] is True
        assert "expected" in check["detail"]

    def test_unacknowledged_violation_still_escalates(self, fake_tree: Path) -> None:
        prices = pd.read_parquet(wd.FACTORS_DIR / "prices.parquet")
        prices.iloc[5, 0] = 0.0
        prices.to_parquet(wd.FACTORS_DIR / "prices.parquet")

        snapshot = wd.run_watchdog(deep=True, now=NOW)
        assert snapshot["status"] == "error"
        assert any(c["name"] == "invariant:non_positive_price" for c in snapshot["checks"])


class TestPersistence:
    def test_status_round_trips(self, fake_tree: Path) -> None:
        snapshot = wd.run_watchdog(now=NOW)
        wd.write_status(snapshot)
        assert wd.load_status() == snapshot

    def test_load_status_is_none_before_first_run(self, fake_tree: Path) -> None:
        assert wd.load_status() is None
