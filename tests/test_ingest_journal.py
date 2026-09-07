"""
Unit tests for core.ingest.journal.

The journal is the audit trail a raw layer is trusted on the strength of, so the
properties that matter are durability and completeness: worker threads hammering
record() must not lose rows, the status grouping must separate "the vendor has no
data" from "we got a 502", and the file must still answer questions after the
process that wrote it has gone away.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import pytest

from core.ingest.journal import (
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    IngestJournal,
    TaskResult,
)


@pytest.fixture()
def journal_path(tmp_path: Path) -> Path:
    return tmp_path / "journal" / "ingest.sqlite"


@pytest.fixture()
def journal(journal_path: Path) -> Iterator[IngestJournal]:
    opened = IngestJournal(journal_path, vendor="fmp", run_id="run-1", argv="pytest")
    yield opened
    opened.close()


def _row_count(path: Path, run_id: str) -> int:
    """Count journalled task rows for one run using an independent connection."""
    connection = sqlite3.connect(str(path))
    try:
        return int(
            connection.execute("SELECT COUNT(*) FROM tasks WHERE run_id=?", (run_id,)).fetchone()[0]
        )
    finally:
        connection.close()


def test_journal_creates_parent_directories_and_run_row(journal_path: Path) -> None:
    opened = IngestJournal(journal_path, vendor="fmp", run_id="run-x")
    opened.close()

    assert journal_path.is_file()
    connection = sqlite3.connect(str(journal_path))
    try:
        vendor, n_tasks, finished = connection.execute(
            "SELECT vendor, n_tasks, finished_at FROM runs WHERE run_id=?", ("run-x",)
        ).fetchone()
    finally:
        connection.close()
    assert vendor == "fmp"
    assert n_tasks is None
    assert finished is None


def test_run_id_property_returns_the_configured_identifier(journal: IngestJournal) -> None:
    assert journal.run_id == "run-1"


def test_counts_by_status_groups_correctly(journal: IngestJournal) -> None:
    journal.record(TaskResult("quote", "AAPL", STATUS_OK, 200, 10, 100))
    journal.record(TaskResult("quote", "MSFT", STATUS_OK, 200, 8, 90))
    journal.record(TaskResult("quote", "ZZZZ", STATUS_EMPTY, 200, 0, 0))
    journal.record(TaskResult("quote", "FAIL", STATUS_FAILED, 502, error="HTTP 502"))
    journal.record(TaskResult("quote", "DONE", STATUS_SKIPPED, attempts=0))

    counts = journal.counts_by_status()

    assert counts == {STATUS_OK: 2, STATUS_EMPTY: 1, STATUS_FAILED: 1, STATUS_SKIPPED: 1}


def test_counts_by_status_returns_empty_dict_for_a_run_with_no_tasks(
    journal: IngestJournal,
) -> None:
    assert journal.counts_by_status() == {}
    assert journal.counts_by_status(run_id="never-ran") == {}


def test_counts_by_status_isolates_runs(journal_path: Path) -> None:
    first = IngestJournal(journal_path, vendor="fmp", run_id="run-a")
    first.record(TaskResult("quote", "AAPL", STATUS_OK))
    first.close()

    second = IngestJournal(journal_path, vendor="fmp", run_id="run-b")
    second.record(TaskResult("quote", "MSFT", STATUS_FAILED, 402, error="not entitled"))
    try:
        assert second.counts_by_status() == {STATUS_FAILED: 1}
        assert second.counts_by_status(run_id="run-a") == {STATUS_OK: 1}
    finally:
        second.close()


def test_record_from_many_threads_loses_no_rows(journal: IngestJournal) -> None:
    n_records = 600
    statuses = [STATUS_OK, STATUS_EMPTY, STATUS_FAILED, STATUS_SKIPPED]

    def write(index: int) -> None:
        journal.record(
            TaskResult(
                spec_name="quote",
                partition_key=f"SYM{index:04d}",
                status=statuses[index % len(statuses)],
                http_code=200,
                n_rows=index,
            )
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write, range(n_records)))

    counts = journal.counts_by_status()
    assert sum(counts.values()) == n_records
    assert set(counts) == set(statuses)
    assert counts[STATUS_OK] == n_records // len(statuses)


def test_failures_returns_error_text(journal: IngestJournal) -> None:
    journal.record(TaskResult("quote", "AAPL", STATUS_OK, 200, 5, 50))
    journal.record(
        TaskResult("ratios", "BRK-B", STATUS_FAILED, 502, attempts=4, error="HTTP 502 bad gateway")
    )

    failures = journal.failures()

    assert len(failures) == 1
    failure = failures[0]
    assert failure["spec_name"] == "ratios"
    assert failure["partition_key"] == "BRK-B"
    assert failure["http_code"] == 502
    assert failure["attempts"] == 4
    assert failure["error"] == "HTTP 502 bad gateway"


def test_failures_respects_the_limit_and_returns_empty_when_nothing_failed(
    journal: IngestJournal,
) -> None:
    assert journal.failures() == []

    for index in range(5):
        journal.record(TaskResult("quote", f"S{index}", STATUS_FAILED, 500, error=f"e{index}"))

    assert len(journal.failures(limit=2)) == 2
    assert len(journal.failures()) == 5


def test_record_stores_an_absent_error_as_null(journal: IngestJournal) -> None:
    journal.record(TaskResult("quote", "AAPL", STATUS_FAILED, 500, error=""))

    assert journal.failures()[0]["error"] is None


def test_record_truncates_a_very_long_error(journal: IngestJournal) -> None:
    journal.record(TaskResult("quote", "AAPL", STATUS_FAILED, 500, error="x" * 5000))

    assert len(journal.failures()[0]["error"]) == 2000


def test_set_task_count_and_finish_update_the_run_row(
    journal: IngestJournal, journal_path: Path
) -> None:
    journal.set_task_count(1234)
    journal.finish()

    connection = sqlite3.connect(str(journal_path))
    try:
        n_tasks, started, finished = connection.execute(
            "SELECT n_tasks, started_at, finished_at FROM runs WHERE run_id=?", ("run-1",)
        ).fetchone()
    finally:
        connection.close()

    assert n_tasks == 1234
    assert finished is not None
    assert finished >= started


def test_finish_flushes_pending_rows_to_disk(journal: IngestJournal, journal_path: Path) -> None:
    journal.record(TaskResult("quote", "AAPL", STATUS_OK, 200, 5, 50))
    journal.finish()

    assert _row_count(journal_path, "run-1") == 1


def test_journal_survives_close_and_reopen(journal_path: Path) -> None:
    first = IngestJournal(journal_path, vendor="fmp", run_id="run-1")
    first.record(TaskResult("quote", "AAPL", STATUS_OK, 200, 5, 50))
    first.record(TaskResult("quote", "ZZZZ", STATUS_EMPTY, 200, 0, 0))
    first.close()

    reopened = IngestJournal(journal_path, vendor="fmp", run_id="run-1")
    try:
        assert reopened.counts_by_status() == {STATUS_OK: 1, STATUS_EMPTY: 1}
        reopened.record(TaskResult("quote", "MSFT", STATUS_OK, 200, 3, 30))
        assert reopened.counts_by_status()[STATUS_OK] == 2
    finally:
        reopened.close()

    assert _row_count(journal_path, "run-1") == 3


def test_task_result_defaults_are_a_single_successful_attempt() -> None:
    result = TaskResult("quote", "AAPL", STATUS_OK)

    assert result.attempts == 1
    assert result.http_code is None
    assert result.n_rows is None
    assert result.n_bytes is None
    assert result.duration_s is None
    assert result.error is None
