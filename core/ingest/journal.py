"""Durable record of what an ingestion run attempted, and what happened to it.

A multi-hour backfill over hundreds of thousands of requests fails in ways a log
file answers badly. "Which symbols have no earnings data because the vendor has
none, versus because we got a 502 at 3am?" is a query, not a grep — and it is the
question you actually need answered before trusting a panel built on the result.

So every task outcome is written to SQLite as a row: which endpoint, which
partition key, the HTTP status, row count, bytes, duration, attempt number, and
the error text when there was one. The journal is the audit trail for the raw
layer, and :mod:`core.ingest.report` renders it for humans.

Concurrency: worker threads call :meth:`IngestJournal.record` freely; writes are
serialised behind a lock and committed in batches, because SQLite under WAL still
serialises writers and a commit per task would dominate a fast endpoint's runtime.

Batching is bounded by time as well as by count. A purely count-based commit
loses the audit trail of any run that is killed before it accumulates a full
batch — including every run smaller than the batch size, and the slow tail of a
long one. Since the journal exists precisely to explain runs that did not finish
cleanly, it also commits on a timer.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Task outcomes. OK/EMPTY are both successes: EMPTY means the vendor genuinely
# has no rows for this key, which is information, not a failure.
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"

_COMMIT_EVERY = 200
_COMMIT_EVERY_SECONDS = 20.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    vendor      TEXT NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    argv        TEXT,
    n_tasks     INTEGER
);
CREATE TABLE IF NOT EXISTS tasks (
    run_id       TEXT NOT NULL,
    spec_name    TEXT NOT NULL,
    partition_key TEXT NOT NULL,
    status       TEXT NOT NULL,
    http_code    INTEGER,
    n_rows       INTEGER,
    n_bytes      INTEGER,
    duration_s   REAL,
    attempts     INTEGER,
    error        TEXT,
    recorded_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id, spec_name, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""


@dataclass
class TaskResult:
    """Outcome of one task, as written to the journal."""

    spec_name: str
    partition_key: str
    status: str
    http_code: Optional[int] = None
    n_rows: Optional[int] = None
    n_bytes: Optional[int] = None
    duration_s: Optional[float] = None
    attempts: int = 1
    error: Optional[str] = None


class IngestJournal:
    """
    SQLite-backed audit trail for ingestion runs.

    Args:
        path: Database file; parent directories are created.
        vendor: Vendor identifier stored on the run row.
        run_id: Unique run identifier.
        argv: Command line that started the run, for reproducibility.
    """

    def __init__(self, path: Path, vendor: str, run_id: str, argv: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._run_id = run_id
        self._lock = threading.Lock()
        self._pending = 0
        self._last_commit = time.monotonic()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._connection.execute(
            "INSERT OR REPLACE INTO runs (run_id, vendor, started_at, argv) VALUES (?,?,?,?)",
            (run_id, vendor, time.time(), argv),
        )
        self._connection.commit()

    @property
    def run_id(self) -> str:
        """Identifier of the run being journalled."""
        return self._run_id

    def record(self, result: TaskResult) -> None:
        """Append one task outcome. Safe to call from worker threads."""
        with self._lock:
            self._connection.execute(
                "INSERT INTO tasks (run_id, spec_name, partition_key, status, http_code,"
                " n_rows, n_bytes, duration_s, attempts, error, recorded_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self._run_id,
                    result.spec_name,
                    result.partition_key,
                    result.status,
                    result.http_code,
                    result.n_rows,
                    result.n_bytes,
                    result.duration_s,
                    result.attempts,
                    (result.error or "")[:2000] or None,
                    time.time(),
                ),
            )
            self._pending += 1
            now = time.monotonic()
            if self._pending >= _COMMIT_EVERY or now - self._last_commit >= _COMMIT_EVERY_SECONDS:
                self._connection.commit()
                self._pending = 0
                self._last_commit = now

    def set_task_count(self, n_tasks: int) -> None:
        """Store how many tasks the run planned to execute."""
        with self._lock:
            self._connection.execute(
                "UPDATE runs SET n_tasks=? WHERE run_id=?", (n_tasks, self._run_id)
            )
            self._connection.commit()

    def finish(self) -> None:
        """Flush pending rows and stamp the run as finished."""
        with self._lock:
            self._connection.execute(
                "UPDATE runs SET finished_at=? WHERE run_id=?", (time.time(), self._run_id)
            )
            self._connection.commit()
            self._pending = 0
            self._last_commit = time.monotonic()

    def close(self) -> None:
        """Commit and close the connection."""
        with self._lock:
            self._connection.commit()
            self._connection.close()

    def counts_by_status(self, run_id: Optional[str] = None) -> dict[str, int]:
        """Task counts grouped by status for a run (defaults to this run)."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) FROM tasks WHERE run_id=? GROUP BY status",
                (run_id or self._run_id,),
            ).fetchall()
        return {status: count for status, count in rows}

    def failures(self, run_id: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
        """Failed tasks with their error text, newest first."""
        with self._lock:
            cursor = self._connection.execute(
                "SELECT spec_name, partition_key, http_code, attempts, error FROM tasks"
                " WHERE run_id=? AND status=? ORDER BY recorded_at DESC LIMIT ?",
                (run_id or self._run_id, STATUS_FAILED, limit),
            )
            columns = [c[0] for c in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
