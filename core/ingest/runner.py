"""Concurrent, resumable execution of ingestion tasks.

The runner is the vendor-agnostic half of the pipeline: it knows how to expand
specs into tasks, how to keep a worker pool inside a rate ceiling, how to retry,
how to store a payload so an interrupted run resumes cleanly, and how to record
every outcome. It knows nothing about any particular vendor — the caller supplies
a ``fetch`` callable.

Three properties matter more than speed:

- **Resumable.** Completion is a file on disk, not an in-memory cursor. A run
  killed at hour six re-plans and skips what landed. Writes are atomic (temp file
  plus rename), so a process killed mid-write cannot leave a truncated file that
  the next run mistakes for complete.
- **Honest about empty.** A vendor with no rows for a symbol is recorded as
  ``empty`` and marked done, not retried forever and not confused with an error.
- **Interruptible.** SIGINT/SIGTERM stops scheduling new work and drains the pool,
  so stopping a 22-hour backfill is safe at any moment.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from core.ingest.journal import (
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    IngestJournal,
    TaskResult,
)
from core.ingest.ratelimit import TokenBucket
from core.ingest.spec import EndpointSpec, Payload, Subject, Task

# Reserved column stamped onto every stored row, naming the partition key the
# request was made with. Without it the only record of which company a file was
# fetched for is its filename, which is lost the moment files are concatenated
# into a panel.
REQUEST_KEY_COLUMN = "_request_key"

logger = logging.getLogger(__name__)

# HTTP statuses worth another attempt; 402/403/404 are verdicts, not hiccups.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 1.5

# Suffix order matters only for lookup; any one of these means "already fetched".
_OUTPUT_SUFFIXES = (".parquet", ".json.gz", ".bin")


def output_path(raw_root: Path, spec_name: str, key: str, suffix: str) -> Path:
    """
    Build a task's output path by APPENDING the suffix.

    ``Path.with_suffix`` replaces everything after the last dot, so the ticker
    ``RDS.A`` would be stored as ``RDS.parquet`` and ``RDS.B`` would then find
    that file and report itself already fetched — one file for two securities,
    with the second one's data silently never downloaded. Appending keeps
    ``RDS.A.parquet`` distinct from ``RDS.B.parquet``.

    Args:
        raw_root: Vendor raw layer root.
        spec_name: Endpoint directory.
        key: Partition key, already filesystem-safe.
        suffix: Extension to append, including the leading dot.

    Returns:
        Full path to the output file.
    """
    return raw_root / spec_name / f"{key}{suffix}"


@dataclass
class FetchResult:
    """One HTTP response, normalised for the runner."""

    status_code: int
    rows: Optional[list[dict[str, Any]]] = None
    content: Optional[bytes] = None
    error: Optional[str] = None
    partial: bool = False
    """True when a page walk ended early, so the stored rows are incomplete."""


@dataclass
class RunSummary:
    """Aggregate outcome of a run."""

    run_id: str
    planned: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    interrupted: bool = False

    @property
    def completed(self) -> int:
        """Tasks that produced a stored file or a recorded empty."""
        return self.counts.get(STATUS_OK, 0) + self.counts.get(STATUS_EMPTY, 0)


FetchCallable = Callable[[str, dict[str, Any]], FetchResult]


def check_identity(frame: pd.DataFrame, spec: EndpointSpec, key: str) -> Optional[str]:
    """
    Confirm a payload describes the entity it was requested for.

    Only meaningful for specs whose subject is the request key. A response whose
    ``symbol`` column names a different company is the failure this exists to
    catch: it would otherwise be stored under the requested ticker and silently
    attribute one company's fundamentals to another.

    Args:
        frame: Parsed payload.
        spec: Owning endpoint spec.
        key: Partition key the request was made with.

    Returns:
        A description of the mismatch, or None when the payload checks out or
        the check does not apply.
    """
    if spec.subject is not Subject.REQUEST_KEY:
        return None
    if frame.empty or "symbol" not in frame.columns:
        return None
    found = {str(value) for value in frame["symbol"].dropna().unique()}
    if not found or key in found:
        return None
    return f"identity mismatch: requested {key!r}, payload contains {sorted(found)[:3]}"


def store_payload(
    result: FetchResult, spec: EndpointSpec, path: Path, key: str = ""
) -> tuple[int, int]:
    """
    Persist one payload atomically and report what was written.

    Nested vendor JSON (whole financial reports, for instance) does not always
    survive a DataFrame round-trip, so parquet is attempted first and gzipped
    JSON is the fallback rather than the task failing.

    Args:
        result: Successful fetch result.
        spec: Owning endpoint spec.
        path: Destination path without a suffix decision applied.
        key: Partition key the request was made with; stamped onto every row as
            :data:`REQUEST_KEY_COLUMN` so the association survives concatenation
            into a panel, where the filename is gone.

    Returns:
        ``(n_rows, n_bytes)`` actually written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if spec.payload is Payload.BINARY:
        content = result.content or b""
        target = Path(f"{path}.bin")
        temporary = Path(f"{target}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, target)
        return 1, len(content)

    rows = result.rows or []
    try:
        frame = pd.DataFrame(rows)
        for column in spec.date_columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if spec.primary_date and spec.primary_date in frame.columns:
            frame = frame.sort_values(spec.primary_date).reset_index(drop=True)
        if key:
            frame[REQUEST_KEY_COLUMN] = key
        target = Path(f"{path}.parquet")
        temporary = Path(f"{target}.tmp")
        frame.to_parquet(temporary)
        os.replace(temporary, target)
        return len(frame), target.stat().st_size
    except Exception as exc:  # noqa: BLE001 - fall back rather than lose the payload
        logger.debug("parquet write failed for %s (%s); storing JSON", path.name, exc)
        target = Path(f"{path}.json.gz")
        temporary = Path(f"{target}.tmp")
        stamped = [{**row, REQUEST_KEY_COLUMN: key} for row in rows] if key else rows
        temporary.write_bytes(gzip.compress(json.dumps(stamped, default=str).encode()))
        os.replace(temporary, target)
        return len(rows), target.stat().st_size


def existing_output(raw_root: Path, spec: EndpointSpec, key: str) -> Optional[Path]:
    """Return the stored file for a task if any suffix variant already exists."""
    for suffix in _OUTPUT_SUFFIXES:
        candidate = output_path(raw_root, spec.name, key, suffix)
        if candidate.exists():
            return candidate
    return None


class IngestRunner:
    """
    Executes tasks concurrently under a rate ceiling, journalling every outcome.

    Args:
        fetch: Vendor call returning a :class:`FetchResult`; must be thread-safe.
        raw_root: Root of the vendor's raw layer, e.g. ``data/raw/fmp``.
        journal: Open journal for this run.
        bucket: Shared rate limiter.
        workers: Thread pool size.
        max_retries: Attempts per task before it is recorded as failed.
    """

    def __init__(
        self,
        fetch: FetchCallable,
        raw_root: Path,
        journal: IngestJournal,
        bucket: TokenBucket,
        workers: int = 12,
        max_retries: int = 4,
    ) -> None:
        self._fetch = fetch
        self._raw_root = raw_root
        self._journal = journal
        self._bucket = bucket
        self._workers = workers
        self._max_retries = max_retries
        self._stop = threading.Event()

    @property
    def stopping(self) -> bool:
        """Whether a stop has been requested and new work should not start."""
        return self._stop.is_set()

    def request_stop(self) -> None:
        """Stop scheduling new tasks; in-flight work drains."""
        self._stop.set()

    def install_signal_handlers(self) -> None:
        """Make SIGINT/SIGTERM drain the pool instead of killing it mid-write."""

        def handler(signum: int, _frame: Any) -> None:
            logger.warning("signal %s received; draining in-flight tasks", signum)
            self.request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handler)

    def run_task(self, task: Task, spec: EndpointSpec, force: bool = False) -> TaskResult:
        """
        Execute one task: skip, fetch with retries, store, and report.

        Args:
            task: Resolved unit of work.
            spec: Owning endpoint spec.
            force: Re-fetch even when an output file already exists.

        Returns:
            The outcome to journal.
        """
        if not force and existing_output(self._raw_root, spec, task.key) is not None:
            return TaskResult(spec.name, task.key, STATUS_SKIPPED, attempts=0)

        started = time.monotonic()
        last_error: Optional[str] = None
        last_code: Optional[int] = None
        attempt = 0

        for attempt in range(1, self._max_retries + 1):
            if self._stop.is_set():
                return TaskResult(
                    spec.name,
                    task.key,
                    STATUS_FAILED,
                    last_code,
                    attempts=attempt - 1,
                    error="run interrupted before completion",
                )

            if spec.paginate:
                from core.ingest.paginate import fetch_all_pages

                result = fetch_all_pages(
                    self._fetch,
                    self._bucket.acquire,
                    spec.endpoint,
                    task.params,
                    spec.page_size,
                    spec.max_pages,
                )
            else:
                self._bucket.acquire()
                result = self._fetch(spec.endpoint, task.params)
            last_code = result.status_code

            if result.status_code == 200:
                self._bucket.recover()
                rows = result.rows or []
                if spec.payload is Payload.JSON and not rows:
                    # Record the vendor's "no data" verdict so it is not retried
                    # on every future run.
                    path = output_path(self._raw_root, spec.name, task.key, ".parquet")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = Path(f"{path}.tmp")
                    pd.DataFrame().to_parquet(temporary)
                    os.replace(temporary, path)
                    return TaskResult(
                        spec.name,
                        task.key,
                        STATUS_EMPTY,
                        200,
                        0,
                        path.stat().st_size,
                        time.monotonic() - started,
                        attempt,
                    )
                mismatch = check_identity(pd.DataFrame(rows), spec, task.key)
                if mismatch:
                    # Keep the payload — it is real data and discarding it would
                    # lose the evidence — but record the mismatch so the run
                    # report surfaces it, instead of a panel builder silently
                    # attributing one company's data to another.
                    logger.error("%s/%s: %s", spec.name, task.key, mismatch)
                n_rows, n_bytes = store_payload(
                    result, spec, self._raw_root / spec.name / task.key, task.key
                )
                notes = [
                    note
                    for note in (
                        "partial: page walk ended early" if result.partial else None,
                        mismatch,
                    )
                    if note
                ]
                return TaskResult(
                    spec.name,
                    task.key,
                    STATUS_OK,
                    200,
                    n_rows,
                    n_bytes,
                    time.monotonic() - started,
                    attempt,
                    error="; ".join(notes) or None,
                )

            last_error = result.error or f"HTTP {result.status_code}"
            if result.status_code == 429:
                self._bucket.penalize()
            if result.status_code not in RETRYABLE_STATUS:
                break
            if attempt < self._max_retries:
                time.sleep(_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

        return TaskResult(
            spec.name,
            task.key,
            STATUS_FAILED,
            last_code,
            duration_s=time.monotonic() - started,
            attempts=attempt,
            error=last_error,
        )
