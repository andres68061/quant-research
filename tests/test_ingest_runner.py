"""
Unit tests for core.ingest.runner.

This is the file that pins the runner's three promises. *Resumable*: a task whose
output file already exists is skipped without a request, and every write is
atomic so a killed process cannot leave a truncated file the next run mistakes
for complete. *Honest about empty*: a 200 carrying no rows is the vendor saying
"no data for this key", which is recorded as ``empty`` and marked done rather
than retried on every future run. *Frugal with retries*: 500/502/503/429 are
hiccups worth another attempt, while 402 (unentitled) and 404 (no such resource)
are verdicts — retrying a 402 four times across 9,011 symbols burns 36,000 calls
to learn nothing.

No test here touches the network: ``IngestRunner`` takes ``fetch`` as a
constructor argument precisely so a recording fake can stand in for it, and every
file lands under ``tmp_path``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd
import pytest

from core.ingest import runner as runner_module
from core.ingest.journal import (
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    IngestJournal,
)
from core.ingest.ratelimit import TokenBucket
from core.ingest.runner import (
    REQUEST_KEY_COLUMN,
    RETRYABLE_STATUS,
    FetchResult,
    IngestRunner,
    RunSummary,
    existing_output,
    store_payload,
)
from core.ingest.spec import (
    PERIOD_END_ONLY,
    EndpointSpec,
    Partition,
    Payload,
    Subject,
    Task,
)


class RecordingFetch:
    """Fake vendor transport: records every call, replays scripted results."""

    def __init__(self, results: Sequence[FetchResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, endpoint: str, params: dict[str, Any]) -> FetchResult:
        self.calls.append((endpoint, dict(params)))
        index = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[index]

    @property
    def n_calls(self) -> int:
        return len(self.calls)


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse retry backoff so retry tests run in milliseconds, not seconds."""
    monkeypatch.setattr(runner_module, "_BACKOFF_BASE_SECONDS", 0.0)


@pytest.fixture()
def raw_root(tmp_path: Path) -> Path:
    return tmp_path / "raw" / "fmp"


@pytest.fixture()
def journal(tmp_path: Path) -> Iterator[IngestJournal]:
    opened = IngestJournal(tmp_path / "ingest.sqlite", vendor="fmp", run_id="run-1")
    yield opened
    opened.close()


@pytest.fixture()
def bucket() -> TokenBucket:
    # A rate high enough that pacing never dominates the test, but a real bucket
    # so penalize/recover accounting is exercised rather than mocked.
    return TokenBucket(60_000)


def _spec(**overrides: object) -> EndpointSpec:
    """Build a minimal valid spec with the given fields overridden."""
    fields: dict[str, object] = {
        "name": "ratios",
        "endpoint": "ratios",
        "partition": Partition.PER_SYMBOL,
        "pit_status": "point_in_time",
    }
    fields.update(overrides)
    return EndpointSpec(**fields)  # type: ignore[arg-type]


def _runner(
    fetch: RecordingFetch,
    raw_root: Path,
    journal: IngestJournal,
    bucket: TokenBucket,
    max_retries: int = 3,
) -> IngestRunner:
    return IngestRunner(fetch, raw_root, journal, bucket, workers=1, max_retries=max_retries)


def _task(key: str = "AAPL") -> Task:
    return Task("ratios", key, {"symbol": key})


ROWS: list[dict[str, Any]] = [
    {"symbol": "AAPL", "date": "2024-03-31", "pe": 27.1},
    {"symbol": "AAPL", "date": "2024-06-30", "pe": 29.4},
    {"symbol": "AAPL", "date": "2023-12-31", "pe": 25.8},
]


# --------------------------------------------------------------------------
# run_task: resume semantics
# --------------------------------------------------------------------------


def test_run_task_skips_when_output_exists_and_does_not_fetch(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    existing = raw_root / spec.name / "AAPL.parquet"
    existing.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS).to_parquet(existing)
    fetch = RecordingFetch([FetchResult(200, rows=ROWS)])

    result = _runner(fetch, raw_root, journal, bucket).run_task(_task(), spec)

    assert result.status == STATUS_SKIPPED
    assert result.attempts == 0
    assert fetch.n_calls == 0
    assert fetch.calls == []


def test_run_task_force_refetches_even_when_output_exists(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    existing = raw_root / spec.name / "AAPL.parquet"
    existing.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": "AAPL", "pe": 1.0}]).to_parquet(existing)
    fetch = RecordingFetch([FetchResult(200, rows=ROWS)])

    result = _runner(fetch, raw_root, journal, bucket).run_task(_task(), spec, force=True)

    assert fetch.n_calls == 1
    assert result.status == STATUS_OK
    assert result.n_rows == len(ROWS)
    assert len(pd.read_parquet(existing)) == len(ROWS)


# --------------------------------------------------------------------------
# run_task: successful responses
# --------------------------------------------------------------------------


def test_run_task_writes_parquet_and_reports_row_count(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec(date_columns=("date",), primary_date="date")
    fetch = RecordingFetch([FetchResult(200, rows=ROWS)])

    result = _runner(fetch, raw_root, journal, bucket).run_task(_task(), spec)

    assert result.status == STATUS_OK
    assert result.http_code == 200
    assert result.n_rows == 3
    assert result.attempts == 1
    assert result.n_bytes is not None and result.n_bytes > 0
    assert result.duration_s is not None and result.duration_s >= 0.0

    stored = raw_root / "ratios" / "AAPL.parquet"
    assert stored.is_file()
    frame = pd.read_parquet(stored)
    assert len(frame) == 3
    assert frame["date"].is_monotonic_increasing
    assert fetch.calls == [("ratios", {"symbol": "AAPL"})]


def test_run_task_empty_list_is_recorded_empty_not_failed(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(200, rows=[])])

    result = _runner(fetch, raw_root, journal, bucket).run_task(_task("ZZZZ"), spec)

    assert result.status == STATUS_EMPTY
    assert result.status != STATUS_FAILED
    assert result.http_code == 200
    assert result.n_rows == 0
    assert fetch.n_calls == 1

    marker = raw_root / "ratios" / "ZZZZ.parquet"
    assert marker.is_file()
    assert pd.read_parquet(marker).empty


def test_run_task_empty_marker_makes_the_next_run_skip_rather_than_refetch(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(200, rows=[])])
    ingest = _runner(fetch, raw_root, journal, bucket)

    first = ingest.run_task(_task("ZZZZ"), spec)
    second = ingest.run_task(_task("ZZZZ"), spec)

    assert first.status == STATUS_EMPTY
    assert second.status == STATUS_SKIPPED
    assert fetch.n_calls == 1


def test_run_task_binary_payload_with_no_rows_is_stored_not_marked_empty(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec(name="filing_pdf", payload=Payload.BINARY)
    fetch = RecordingFetch([FetchResult(200, content=b"%PDF-1.7 payload")])

    result = _runner(fetch, raw_root, journal, bucket).run_task(
        Task("filing_pdf", "AAPL", {}), spec
    )

    assert result.status == STATUS_OK
    assert (raw_root / "filing_pdf" / "AAPL.bin").read_bytes() == b"%PDF-1.7 payload"


# --------------------------------------------------------------------------
# run_task: failure semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", sorted(RETRYABLE_STATUS))
def test_run_task_retries_a_retryable_status_then_records_failed(
    status_code: int, raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    assert status_code in RETRYABLE_STATUS
    spec = _spec()
    fetch = RecordingFetch([FetchResult(status_code, error=f"HTTP {status_code}")])

    result = _runner(fetch, raw_root, journal, bucket, max_retries=3).run_task(_task(), spec)

    assert fetch.n_calls == 3
    assert result.status == STATUS_FAILED
    assert result.http_code == status_code
    assert result.attempts == 3
    assert result.error == f"HTTP {status_code}"
    assert existing_output(raw_root, spec, "AAPL") is None


@pytest.mark.parametrize("status_code", [402, 404])
def test_run_task_does_not_retry_a_non_retryable_status(
    status_code: int, raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    assert status_code not in RETRYABLE_STATUS
    spec = _spec()
    fetch = RecordingFetch([FetchResult(status_code, error=f"HTTP {status_code}")])

    result = _runner(fetch, raw_root, journal, bucket, max_retries=4).run_task(_task(), spec)

    # A verdict, not a hiccup: one call, not four times 9,011 symbols.
    assert fetch.n_calls == 1
    assert result.status == STATUS_FAILED
    assert result.http_code == status_code
    assert result.attempts == 1


def test_run_task_succeeds_on_a_later_attempt_after_a_transient_failure(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(503, error="HTTP 503"), FetchResult(200, rows=ROWS)])

    result = _runner(fetch, raw_root, journal, bucket, max_retries=4).run_task(_task(), spec)

    assert fetch.n_calls == 2
    assert result.status == STATUS_OK
    assert result.attempts == 2
    assert result.n_rows == len(ROWS)


def test_run_task_penalizes_the_bucket_on_429(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(429, error="Limit Reach")])

    _runner(fetch, raw_root, journal, bucket, max_retries=2).run_task(_task(), spec)

    stats = bucket.stats
    assert stats.penalties == 2
    assert stats.current_rate_per_minute < 60_000


def test_run_task_does_not_penalize_the_bucket_on_a_server_error(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(500, error="HTTP 500")])

    _runner(fetch, raw_root, journal, bucket, max_retries=2).run_task(_task(), spec)

    assert bucket.stats.penalties == 0


def test_run_task_returns_failed_without_fetching_when_stop_is_requested(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(200, rows=ROWS)])
    ingest = _runner(fetch, raw_root, journal, bucket)
    ingest.request_stop()

    result = ingest.run_task(_task(), spec)

    assert fetch.n_calls == 0
    assert result.status == STATUS_FAILED
    assert result.error == "run interrupted before completion"


# --------------------------------------------------------------------------
# store_payload
# --------------------------------------------------------------------------


def test_store_payload_writes_parquet_sorted_by_primary_date(tmp_path: Path) -> None:
    spec = _spec(date_columns=("date",), primary_date="date")

    n_rows, n_bytes = store_payload(FetchResult(200, rows=ROWS), spec, tmp_path / "ratios" / "AAPL")

    stored = tmp_path / "ratios" / "AAPL.parquet"
    assert stored.is_file()
    assert n_rows == 3
    assert n_bytes == stored.stat().st_size
    frame = pd.read_parquet(stored)
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2023-12-31", "2024-03-31", "2024-06-30"]


def test_store_payload_falls_back_to_json_gz_when_parquet_round_trip_fails(
    tmp_path: Path,
) -> None:
    spec = _spec(name="financial_report")
    # A column holding a struct in one row and a scalar in another is exactly the
    # shape whole-report vendor JSON takes, and pyarrow refuses to type it.
    nested_rows: list[dict[str, Any]] = [
        {"symbol": "AAPL", "value": 1},
        {"symbol": "AAPL", "value": {"q1": 1, "q2": 2}},
    ]

    n_rows, n_bytes = store_payload(
        FetchResult(200, rows=nested_rows), spec, tmp_path / "financial_report" / "AAPL"
    )

    stored = tmp_path / "financial_report" / "AAPL.json.gz"
    assert stored.is_file()
    assert not (tmp_path / "financial_report" / "AAPL.parquet").exists()
    assert n_rows == 2
    assert n_bytes == stored.stat().st_size
    assert json.loads(gzip.decompress(stored.read_bytes()).decode()) == nested_rows


def test_store_payload_writes_bin_for_binary_payload(tmp_path: Path) -> None:
    spec = _spec(name="filing_pdf", payload=Payload.BINARY)
    content = b"\x00\x01binary-bytes\xff"

    n_rows, n_bytes = store_payload(
        FetchResult(200, content=content), spec, tmp_path / "filing_pdf" / "AAPL"
    )

    stored = tmp_path / "filing_pdf" / "AAPL.bin"
    assert stored.read_bytes() == content
    assert n_rows == 1
    assert n_bytes == len(content)


def test_store_payload_binary_with_no_content_writes_an_empty_file(tmp_path: Path) -> None:
    spec = _spec(name="filing_pdf", payload=Payload.BINARY)

    n_rows, n_bytes = store_payload(FetchResult(200), spec, tmp_path / "filing_pdf" / "AAPL")

    assert (tmp_path / "filing_pdf" / "AAPL.bin").read_bytes() == b""
    assert (n_rows, n_bytes) == (1, 0)


def test_store_payload_creates_missing_parent_directories(tmp_path: Path) -> None:
    spec = _spec()
    target_stem = tmp_path / "deep" / "nested" / "ratios" / "AAPL"

    store_payload(FetchResult(200, rows=ROWS), spec, target_stem)

    assert target_stem.with_suffix(".parquet").is_file()


@pytest.mark.parametrize(
    "result, spec_overrides",
    [
        (FetchResult(200, rows=ROWS), {}),
        (FetchResult(200, rows=[{"a": 1}, {"a": {"b": 2}}]), {}),
        (FetchResult(200, content=b"bytes"), {"payload": Payload.BINARY}),
    ],
)
def test_store_payload_leaves_no_tmp_files_behind(
    result: FetchResult, spec_overrides: dict[str, Any], tmp_path: Path
) -> None:
    spec = _spec(**spec_overrides)

    store_payload(result, spec, tmp_path / "ratios" / "AAPL")

    assert list(tmp_path.rglob("*.tmp")) == []


def test_run_task_leaves_no_tmp_files_behind(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(200, rows=ROWS)])
    ingest = _runner(fetch, raw_root, journal, bucket)

    ingest.run_task(_task("AAPL"), spec)
    ingest.run_task(_task("MSFT"), spec)

    assert list(raw_root.rglob("*.tmp")) == []


def test_run_task_empty_branch_leaves_no_tmp_files_behind(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(200, rows=[])])

    _runner(fetch, raw_root, journal, bucket).run_task(_task("ZZZZ"), spec)

    assert list(raw_root.rglob("*.tmp")) == []


# --------------------------------------------------------------------------
# existing_output
# --------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".parquet", ".json.gz", ".bin"])
def test_existing_output_finds_every_stored_suffix(suffix: str, raw_root: Path) -> None:
    spec = _spec()
    stored = raw_root / spec.name / f"AAPL{suffix}"
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(b"payload")

    found = existing_output(raw_root, spec, "AAPL")

    assert found == stored


# Regression: Path.with_suffix REPLACES a trailing segment, so the tickers RDS.A
# and RDS.B both resolved to RDS.parquet — the second was reported "skipped" and
# its data was never fetched. Output paths now append their suffix instead.
def test_run_task_does_not_let_two_dotted_keys_collide_on_one_file(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(200, rows=ROWS)])
    ingest = _runner(fetch, raw_root, journal, bucket)

    first = ingest.run_task(Task("ratios", "RDS.A", {"symbol": "RDS.A"}), spec)
    second = ingest.run_task(Task("ratios", "RDS.B", {"symbol": "RDS.B"}), spec)

    assert first.status == STATUS_OK
    assert second.status == STATUS_OK
    assert len(list((raw_root / "ratios").iterdir())) == 2


# Regression: max_retries=0 left the loop variable unbound, crashing the worker
# with UnboundLocalError instead of journalling a failed task.
def test_run_task_with_zero_retries_returns_failed_rather_than_raising(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _spec()
    fetch = RecordingFetch([FetchResult(500, error="HTTP 500")])

    result = _runner(fetch, raw_root, journal, bucket, max_retries=0).run_task(_task(), spec)

    assert result.status == STATUS_FAILED
    assert fetch.n_calls == 0


def test_existing_output_returns_none_when_nothing_was_stored(raw_root: Path) -> None:
    assert existing_output(raw_root, _spec(), "AAPL") is None


def test_existing_output_is_scoped_to_the_spec_directory(raw_root: Path) -> None:
    other = raw_root / "other_endpoint" / "AAPL.parquet"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"payload")

    assert existing_output(raw_root, _spec(name="ratios"), "AAPL") is None


# --------------------------------------------------------------------------
# RunSummary
# --------------------------------------------------------------------------


def test_run_summary_completed_counts_ok_and_empty_only() -> None:
    summary = RunSummary(
        run_id="run-1",
        planned=10,
        counts={STATUS_OK: 4, STATUS_EMPTY: 3, STATUS_FAILED: 2, STATUS_SKIPPED: 1},
    )

    assert summary.completed == 7


def test_run_summary_completed_is_zero_for_an_empty_run() -> None:
    summary = RunSummary(run_id="run-1")

    assert summary.completed == 0
    assert summary.counts == {}
    assert summary.interrupted is False


# --- payload identity -------------------------------------------------------
# The framework binds data to a company by the request key, so the question
# "could this file hold the wrong company's earnings?" must have a mechanical
# answer, not a hopeful one.


def _identity_spec(subject: Subject = Subject.REQUEST_KEY) -> EndpointSpec:
    return EndpointSpec(
        name="ratios",
        endpoint="ratios",
        partition=Partition.PER_SYMBOL,
        pit_status=PERIOD_END_ONLY,
        subject=subject,
    )


def test_store_payload_stamps_the_request_key_onto_every_row(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    """The company a file was fetched for must survive concatenation."""
    spec = _identity_spec()
    fetch = RecordingFetch([FetchResult(200, rows=[{"value": 1}, {"value": 2}])])
    ingest = _runner(fetch, raw_root, journal, bucket)

    result = ingest.run_task(Task("ratios", "AAPL", {"symbol": "AAPL"}), spec)

    assert result.status == STATUS_OK
    stored = pd.read_parquet(raw_root / "ratios" / "AAPL.parquet")
    assert list(stored[REQUEST_KEY_COLUMN]) == ["AAPL", "AAPL"]


def test_run_task_flags_a_payload_naming_a_different_company(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    """A response for the wrong company is recorded, not silently stored clean."""
    spec = _identity_spec()
    fetch = RecordingFetch([FetchResult(200, rows=[{"symbol": "MSFT", "value": 1}])])
    ingest = _runner(fetch, raw_root, journal, bucket)

    result = ingest.run_task(Task("ratios", "AAPL", {"symbol": "AAPL"}), spec)

    assert result.status == STATUS_OK
    assert result.error is not None
    assert "identity mismatch" in result.error
    assert "AAPL" in result.error and "MSFT" in result.error


def test_run_task_does_not_flag_related_subject_endpoints(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    """stock-peers returns other companies by design; that is not a mismatch."""
    spec = _identity_spec(subject=Subject.RELATED)
    fetch = RecordingFetch([FetchResult(200, rows=[{"symbol": "CRS"}, {"symbol": "JBHT"}])])
    ingest = _runner(fetch, raw_root, journal, bucket)

    result = ingest.run_task(Task("ratios", "LUV", {"symbol": "LUV"}), spec)

    assert result.status == STATUS_OK
    assert result.error is None
    stored = pd.read_parquet(raw_root / "ratios" / "LUV.parquet")
    # The subject is recoverable even though no row names LUV.
    assert set(stored[REQUEST_KEY_COLUMN]) == {"LUV"}


def test_run_task_accepts_a_payload_whose_symbol_matches_the_key(
    raw_root: Path, journal: IngestJournal, bucket: TokenBucket
) -> None:
    spec = _identity_spec()
    fetch = RecordingFetch([FetchResult(200, rows=[{"symbol": "AAPL", "value": 1}])])
    ingest = _runner(fetch, raw_root, journal, bucket)

    result = ingest.run_task(Task("ratios", "AAPL", {"symbol": "AAPL"}), spec)

    assert result.status == STATUS_OK
    assert result.error is None
