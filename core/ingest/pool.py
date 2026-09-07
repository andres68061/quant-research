"""Worker-pool driver: runs planned tasks and reports progress as it goes.

Kept apart from :mod:`core.ingest.runner` so the per-task semantics (retry,
store, journal) can be tested without a thread pool, and so the progress
reporting a multi-hour run needs does not clutter them.
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
import time
from typing import Mapping, Sequence

from core.ingest.journal import STATUS_FAILED, IngestJournal
from core.ingest.runner import IngestRunner, RunSummary
from core.ingest.spec import EndpointSpec, Task

logger = logging.getLogger(__name__)

_PROGRESS_EVERY_SECONDS = 30.0


def execute(
    runner: IngestRunner,
    journal: IngestJournal,
    tasks: Sequence[Task],
    specs: Mapping[str, EndpointSpec],
    workers: int = 12,
    force: bool = False,
) -> RunSummary:
    """
    Run every task through the pool, journalling outcomes and logging progress.

    Args:
        runner: Configured runner.
        journal: Open journal; receives one row per task.
        tasks: Planned work.
        specs: Spec lookup by name.
        workers: Thread pool size.
        force: Re-fetch tasks whose output already exists.

    Returns:
        Aggregate :class:`RunSummary`.
    """
    summary = RunSummary(run_id=journal.run_id, planned=len(tasks))
    journal.set_task_count(len(tasks))
    started = time.monotonic()
    last_report = started
    done = 0

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(runner.run_task, task, specs[task.spec_name], force): task for task in tasks
        }
        for future in futures.as_completed(pending):
            task = pending[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - a crashed worker is a failed task
                logger.exception("task %s/%s raised", task.spec_name, task.key)
                from core.ingest.journal import TaskResult

                result = TaskResult(
                    task.spec_name, task.key, STATUS_FAILED, error=f"{type(exc).__name__}: {exc}"
                )
            journal.record(result)
            summary.counts[result.status] = summary.counts.get(result.status, 0) + 1
            done += 1

            now = time.monotonic()
            if now - last_report >= _PROGRESS_EVERY_SECONDS:
                elapsed = now - started
                rate = done / elapsed * 60 if elapsed else 0.0
                remaining = (len(tasks) - done) / rate if rate else 0.0
                logger.info(
                    "progress %d/%d (%.1f%%) | %.0f tasks/min | eta %.1f min | %s",
                    done,
                    len(tasks),
                    100 * done / max(len(tasks), 1),
                    rate,
                    remaining,
                    ", ".join(f"{k}={v}" for k, v in sorted(summary.counts.items())),
                )
                last_report = now

    summary.elapsed_seconds = time.monotonic() - started
    summary.interrupted = summary.interrupted or runner.stopping
    journal.finish()
    return summary
