"""Expansion of endpoint specs into the concrete tasks a run will execute.

Planning is separated from execution so a run can be costed before it is started.
A 9,011-symbol universe times seventy per-symbol endpoints is 630,000 requests —
roughly a day at 500 calls/minute — and that number should be visible in a
``--dry-run`` before anyone waits for it, not discovered afterwards.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

import pandas as pd

from core.ingest.spec import EndpointSpec, Partition, Task

logger = logging.getLogger(__name__)

# Partition key used for endpoints that take no key at all.
GLOBAL_KEY = "_all"

# Filesystem-hostile characters seen in real tickers (BRK/B, RDS.A).
_KEY_TRANSLATIONS = str.maketrans({"/": "-", "\\": "-", ":": "-", " ": "_"})


def safe_key(raw: str) -> str:
    """Map a partition key to a filesystem-safe stem."""
    return raw.translate(_KEY_TRANSLATIONS)


def expand_spec(spec: EndpointSpec, keys: Mapping[Partition, Sequence[str]]) -> list[Task]:
    """
    Turn one spec into its tasks.

    Args:
        spec: Endpoint to expand.
        keys: Available partition keys by partition type — symbols, CIKs,
            sectors, and so on.

    Returns:
        Tasks in a stable order. Empty when the spec's partition has no keys
        available, which is logged rather than raised so one missing key source
        does not abort a whole run.
    """
    if spec.partition is Partition.GLOBAL:
        return [Task(spec.name, GLOBAL_KEY, dict(spec.params))]

    available = keys.get(spec.partition, ())
    if not available:
        logger.warning("no keys available for %s (%s); skipping", spec.name, spec.partition.value)
        return []

    if spec.partition is Partition.BATCH_SYMBOLS:
        tasks = []
        for start in range(0, len(available), spec.batch_size):
            chunk = list(available[start : start + spec.batch_size])
            tasks.append(
                Task(
                    spec.name,
                    f"batch_{start // spec.batch_size:05d}",
                    {**spec.params, "symbols": ",".join(chunk)},
                )
            )
        return tasks

    param_name = _PARAM_FOR_PARTITION[spec.partition]
    if spec.date_chunk_years:
        return [task for key in available for task in _windowed_tasks(spec, key, param_name)]
    return [Task(spec.name, safe_key(key), {**spec.params, param_name: key}) for key in available]


def date_windows(start: str, end: str, chunk_years: int) -> list[tuple[str, str]]:
    """
    Split ``[start, end]`` into consecutive windows of at most ``chunk_years``.

    Args:
        start: Inclusive first date, ISO format.
        end: Inclusive last date, ISO format.
        chunk_years: Window length in years.

    Returns:
        List of ``(from, to)`` ISO date pairs covering the range without gaps.
    """
    windows: list[tuple[str, str]] = []
    cursor = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while cursor <= final:
        window_end = min(cursor + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1), final)
        windows.append((cursor.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def _windowed_tasks(spec: EndpointSpec, key: str, param_name: str) -> list[Task]:
    """Expand one partition key into one task per date window."""
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    tasks = []
    for start, stop in date_windows(spec.history_start, end, spec.date_chunk_years or 5):
        tasks.append(
            Task(
                spec.name,
                f"{safe_key(key)}__{start[:4]}_{stop[:4]}",
                {**spec.params, param_name: key, "from": start, "to": stop},
            )
        )
    return tasks


_PARAM_FOR_PARTITION: dict[Partition, str] = {
    Partition.PER_SYMBOL: "symbol",
    Partition.PER_CIK: "cik",
    Partition.PER_EXCHANGE: "exchange",
    Partition.PER_SECTOR: "sector",
    Partition.PER_INDUSTRY: "industry",
    Partition.PER_NAME: "name",
}


def plan_run(
    specs: Iterable[EndpointSpec],
    keys: Mapping[Partition, Sequence[str]],
) -> list[Task]:
    """
    Expand many specs, preserving the order they were given in.

    Args:
        specs: Endpoints to ingest, already filtered and priority-sorted.
        keys: Partition keys by type.

    Returns:
        Flat task list ready for the runner.
    """
    tasks: list[Task] = []
    for spec in specs:
        expanded = expand_spec(spec, keys)
        logger.debug("%s -> %d tasks", spec.name, len(expanded))
        tasks.extend(expanded)
    return tasks
