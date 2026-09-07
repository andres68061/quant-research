"""
Unit tests for core.ingest.spec.

An EndpointSpec is validated at construction because the two mistakes it guards
against are both silent later: a mistyped point-in-time classification becomes a
leaked backtest months afterwards, and a name containing a path separator writes
raw files outside the directory the resume logic looks in.
"""

from __future__ import annotations

import pytest

from core.ingest.spec import (
    PERIOD_END_ONLY,
    PIT_STATUSES,
    POINT_IN_TIME,
    SNAPSHOT,
    EndpointSpec,
    Partition,
    Payload,
    Task,
)


def _spec(**overrides: object) -> EndpointSpec:
    """Build a minimal valid spec with the given fields overridden."""
    fields: dict[str, object] = {
        "name": "income_statement",
        "endpoint": "income-statement",
        "partition": Partition.PER_SYMBOL,
        "pit_status": POINT_IN_TIME,
    }
    fields.update(overrides)
    return EndpointSpec(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("pit_status", sorted(PIT_STATUSES))
def test_endpoint_spec_accepts_every_known_pit_status(pit_status: str) -> None:
    spec = _spec(pit_status=pit_status)

    assert spec.pit_status == pit_status
    assert spec.pit_status in PIT_STATUSES


def test_pit_status_constants_match_the_frozen_set() -> None:
    assert PIT_STATUSES == frozenset({POINT_IN_TIME, PERIOD_END_ONLY, SNAPSHOT})


@pytest.mark.parametrize("pit_status", ["pointintime", "POINT_IN_TIME", "", "unknown", None])
def test_endpoint_spec_rejects_unknown_pit_status(pit_status: object) -> None:
    with pytest.raises(ValueError, match="pit_status"):
        _spec(pit_status=pit_status)


@pytest.mark.parametrize("name", ["income/statement", "a/b/c", "/leading"])
def test_endpoint_spec_rejects_name_with_path_separator(name: str) -> None:
    with pytest.raises(ValueError, match="single path segment"):
        _spec(name=name)


def test_endpoint_spec_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="single path segment"):
        _spec(name="")


@pytest.mark.parametrize(
    "partition, expected",
    [
        (Partition.PER_SYMBOL, True),
        (Partition.BATCH_SYMBOLS, True),
        (Partition.GLOBAL, False),
        (Partition.PER_CIK, False),
        (Partition.PER_EXCHANGE, False),
        (Partition.PER_SECTOR, False),
        (Partition.PER_INDUSTRY, False),
        (Partition.PER_NAME, False),
    ],
)
def test_needs_universe_is_true_only_for_symbol_partitions(
    partition: Partition, expected: bool
) -> None:
    spec = _spec(partition=partition)

    assert spec.needs_universe is expected


def test_needs_universe_covers_every_partition_member() -> None:
    symbol_partitions = {Partition.PER_SYMBOL, Partition.BATCH_SYMBOLS}

    for partition in Partition:
        assert _spec(partition=partition).needs_universe is (partition in symbol_partitions)


def test_endpoint_spec_defaults_are_conservative() -> None:
    spec = _spec()

    assert spec.params == {}
    assert spec.date_columns == ()
    assert spec.primary_date is None
    assert spec.payload is Payload.JSON
    assert spec.priority == 3
    assert spec.derivable is False
    assert spec.paginate is False
    assert spec.date_chunk_years is None
    assert spec.history_start == "1985-01-01"


def test_endpoint_spec_is_frozen() -> None:
    spec = _spec()

    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        spec.name = "renamed"  # type: ignore[misc]


def test_partition_and_payload_are_string_enums() -> None:
    assert Partition("per_symbol") is Partition.PER_SYMBOL
    assert Payload("binary") is Payload.BINARY
    assert Partition.GLOBAL.value == "global"
    assert isinstance(Partition.GLOBAL, str)
    assert isinstance(Payload.JSON, str)


def test_task_carries_resolved_params() -> None:
    task = Task("income_statement", "AAPL", {"symbol": "AAPL", "limit": 5})

    assert task.spec_name == "income_statement"
    assert task.key == "AAPL"
    assert task.params == {"symbol": "AAPL", "limit": 5}
