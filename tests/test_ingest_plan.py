"""
Unit tests for core.ingest.plan.

Planning is where a partition key becomes both a filename and a query parameter,
and those two must not be the same string: "BRK/B" is a valid ticker and an
invalid path. The tests below pin that split, the batching arithmetic, the date
windowing that keeps a capped endpoint from silently truncating history, and the
decision to return no tasks (rather than raise) when a key source is missing.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from core.ingest.plan import GLOBAL_KEY, date_windows, expand_spec, plan_run, safe_key
from core.ingest.spec import EndpointSpec, Partition, Task


def _spec(**overrides: object) -> EndpointSpec:
    """Build a minimal valid spec with the given fields overridden."""
    fields: dict[str, object] = {
        "name": "quote",
        "endpoint": "quote",
        "partition": Partition.PER_SYMBOL,
        "pit_status": "point_in_time",
    }
    fields.update(overrides)
    return EndpointSpec(**fields)  # type: ignore[arg-type]


def test_safe_key_maps_filesystem_hostile_characters() -> None:
    assert safe_key("BRK/B") == "BRK-B"
    assert safe_key("A\\B") == "A-B"
    assert safe_key("NYSE:IBM") == "NYSE-IBM"
    assert safe_key("Real Estate") == "Real_Estate"
    assert safe_key("AAPL") == "AAPL"


def test_safe_key_returns_empty_for_empty_input() -> None:
    assert safe_key("") == ""


def test_expand_spec_global_yields_exactly_one_task() -> None:
    spec = _spec(name="cik_list", partition=Partition.GLOBAL, params={"limit": 1000})

    tasks = expand_spec(spec, {Partition.PER_SYMBOL: ["AAPL", "MSFT"]})

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, Task)
    assert task.key == GLOBAL_KEY == "_all"
    assert task.spec_name == "cik_list"
    assert task.params == {"limit": 1000}


def test_expand_spec_global_copies_params_rather_than_aliasing_them() -> None:
    spec = _spec(partition=Partition.GLOBAL, params={"limit": 1000})

    task = expand_spec(spec, {})[0]
    task.params["limit"] = 1

    assert spec.params == {"limit": 1000}


def test_expand_spec_per_symbol_keeps_real_ticker_in_params_and_sanitises_the_key() -> None:
    spec = _spec(partition=Partition.PER_SYMBOL)

    tasks = expand_spec(spec, {Partition.PER_SYMBOL: ["AAPL", "BRK/B"]})

    assert [task.key for task in tasks] == ["AAPL", "BRK-B"]
    # The key is a filename; the parameter must remain the ticker the vendor knows.
    assert [task.params["symbol"] for task in tasks] == ["AAPL", "BRK/B"]


def test_expand_spec_per_symbol_merges_static_params() -> None:
    spec = _spec(partition=Partition.PER_SYMBOL, params={"period": "quarter", "limit": 40})

    task = expand_spec(spec, {Partition.PER_SYMBOL: ["AAPL"]})[0]

    assert task.params == {"period": "quarter", "limit": 40, "symbol": "AAPL"}


@pytest.mark.parametrize(
    "partition, param_name",
    [
        (Partition.PER_CIK, "cik"),
        (Partition.PER_EXCHANGE, "exchange"),
        (Partition.PER_SECTOR, "sector"),
        (Partition.PER_INDUSTRY, "industry"),
        (Partition.PER_NAME, "name"),
    ],
)
def test_expand_spec_uses_the_right_query_parameter_per_partition(
    partition: Partition, param_name: str
) -> None:
    spec = _spec(partition=partition)

    task = expand_spec(spec, {partition: ["KEY-1"]})[0]

    assert task.params == {param_name: "KEY-1"}


def test_expand_spec_batch_symbols_chunks_with_a_ragged_final_chunk() -> None:
    spec = _spec(partition=Partition.BATCH_SYMBOLS, batch_size=3)
    symbols = ["A", "B", "C", "D", "E", "F", "G"]

    tasks = expand_spec(spec, {Partition.BATCH_SYMBOLS: symbols})

    assert [task.key for task in tasks] == ["batch_00000", "batch_00001", "batch_00002"]
    assert [task.params["symbols"] for task in tasks] == ["A,B,C", "D,E,F", "G"]
    rejoined = ",".join(task.params["symbols"] for task in tasks).split(",")
    assert rejoined == symbols


def test_expand_spec_batch_symbols_with_exact_multiple_has_no_trailing_empty_chunk() -> None:
    spec = _spec(partition=Partition.BATCH_SYMBOLS, batch_size=2)

    tasks = expand_spec(spec, {Partition.BATCH_SYMBOLS: ["A", "B", "C", "D"]})

    assert len(tasks) == 2
    assert all(task.params["symbols"] for task in tasks)


def test_expand_spec_returns_empty_list_when_no_keys_are_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec = _spec(name="ratios", partition=Partition.PER_SYMBOL)

    with caplog.at_level(logging.WARNING, logger="core.ingest.plan"):
        tasks = expand_spec(spec, {})

    assert tasks == []
    assert "no keys available for ratios" in caplog.text


def test_expand_spec_returns_empty_list_for_an_empty_key_sequence() -> None:
    spec = _spec(partition=Partition.PER_CIK)

    assert expand_spec(spec, {Partition.PER_CIK: []}) == []


def test_expand_spec_date_chunking_produces_contiguous_windows_per_key() -> None:
    spec = _spec(
        name="eod",
        partition=Partition.PER_SYMBOL,
        date_chunk_years=5,
        history_start="2000-01-01",
    )

    tasks = expand_spec(spec, {Partition.PER_SYMBOL: ["BRK/B"]})

    assert len(tasks) >= 2
    assert all(task.params["symbol"] == "BRK/B" for task in tasks)
    assert all(task.key.startswith("BRK-B__") for task in tasks)
    assert tasks[0].params["from"] == "2000-01-01"
    for earlier, later in zip(tasks, tasks[1:], strict=False):
        gap = pd.Timestamp(later.params["from"]) - pd.Timestamp(earlier.params["to"])
        assert gap == pd.Timedelta(days=1)


def test_date_windows_covers_the_range_without_gaps_or_overlaps() -> None:
    windows = date_windows("2020-01-01", "2024-06-30", 1)

    assert windows[0][0] == "2020-01-01"
    assert windows[-1][1] == "2024-06-30"
    for (_, earlier_end), (later_start, _) in zip(windows, windows[1:], strict=False):
        assert pd.Timestamp(later_start) - pd.Timestamp(earlier_end) == pd.Timedelta(days=1)
    for start, end in windows:
        assert pd.Timestamp(start) <= pd.Timestamp(end)
        assert isinstance(start, str) and isinstance(end, str)


def test_date_windows_never_runs_past_the_end_date() -> None:
    windows = date_windows("2020-01-01", "2020-07-15", 5)

    assert windows == [("2020-01-01", "2020-07-15")]


def test_date_windows_single_day_range_yields_one_window() -> None:
    assert date_windows("2021-03-04", "2021-03-04", 1) == [("2021-03-04", "2021-03-04")]


def test_date_windows_returns_empty_when_start_is_after_end() -> None:
    assert date_windows("2024-01-01", "2023-01-01", 5) == []


def test_date_windows_chunk_length_matches_the_requested_years() -> None:
    windows = date_windows("2010-01-01", "2019-12-31", 5)

    assert windows == [("2010-01-01", "2014-12-31"), ("2015-01-01", "2019-12-31")]


def test_plan_run_preserves_spec_order_and_flattens_tasks() -> None:
    global_spec = _spec(name="cik_list", partition=Partition.GLOBAL)
    symbol_spec = _spec(name="ratios", partition=Partition.PER_SYMBOL)

    tasks = plan_run(
        [global_spec, symbol_spec],
        {Partition.PER_SYMBOL: ["AAPL", "MSFT"]},
    )

    assert [(task.spec_name, task.key) for task in tasks] == [
        ("cik_list", "_all"),
        ("ratios", "AAPL"),
        ("ratios", "MSFT"),
    ]


def test_plan_run_with_no_specs_returns_no_tasks() -> None:
    assert plan_run([], {Partition.PER_SYMBOL: ["AAPL"]}) == []


def test_plan_run_skips_specs_without_keys_but_keeps_the_rest() -> None:
    keyless = _spec(name="insider_trades", partition=Partition.PER_CIK)
    global_spec = _spec(name="cik_list", partition=Partition.GLOBAL)

    tasks = plan_run([keyless, global_spec], {Partition.PER_SYMBOL: ["AAPL"]})

    assert [task.spec_name for task in tasks] == ["cik_list"]
