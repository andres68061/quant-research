"""Tests for core.data.fmp.prices parsing and chunking (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from core.data.fmp.prices import (
    PANEL_TIMEZONE,
    generate_date_chunks,
    parse_dividend_adjusted_rows,
)
from core.exceptions import DataSchemaError


def _row(date: str, close: float, volume: int = 100) -> dict:
    return {
        "symbol": "TEST",
        "date": date,
        "adjOpen": close * 0.99,
        "adjHigh": close * 1.01,
        "adjLow": close * 0.98,
        "adjClose": close,
        "volume": volume,
    }


class TestParseDividendAdjustedRows:
    def test_happy_path_sorted_tz_aware_float64(self) -> None:
        # Newest-first input (the API's order) must come out sorted ascending.
        daily_bars = parse_dividend_adjusted_rows(
            [_row("2024-01-03", 11.0), _row("2024-01-02", 10.0)]
        )
        assert list(daily_bars["adj_close"]) == [10.0, 11.0]
        assert daily_bars.index.is_monotonic_increasing
        assert str(daily_bars.index.tz) == PANEL_TIMEZONE
        assert daily_bars["adj_close"].dtype == "float64"
        assert daily_bars["volume"].dtype == "int64"

    def test_duplicate_dates_deduplicated(self) -> None:
        daily_bars = parse_dividend_adjusted_rows(
            [_row("2024-01-02", 10.0), _row("2024-01-02", 10.5)]
        )
        assert len(daily_bars) == 1

    def test_empty_input_returns_empty_schema(self) -> None:
        daily_bars = parse_dividend_adjusted_rows([])
        assert daily_bars.empty
        assert list(daily_bars.columns) == [
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "volume",
        ]
        assert str(daily_bars.index.tz) == PANEL_TIMEZONE

    def test_missing_fields_raises_schema_error(self) -> None:
        with pytest.raises(DataSchemaError):
            parse_dividend_adjusted_rows([{"symbol": "X", "date": "2024-01-02"}])


class TestGenerateDateChunks:
    def test_single_chunk_when_range_fits(self) -> None:
        chunks = generate_date_chunks(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30"))
        assert chunks == [("2020-01-01", "2021-06-30")]

    def test_chunks_are_contiguous_and_cover_range(self) -> None:
        start, end = pd.Timestamp("1985-01-01"), pd.Timestamp("2026-07-11")
        chunks = generate_date_chunks(start, end)
        assert chunks[0][0] == "1985-01-01"
        assert chunks[-1][1] == "2026-07-11"
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert pd.Timestamp(next_start) == pd.Timestamp(prev_end) + pd.Timedelta(days=1)

    def test_start_after_end_raises(self) -> None:
        with pytest.raises(DataSchemaError):
            generate_date_chunks(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01"))
