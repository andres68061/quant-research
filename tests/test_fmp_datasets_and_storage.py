"""Tests for the FMP dataset registry, raw-layer storage helpers, and intraday chunking."""

from __future__ import annotations

import pandas as pd
import pytest

from core.data.fmp.datasets import (
    PERIOD_END_ONLY,
    POINT_IN_TIME,
    SNAPSHOT,
    SYMBOL_DATASETS,
    describe_datasets,
    fetch_symbol_dataset,
)
from core.data.fmp.intraday import (
    INTRADAY_INTERVALS,
    apply_split_adjustment,
    generate_intraday_chunks,
    parse_intraday_rows,
)
from core.data.fmp.storage import load_fetch_windows, safe_filename, write_atomic
from core.exceptions import DataSchemaError


class TestDatasetRegistry:
    def test_every_spec_declares_a_known_pit_status(self) -> None:
        valid = {POINT_IN_TIME, PERIOD_END_ONLY, SNAPSHOT}
        for name, spec in SYMBOL_DATASETS.items():
            assert spec.pit_status in valid, f"{name} has pit_status {spec.pit_status!r}"

    def test_registry_key_matches_spec_name(self) -> None:
        for key, spec in SYMBOL_DATASETS.items():
            assert key == spec.name

    def test_every_spec_has_notes(self) -> None:
        for name, spec in SYMBOL_DATASETS.items():
            assert spec.notes.strip(), f"{name} has no notes"

    def test_primary_date_is_declared_as_a_date_column(self) -> None:
        for name, spec in SYMBOL_DATASETS.items():
            if spec.primary_date is not None:
                assert spec.primary_date in spec.date_columns, name

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(DataSchemaError):
            fetch_symbol_dataset("AAPL", "not_a_dataset")

    def test_describe_returns_one_row_per_dataset(self) -> None:
        described = describe_datasets()
        assert len(described) == len(SYMBOL_DATASETS)
        assert set(described.columns) == {"dataset", "endpoint", "pit_status", "notes"}


class TestStorageHelpers:
    def test_write_atomic_leaves_no_temp_file(self, tmp_path) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3]})
        destination = tmp_path / "x.parquet"
        write_atomic(frame, destination)

        assert destination.exists()
        assert not list(tmp_path.glob("*.tmp"))
        pd.testing.assert_frame_equal(pd.read_parquet(destination), frame)

    def test_write_atomic_overwrites_cleanly(self, tmp_path) -> None:
        destination = tmp_path / "x.parquet"
        write_atomic(pd.DataFrame({"a": [1]}), destination)
        write_atomic(pd.DataFrame({"a": [9, 9]}), destination)
        assert len(pd.read_parquet(destination)) == 2

    def test_safe_filename_strips_path_separators(self) -> None:
        assert safe_filename("BRK/B") == "BRK-B"
        assert safe_filename("AAPL") == "AAPL"

    def test_fetch_windows_narrow_to_listed_life(self, tmp_path) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["LIVE", "DEAD"],
                "ipo_date": [pd.NaT, pd.Timestamp("1998-05-01")],
                "delisted_date": [pd.NaT, pd.Timestamp("2003-07-15")],
            }
        )
        path = tmp_path / "u.parquet"
        universe.to_parquet(path, index=False)

        windows = load_fetch_windows(
            path,
            default_start=pd.Timestamp("1985-01-01"),
            default_end=pd.Timestamp("2026-01-01"),
        )
        assert windows["LIVE"] == (pd.Timestamp("1985-01-01"), pd.Timestamp("2026-01-01"))
        dead_start, dead_end = windows["DEAD"]
        assert dead_start == pd.Timestamp("1998-05-01")
        # Delisting date plus the tail buffer.
        assert pd.Timestamp("2003-07-15") < dead_end < pd.Timestamp("2003-08-01")

    def test_fetch_windows_ignore_inverted_ranges(self, tmp_path) -> None:
        universe = pd.DataFrame(
            {
                "symbol": ["ODD"],
                "ipo_date": [pd.Timestamp("2020-01-01")],
                "delisted_date": [pd.Timestamp("1990-01-01")],
            }
        )
        path = tmp_path / "u.parquet"
        universe.to_parquet(path, index=False)
        start, end = load_fetch_windows(path)["ODD"]
        assert start < end

    def test_no_universe_file_means_no_windows(self) -> None:
        assert load_fetch_windows(None) == {}


class TestIntradayChunking:
    def test_chunks_cover_the_range_without_gaps_or_overlap(self) -> None:
        chunks = generate_intraday_chunks(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"), chunk_days=3
        )
        assert chunks[0][0] == "2024-01-01"
        assert chunks[-1][1] == "2024-01-10"
        for (_, previous_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert pd.Timestamp(next_start) == pd.Timestamp(previous_end) + pd.Timedelta(days=1)

    def test_every_interval_chunk_stays_under_its_bar_cap(self) -> None:
        """A chunk must not be able to request more bars than the endpoint returns."""
        bars_per_trading_day = {
            "1min": 390,
            "5min": 78,
            "15min": 26,
            "30min": 13,
            "1hour": 7,
            "4hour": 2,
        }
        for name, spec in INTRADAY_INTERVALS.items():
            trading_days = spec.chunk_days * 5 / 7
            assert trading_days * bars_per_trading_day[name] <= spec.bar_cap, name

    def test_inverted_range_raises(self) -> None:
        with pytest.raises(DataSchemaError):
            generate_intraday_chunks(
                pd.Timestamp("2024-02-01"), pd.Timestamp("2024-01-01"), chunk_days=3
            )


class TestIntradayParsing:
    def test_parses_and_sorts_bars(self) -> None:
        rows = [
            {
                "date": "2024-01-02 10:00:00",
                "open": 2.0,
                "high": 3.0,
                "low": 1.0,
                "close": 2.5,
                "volume": 20,
            },
            {
                "date": "2024-01-02 09:30:00",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 10,
            },
        ]
        bars = parse_intraday_rows(rows)
        assert list(bars.index.hour) == [9, 10]
        assert str(bars.index.tz) == "America/New_York"
        assert bars["volume"].dtype == "int64"

    def test_empty_input_returns_typed_empty_frame(self) -> None:
        bars = parse_intraday_rows([])
        assert bars.empty
        assert list(bars.columns) == ["open", "high", "low", "close", "volume"]

    def test_missing_fields_raise(self) -> None:
        with pytest.raises(DataSchemaError):
            parse_intraday_rows([{"date": "2024-01-02 09:30:00", "open": 1.0}])


class TestSplitAdjustment:
    def _bars(self) -> pd.DataFrame:
        index = pd.DatetimeIndex(
            ["2020-08-28 10:00", "2020-08-31 10:00", "2020-09-01 10:00"],
            tz="America/New_York",
            name="date",
        )
        return pd.DataFrame(
            {
                "open": [400.0, 100.0, 101.0],
                "high": [404.0, 101.0, 102.0],
                "low": [396.0, 99.0, 100.0],
                "close": [400.0, 100.0, 101.0],
                "volume": [1000, 4000, 4000],
            },
            index=index,
        )

    def test_pre_split_prices_are_divided_by_the_ratio(self) -> None:
        splits = pd.DataFrame({"date": ["2020-08-31"], "numerator": [4], "denominator": [1]})
        adjusted = apply_split_adjustment(self._bars(), splits)
        # The 4:1 split makes the pre-split 400 comparable to the post-split 100.
        assert adjusted["close"].iloc[0] == 100.0
        assert adjusted["close"].iloc[1] == 100.0
        assert adjusted["volume"].iloc[0] == 4000

    def test_no_splits_returns_input_unchanged(self) -> None:
        bars = self._bars()
        pd.testing.assert_frame_equal(apply_split_adjustment(bars, pd.DataFrame()), bars)

    def test_identity_ratio_is_ignored(self) -> None:
        bars = self._bars()
        splits = pd.DataFrame({"date": ["2020-08-31"], "numerator": [1], "denominator": [1]})
        pd.testing.assert_frame_equal(apply_split_adjustment(bars, splits), bars)
