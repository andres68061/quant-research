"""Tests for guarded artifact writes.

Every test here corresponds to a way a derived panel has been, or could be,
silently destroyed by a run that looked successful.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.data.store.artifacts import (
    SCRATCH_DIRNAME,
    ArtifactShrinkError,
    resolve_artifact_path,
    streamed_artifact,
    write_artifact,
)


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"value": range(rows)})


class TestSubsetRedirect:
    def test_full_build_uses_the_production_path(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        assert resolve_artifact_path(target, subset=False) == target

    def test_subset_build_cannot_reach_the_production_path(self, tmp_path: Path) -> None:
        """The actual incident: --symbols AAPL,MSFT,XOM replaced a 23M-row panel."""
        target = tmp_path / "panel.parquet"
        redirected = resolve_artifact_path(target, subset=True)
        assert redirected != target
        assert redirected.parent.name == SCRATCH_DIRNAME
        assert redirected.name == target.name

    def test_subset_write_leaves_the_production_artifact_untouched(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        write_artifact(_frame(1000), target)

        write_artifact(_frame(3), target, subset=True)

        assert pq.ParquetFile(target).metadata.num_rows == 1000
        assert (tmp_path / SCRATCH_DIRNAME / "panel.parquet").exists()


class TestShrinkGuard:
    def test_new_artifact_writes_without_a_baseline(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        write_artifact(_frame(100), target)
        assert pq.ParquetFile(target).metadata.num_rows == 100

    def test_large_shrink_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        write_artifact(_frame(1000), target)
        with pytest.raises(ArtifactShrinkError, match="refusing to replace"):
            write_artifact(_frame(10), target)
        assert pq.ParquetFile(target).metadata.num_rows == 1000

    def test_small_shrink_is_allowed(self, tmp_path: Path) -> None:
        """Rebuilds legitimately lose a few rows; only a collapse is a bug."""
        target = tmp_path / "panel.parquet"
        write_artifact(_frame(1000), target)
        write_artifact(_frame(950), target)
        assert pq.ParquetFile(target).metadata.num_rows == 950

    def test_growth_is_always_allowed(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        write_artifact(_frame(100), target)
        write_artifact(_frame(50_000), target)
        assert pq.ParquetFile(target).metadata.num_rows == 50_000

    def test_explicit_allow_shrink_overrides(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        write_artifact(_frame(1000), target)
        write_artifact(_frame(1), target, allow_shrink=True)
        assert pq.ParquetFile(target).metadata.num_rows == 1

    def test_no_temp_file_survives_a_refused_write(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        write_artifact(_frame(1000), target)
        with pytest.raises(ArtifactShrinkError):
            write_artifact(_frame(1), target)
        assert not list(tmp_path.glob("*.tmp"))


class TestStreamedArtifact:
    def _stream(self, path: Path, rows: int) -> None:
        table = pa.Table.from_pandas(_frame(rows))
        writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        writer.close()

    def test_streamed_build_publishes_to_the_target(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        with streamed_artifact(target) as temp:
            self._stream(temp, 500)
        assert pq.ParquetFile(target).metadata.num_rows == 500

    def test_streamed_shrink_is_refused_and_target_survives(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        with streamed_artifact(target) as temp:
            self._stream(temp, 5000)

        with pytest.raises(ArtifactShrinkError):
            with streamed_artifact(target) as temp:
                self._stream(temp, 12)

        assert pq.ParquetFile(target).metadata.num_rows == 5000

    def test_streamed_subset_redirects(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        with streamed_artifact(target) as temp:
            self._stream(temp, 5000)

        with streamed_artifact(target, subset=True) as temp:
            self._stream(temp, 3)

        assert pq.ParquetFile(target).metadata.num_rows == 5000
        assert (tmp_path / SCRATCH_DIRNAME / "panel.parquet").exists()

    def test_empty_build_leaves_the_previous_artifact(self, tmp_path: Path) -> None:
        """A build that produced nothing is a failure, not an instruction to empty the panel."""
        target = tmp_path / "panel.parquet"
        with streamed_artifact(target) as temp:
            self._stream(temp, 400)

        with streamed_artifact(target) as temp:
            self._stream(temp, 0)

        assert pq.ParquetFile(target).metadata.num_rows == 400

    def test_exception_mid_stream_leaves_the_previous_artifact(self, tmp_path: Path) -> None:
        target = tmp_path / "panel.parquet"
        with streamed_artifact(target) as temp:
            self._stream(temp, 400)

        with pytest.raises(RuntimeError, match="vendor timeout"):
            with streamed_artifact(target) as temp:
                self._stream(temp, 10)
                raise RuntimeError("vendor timeout")

        assert pq.ParquetFile(target).metadata.num_rows == 400
        assert not list(tmp_path.glob("*.tmp"))
