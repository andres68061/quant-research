"""Tests for core.data.factors.fama_french (FF5 download, persist, dedup)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.data.factors.fama_french import (
    _COLUMN_MAP,
    fetch_ff5_daily,
    load_ff5_parquet,
    update_ff5_parquet,
)


def _mock_ff5_table(n: int = 100, start: str = "2020-01-02") -> dict:
    """Build a fake ``pandas_datareader.data.DataReader`` return value."""
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (n, 6))
    df = pd.DataFrame(data, index=idx, columns=list(_COLUMN_MAP.keys()))
    return {0: df, "DESCR": "mock"}


@patch("core.data.factors.fama_french.web.DataReader")
def test_fetch_ff5_columns_and_scale(mock_reader: object) -> None:
    mock_reader.return_value = _mock_ff5_table(50)  # type: ignore[attr-defined]
    df = fetch_ff5_daily()
    assert sorted(df.columns.tolist()) == sorted(_COLUMN_MAP.values())
    assert df.index.name == "date"
    raw = _mock_ff5_table(50)[0]
    np.testing.assert_allclose(df["mkt_rf"].values, raw["Mkt-RF"].values / 100.0)


@patch("core.data.factors.fama_french.web.DataReader")
def test_fetch_ff5_start_filter(mock_reader: object) -> None:
    mock_reader.return_value = _mock_ff5_table(50, start="2020-01-02")  # type: ignore[attr-defined]
    df = fetch_ff5_daily(start="2020-02-01")
    assert df.index.min() >= pd.Timestamp("2020-01-02")


@patch("core.data.factors.fama_french.web.DataReader")
def test_fetch_ff5_retries_on_failure(mock_reader: object) -> None:
    mock_reader.side_effect = [  # type: ignore[attr-defined]
        ConnectionError("fail"),
        ConnectionError("fail"),
        _mock_ff5_table(10),
    ]
    df = fetch_ff5_daily()
    assert len(df) == 10
    assert mock_reader.call_count == 3  # type: ignore[attr-defined]


@patch("core.data.factors.fama_french.web.DataReader")
def test_fetch_ff5_raises_after_max_retries(mock_reader: object) -> None:
    mock_reader.side_effect = ConnectionError("fail")  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="Failed to fetch"):
        fetch_ff5_daily()


def test_load_ff5_parquet_missing(tmp_path: Path) -> None:
    result = load_ff5_parquet(tmp_path / "nope.parquet")
    assert result is None


@patch("core.data.factors.fama_french.web.DataReader")
def test_update_ff5_round_trip(mock_reader: object, tmp_path: Path) -> None:
    mock_reader.return_value = _mock_ff5_table(30)  # type: ignore[attr-defined]
    out = tmp_path / "ff5.parquet"
    df = update_ff5_parquet(out)
    assert out.exists()
    assert len(df) == 30

    loaded = load_ff5_parquet(out)
    assert loaded is not None
    pd.testing.assert_frame_equal(df, loaded, check_freq=False)


@patch("core.data.factors.fama_french.web.DataReader")
def test_update_ff5_dedup(mock_reader: object, tmp_path: Path) -> None:
    out = tmp_path / "ff5.parquet"
    mock_reader.return_value = _mock_ff5_table(20)  # type: ignore[attr-defined]
    update_ff5_parquet(out)

    mock_reader.return_value = _mock_ff5_table(25)  # type: ignore[attr-defined]
    df2 = update_ff5_parquet(out)
    assert len(df2) == 25
