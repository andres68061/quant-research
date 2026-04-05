"""Tests for log_market_cap merge into factor pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.factors.build_factors import load_market_cap, merge_market_cap


@pytest.fixture()
def toy_factors() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=5, tz="America/New_York")
    rows = []
    for d in dates:
        for s in ["AAA", "BBB"]:
            rows.append({"date": d, "symbol": s, "mom_12_1": 0.1, "vol_60d": 0.2})
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


@pytest.fixture()
def toy_mcap(tmp_path: Path) -> Path:
    dates = pd.bdate_range("2024-01-02", periods=5, tz="America/New_York")
    rows = []
    for d in dates:
        for t in ["AAA", "BBB"]:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "market_cap": 1e9 + np.random.default_rng(0).random() * 1e8,
                }
            )
    mc = pd.DataFrame(rows).set_index(["date", "ticker"])
    path = tmp_path / "mc.parquet"
    mc.to_parquet(path)
    return path


def test_load_market_cap_missing(tmp_path: Path) -> None:
    result = load_market_cap(tmp_path / "nope.parquet")
    assert result is None


def test_load_market_cap_ok(toy_mcap: Path) -> None:
    mc = load_market_cap(toy_mcap)
    assert mc is not None
    assert "log_market_cap" in mc.columns
    assert mc.index.names == ["date", "symbol"]


def test_merge_market_cap_adds_column(toy_factors: pd.DataFrame, toy_mcap: Path) -> None:
    merged = merge_market_cap(toy_factors, toy_mcap)
    assert "log_market_cap" in merged.columns
    assert merged["log_market_cap"].notna().sum() > 0
    assert "mom_12_1" in merged.columns


def test_merge_market_cap_none_path(toy_factors: pd.DataFrame) -> None:
    result = merge_market_cap(toy_factors, None)
    assert "log_market_cap" not in result.columns
    pd.testing.assert_frame_equal(result, toy_factors)


def test_merge_market_cap_missing_file(toy_factors: pd.DataFrame, tmp_path: Path) -> None:
    result = merge_market_cap(toy_factors, tmp_path / "nope.parquet")
    assert "log_market_cap" not in result.columns
