"""Tests for :func:`~core.data.sp500_constituents.resolve_sp500_historical_csv`."""

from __future__ import annotations

import os
import time

import pytest

from core.data.sp500_constituents import SP500Constituents, resolve_sp500_historical_csv
from core.exceptions import ConfigError


def test_resolve_picks_newest_by_mtime(tmp_path) -> None:
    older = tmp_path / "S&P 500 Historical Components & Changes(01-01-2020).csv"
    newer = tmp_path / "S&P 500 Historical Components & Changes(01-01-2026).csv"
    older.write_text("date,tickers\n2020-01-02,A\n", encoding="utf-8")
    newer.write_text("date,tickers\n2020-01-02,B\n", encoding="utf-8")
    old_ts = (time.time() - 10_000.0, time.time() - 10_000.0)
    os.utime(older, old_ts)

    resolved = resolve_sp500_historical_csv(tmp_path)
    assert resolved == newer


def test_resolve_no_match_raises(tmp_path) -> None:
    with pytest.raises(ConfigError, match="No file matching"):
        resolve_sp500_historical_csv(tmp_path)


def test_resolve_data_dir_not_a_directory_raises(tmp_path) -> None:
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a directory"):
        resolve_sp500_historical_csv(not_a_dir)


def test_sp500_constituents_explicit_csv_path_loads(tmp_path) -> None:
    csv_path = tmp_path / "custom.csv"
    csv_path.write_text(
        'date,tickers\n2020-06-15,"AAA,BBB,CCC"\n',
        encoding="utf-8",
    )
    sp500 = SP500Constituents(csv_path=csv_path)
    df = sp500.load()
    assert len(df) == 1
    assert set(sp500.get_ticker_universe()) == {"AAA", "BBB", "CCC"}


def test_sp500_constituents_data_dir_pass_through(tmp_path) -> None:
    csv_path = tmp_path / "S&P 500 Historical Components & Changes(x).csv"
    csv_path.write_text(
        'date,tickers\n2020-06-15,"ZZZ"\n',
        encoding="utf-8",
    )
    sp500 = SP500Constituents(data_dir=tmp_path)
    assert sp500.csv_path == csv_path
    sp500.load()
    assert "ZZZ" in sp500.get_ticker_universe()
