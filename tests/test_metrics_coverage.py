"""Tests for invested-coverage disclosure and replay position labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.metrics.coverage import position_label_from_counts, summarize_invested_coverage
from core.replay.precompute import precompute_backtest_frames


def test_position_label_from_counts() -> None:
    assert position_label_from_counts(0, 0) == "flat"
    assert position_label_from_counts(5, 0) == "long"
    assert position_label_from_counts(0, 3) == "short"
    assert position_label_from_counts(4, 4) == "long_short"


def test_summarize_invested_coverage_warns_on_flat_stretch() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    n_long = pd.Series([5, 5, 0, 0, 0, 0, 5, 5, 5, 5], index=idx)
    n_short = pd.Series(0, index=idx)

    cov = summarize_invested_coverage(n_long, n_short, min_stocks=20)

    assert cov["n_days"] == 10
    assert cov["n_days_flat"] == 4
    assert cov["n_days_invested"] == 6
    assert cov["longest_flat_streak_days"] == 4
    assert cov["pct_days_invested"] == pytest.approx(0.6)
    assert cov["cash_earns_zero"] is True
    assert cov["warning"] is not None
    assert "min_stocks=20" in cov["warning"]
    assert "0%" in cov["warning"]


def test_summarize_invested_coverage_no_warning_when_mostly_invested() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    n_long = pd.Series(10, index=idx)
    n_long.iloc[50] = 0
    n_short = pd.Series(0, index=idx)

    cov = summarize_invested_coverage(n_long, n_short, min_stocks=20)

    assert cov["n_days_flat"] == 1
    assert cov["pct_days_invested"] == pytest.approx(0.99)
    assert cov["warning"] is None


def test_precompute_frames_use_n_long_n_short_for_position() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    net = pd.Series([0.01, 0.0, -0.01, 0.0, 0.02], index=idx)
    n_long = pd.Series([3, 0, 2, 0, 4], index=idx)
    n_short = pd.Series([3, 0, 0, 1, 0], index=idx)

    frames = precompute_backtest_frames(net, n_long=n_long, n_short=n_short)

    assert [f["position"] for f in frames] == [
        "long_short",
        "flat",
        "long",
        "short",
        "long",
    ]
    assert frames[1]["n_long"] == 0
    assert frames[1]["n_short"] == 0
    assert frames[0]["signal"] == 0.0
    assert frames[2]["signal"] == 2.0


def test_precompute_frames_without_counts_default_flat() -> None:
    """Legacy call site: no headcounts → position stays flat (known limitation)."""
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    net = pd.Series(np.zeros(3), index=idx)
    frames = precompute_backtest_frames(net)
    assert all(f["position"] == "flat" for f in frames)
    assert all(f["n_long"] is None for f in frames)
