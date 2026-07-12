"""Tests for portfolio optimizer helpers and coverage endpoints."""

from unittest.mock import patch

import pandas as pd
import pytest

from api.routes.portfolio import (
    MIN_OPTIMIZER_PRICE_ROWS,
    JointHistoryRequest,
    joint_history,
    price_row_counts,
)


@pytest.fixture
def toy_prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    return pd.DataFrame(
        {"VOO": range(100), "GLD": range(100)},
        index=idx,
    )


def test_price_row_counts_respects_start(toy_prices: pd.DataFrame) -> None:
    start = "2020-03-01"
    with patch("api.routes.portfolio.get_prices", return_value=toy_prices):
        out = price_row_counts(start_date=start)
    assert out["min_required"] == MIN_OPTIMIZER_PRICE_ROWS
    expected = int(toy_prices.loc[toy_prices.index >= start].shape[0])
    voo = out["symbols"]["VOO"]
    gld = out["symbols"]["GLD"]
    assert voo["count"] == expected
    assert gld["count"] == expected
    # First / last reflect the *full* panel, independent of the ``start`` filter.
    assert voo["first"] == "2020-01-01"
    assert voo["last"] == str(toy_prices.index.max().date())
    assert out["last_panel_date"] == str(toy_prices.index.max().date())


def test_price_row_counts_reports_delisted_window(toy_prices: pd.DataFrame) -> None:
    """A symbol whose tail is NaN should report ``last`` at its real last trade."""
    delisted = toy_prices.copy()
    delisted.loc[delisted.index[60:], "GLD"] = float("nan")
    with patch("api.routes.portfolio.get_prices", return_value=delisted):
        out = price_row_counts()
    gld = out["symbols"]["GLD"]
    voo = out["symbols"]["VOO"]
    assert gld["last"] == str(delisted.index[59].date())
    assert voo["last"] == str(delisted.index.max().date())
    # Count over the full window equals the number of non-null observations.
    assert gld["count"] == 60


def test_joint_history_eligible(toy_prices: pd.DataFrame) -> None:
    with patch("api.routes.portfolio.get_prices", return_value=toy_prices):
        r = joint_history(JointHistoryRequest(symbols=["VOO", "GLD"], start_date="2020-01-02"))
    assert r["eligible"] is True
    assert r["joint_rows"] >= MIN_OPTIMIZER_PRICE_ROWS


def test_joint_history_short_overlap(toy_prices: pd.DataFrame) -> None:
    toy2 = toy_prices.copy()
    toy2.loc[toy2.index[:80], "GLD"] = float("nan")
    with patch("api.routes.portfolio.get_prices", return_value=toy2):
        r = joint_history(JointHistoryRequest(symbols=["VOO", "GLD"], start_date="2020-01-01"))
    assert r["joint_rows"] == 20
    assert r["eligible"] is False
