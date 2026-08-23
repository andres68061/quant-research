"""Tests for core.index.top500."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import DataSchemaError
from core.index.top500 import (
    build_quarterly_rebalance_dates,
    compute_cap_weighted_index,
    select_top_n_by_market_cap,
)


def _trading_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, tz="America/New_York")


def test_build_quarterly_rebalance_dates_drops_incomplete_quarter() -> None:
    # Two full quarters plus a stub ending mid-August.
    days = _trading_days("2024-01-01", "2024-08-15")
    result = build_quarterly_rebalance_dates(days)
    assert len(result) == 2
    assert result[0] == days[days.quarter == 1].max()
    assert result[1] == days[days.quarter == 2].max()


def test_build_quarterly_rebalance_dates_respects_bounds() -> None:
    days = _trading_days("2023-01-01", "2024-12-31")
    result = build_quarterly_rebalance_dates(
        days, start=pd.Timestamp("2024-01-01", tz="America/New_York")
    )
    assert (result.year == 2024).all()
    assert len(result) == 4


def test_select_top_n_by_market_cap_picks_largest() -> None:
    days = _trading_days("2024-01-01", "2024-01-10")
    caps = pd.DataFrame(
        {"BIG": 300.0, "MID": 200.0, "SMALL": 100.0, "MISSING": np.nan},
        index=days,
    )
    top = select_top_n_by_market_cap(caps, days[-1], top_n=2)
    assert list(top.index) == ["BIG", "MID"]


def test_select_top_n_by_market_cap_no_history_raises() -> None:
    days = _trading_days("2024-06-01", "2024-06-10")
    caps = pd.DataFrame({"A": 1.0}, index=days)
    with pytest.raises(DataSchemaError):
        select_top_n_by_market_cap(caps, pd.Timestamp("2024-01-02", tz="America/New_York"))


def test_compute_cap_weighted_index_buy_and_hold_math() -> None:
    days = _trading_days("2024-01-01", "2024-07-15")
    prices = pd.DataFrame(
        {
            "A": np.linspace(100.0, 130.0, len(days)),
            "B": np.linspace(50.0, 45.0, len(days)),
        },
        index=days,
    )
    caps = prices * pd.Series({"A": 2.0, "B": 4.0})  # A = 2x B's cap at start
    rebalances = build_quarterly_rebalance_dates(days)
    result = compute_cap_weighted_index(prices, caps, rebalances, top_n=2)

    first = rebalances[0]
    holdings = result.holdings[first]
    assert set(holdings.index) == {"A", "B"}
    assert holdings["weight"].sum() == pytest.approx(1.0)

    # Hand-check the first post-execution day: buy-and-hold of fixed weights.
    exec_date = result.execution_dates[first]
    exec_pos = days.get_loc(exec_date)
    day1 = days[exec_pos + 1]
    w = holdings["weight"]
    rel = prices.loc[day1, w.index] / prices.loc[exec_date, w.index]
    expected = float((rel * w).sum() - 1.0)
    assert result.daily_returns.loc[day1] == pytest.approx(expected)

    # First rebalance has zero turnover; the second is small but defined.
    assert result.turnover.iloc[0] == 0.0
    assert 0.0 <= result.turnover.iloc[1] < 0.5


def test_compute_cap_weighted_index_freezes_delisted_symbol() -> None:
    days = _trading_days("2024-01-01", "2024-07-15")
    a = pd.Series(100.0, index=days)
    b = pd.Series(100.0, index=days)
    # B delists in mid-May: no prices afterwards.
    b[days > pd.Timestamp("2024-05-15", tz="America/New_York")] = np.nan
    prices = pd.DataFrame({"A": a, "B": b})
    caps = pd.DataFrame({"A": 100.0, "B": 100.0}, index=days)
    rebalances = build_quarterly_rebalance_dates(days)
    result = compute_cap_weighted_index(prices, caps, rebalances, top_n=2)
    # Flat prices + frozen delisting => zero returns throughout, no NaN.
    assert result.daily_returns.notna().all()
    assert result.daily_returns.abs().max() == pytest.approx(0.0)


def test_compute_cap_weighted_index_empty_rebalances_raises() -> None:
    days = _trading_days("2024-01-01", "2024-03-31")
    prices = pd.DataFrame({"A": 100.0}, index=days)
    with pytest.raises(DataSchemaError):
        compute_cap_weighted_index(prices, prices, pd.DatetimeIndex([]), top_n=1)
