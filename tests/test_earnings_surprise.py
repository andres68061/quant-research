"""Tests for announcement-dated earnings surprise (SUE) factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.factors.earnings_surprise import (
    SURPRISE_COLUMNS,
    add_days_since_announcement,
    compute_announcement_surprises,
)
from core.exceptions import DataSchemaError


def _prices(start: str = "2019-01-01", periods: int = 900) -> pd.Series:
    """Flat $100 closes so price scaling is trivially checkable."""
    index = pd.bdate_range(start, periods=periods, tz="America/New_York", name="date")
    return pd.Series(100.0, index=index)


def _earnings(n: int = 12) -> pd.DataFrame:
    """Announcements ~40 days after each quarter end, alternating beat/miss."""
    period_ends = pd.date_range("2019-03-31", periods=n, freq="QE")
    announcement = period_ends + pd.Timedelta(days=40)
    beat = np.where(np.arange(n) % 2 == 0, 0.10, -0.10)
    return pd.DataFrame(
        {
            "date": announcement,
            "epsActual": 1.0 + beat,
            "epsEstimated": 1.0,
            "revenueActual": 1000.0 + beat * 100,
            "revenueEstimated": 1000.0,
        }
    )


class TestComputeAnnouncementSurprises:
    def test_price_scaled_sue_matches_hand_calculation(self) -> None:
        result = compute_announcement_surprises(_earnings(), _prices())
        # Surprise of +0.10 on a $100 prior close.
        assert abs(result["sue_price_scaled"].iloc[0] - 0.001) < 1e-12
        assert abs(result["sue_price_scaled"].iloc[1] - (-0.001)) < 1e-12

    def test_percent_surprise_relative_to_estimate(self) -> None:
        result = compute_announcement_surprises(_earnings(), _prices())
        assert abs(result["eps_surprise_pct"].iloc[0] - 0.10) < 1e-12

    def test_revenue_surprise(self) -> None:
        result = compute_announcement_surprises(_earnings(), _prices())
        assert abs(result["revenue_surprise_pct"].iloc[0] - 0.01) < 1e-12

    def test_std_scaling_uses_only_past_surprises(self) -> None:
        """The scaling denominator must not include the surprise being scaled."""
        result = compute_announcement_surprises(_earnings(n=12), _prices())
        # Needs MIN_SURPRISE_HISTORY past observations before it can be computed.
        assert result["sue_std_scaled"].iloc[:6].isna().all()
        assert result["sue_std_scaled"].dropna().notna().any()

    def test_scheduled_future_announcements_are_dropped(self) -> None:
        earnings = _earnings(n=6)
        earnings.loc[len(earnings)] = {
            "date": pd.Timestamp("2027-01-30"),
            "epsActual": np.nan,
            "epsEstimated": 2.0,
            "revenueActual": np.nan,
            "revenueEstimated": 2000.0,
        }
        result = compute_announcement_surprises(earnings, _prices())
        assert len(result) == 6

    def test_indexed_by_announcement_date_in_price_timezone(self) -> None:
        result = compute_announcement_surprises(_earnings(), _prices())
        assert str(result.index.tz) == "America/New_York"
        assert list(result.columns) == list(SURPRISE_COLUMNS)

    def test_price_taken_strictly_before_announcement(self) -> None:
        """An announcement can move its own day's close; scaling by it is circular."""
        prices = _prices()
        earnings = _earnings(n=1)
        announcement = pd.Timestamp(earnings["date"].iloc[0], tz="America/New_York")
        # Spike the close ON the announcement day; the result must ignore it.
        trading_day = prices.index[prices.index.searchsorted(announcement, side="left")]
        prices.loc[trading_day] = 1_000_000.0

        result = compute_announcement_surprises(earnings, prices)
        assert abs(result["sue_price_scaled"].iloc[0] - 0.001) < 1e-12

    def test_zero_estimate_gives_nan_not_infinity(self) -> None:
        earnings = _earnings(n=2)
        earnings.loc[0, "epsEstimated"] = 0.0
        result = compute_announcement_surprises(earnings, _prices())
        assert pd.isna(result["eps_surprise_pct"].iloc[0])
        # Price scaling still works — that is why it is the default.
        assert not pd.isna(result["sue_price_scaled"].iloc[0])

    def test_empty_input_returns_typed_empty_frame(self) -> None:
        result = compute_announcement_surprises(pd.DataFrame(), _prices())
        assert result.empty
        assert list(result.columns) == list(SURPRISE_COLUMNS)

    def test_missing_fields_raise(self) -> None:
        with pytest.raises(DataSchemaError):
            compute_announcement_surprises(_earnings().drop(columns=["epsEstimated"]), _prices())


class TestDaysSinceAnnouncement:
    def test_counts_calendar_days_from_the_announcement(self) -> None:
        index = pd.MultiIndex.from_product(
            [
                pd.DatetimeIndex(
                    ["2024-01-10", "2024-01-11", "2024-01-12"], tz="America/New_York", name="date"
                ),
                ["TEST"],
            ],
            names=["date", "symbol"],
        )
        ordinal = float(pd.Timestamp("2024-01-09").value // (10**9 * 86400))
        panel = pd.DataFrame({"announcement_ordinal": [ordinal] * 3}, index=index)

        result = add_days_since_announcement(panel)
        assert list(result["days_since_earnings"]) == [1.0, 2.0, 3.0]
        assert "announcement_ordinal" not in result.columns

    def test_passthrough_when_column_absent(self) -> None:
        panel = pd.DataFrame({"x": [1.0]})
        pd.testing.assert_frame_equal(add_days_since_announcement(panel), panel)
