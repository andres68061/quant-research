"""Tests for core.research.eda - monitoring statistics for series and curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.research import eda


def _daily(n: int = 600, seed: int = 0, start: str = "2020-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx, name="x")


# ---------------------------------------------------------------- transforms
def test_apply_transform_pct_change_matches_pandas() -> None:
    s = _daily(50)
    out = eda.apply_transform(s, "pct_change")
    pd.testing.assert_series_equal(out, s.pct_change(fill_method=None).dropna())


def test_apply_transform_drops_nans_before_differencing() -> None:
    s = pd.Series([1.0, np.nan, 3.0], index=pd.bdate_range("2024-01-01", periods=3))
    out = eda.apply_transform(s, "diff")
    assert out.tolist() == [2.0]  # not [nan, nan]


def test_apply_transform_empty_and_unknown() -> None:
    assert eda.apply_transform(pd.Series(dtype="float64"), "level").empty
    with pytest.raises(ValueError):
        eda.apply_transform(_daily(10), "bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------- distribution
def test_summarize_distribution_moments_and_percentile() -> None:
    s = _daily(500)
    r = eda.apply_transform(s, "log_return")
    d = eda.summarize_distribution(r)
    assert d.n == len(r)
    assert d.mean == pytest.approx(float(r.mean()))
    assert d.std == pytest.approx(float(r.std()))
    assert d.median == pytest.approx(float(r.median()))
    assert d.last_value == pytest.approx(float(r.iloc[-1]))
    assert 0.0 <= d.percentile_of_last <= 100.0
    assert d.percentile_of_last == pytest.approx(float((r <= r.iloc[-1]).mean() * 100))
    assert d.adf_pvalue is not None and d.adf_pvalue < 0.05  # returns are stationary
    assert isinstance(d.to_dict(), dict)


def test_summarize_distribution_too_short_returns_nones() -> None:
    d = eda.summarize_distribution(_daily(5))
    assert d.n == 5 and d.mean is None and d.adf_pvalue is None


def test_summarize_distribution_constant_series_has_no_adf() -> None:
    s = pd.Series(np.ones(100), index=pd.bdate_range("2020-01-01", periods=100))
    d = eda.summarize_distribution(s)
    assert d.std == 0.0 and d.adf_pvalue is None and d.zscore_of_last == 0.0


def test_histogram_marks_the_bin_of_the_last_value() -> None:
    r = eda.apply_transform(_daily(300), "pct_change")
    h = eda.histogram(r, bins=20, mark=float(r.iloc[-1]))
    assert len(h["edges"]) == len(h["counts"]) + 1 == 21
    assert sum(h["counts"]) == len(r)
    b = h["mark_bin"]
    assert h["edges"][b] <= r.iloc[-1] <= h["edges"][b + 1]


def test_histogram_mark_outside_range_is_clamped_and_empty_is_empty() -> None:
    r = eda.apply_transform(_daily(100), "pct_change")
    assert eda.histogram(r, bins=10, mark=1e9)["mark_bin"] == 9
    assert eda.histogram(r, bins=10, mark=-1e9)["mark_bin"] == 0
    assert eda.histogram(pd.Series(dtype="float64"))["counts"] == []


def test_rolling_profile_band_is_symmetric_and_nan_until_window() -> None:
    s = _daily(400)
    p = eda.rolling_profile(s, window=100, n_std=2.0)
    assert list(p.columns) == ["level", "rolling_mean", "upper", "lower", "zscore"]
    assert p["rolling_mean"].iloc[:49].isna().all()
    ok = p.dropna()
    np.testing.assert_allclose(ok["upper"] - ok["rolling_mean"], ok["rolling_mean"] - ok["lower"])


# ---------------------------------------------------------------- seasonality
def test_seasonality_table_shape_and_month_stats() -> None:
    s = _daily(3 * 252, start="2021-01-04")
    t = eda.seasonality_table(s, "pct_change")
    assert len(t["years"]) == len(t["matrix"]) >= 3
    assert all(len(row) == 12 for row in t["matrix"])
    assert len(t["month_stats"]) == 12
    stat = t["month_stats"][5]  # June
    assert stat["month"] == 6 and stat["n"] >= 2 and 0.0 <= stat["hit_rate"] <= 1.0


def test_seasonality_table_level_has_no_hit_rate() -> None:
    t = eda.seasonality_table(_daily(300), "level")
    assert all(m["hit_rate"] is None for m in t["month_stats"])


def test_seasonality_table_missing_month_is_none_not_inherited() -> None:
    # Two observations a year apart: every month in between must be None.
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2022-01-31", "2023-01-31"]))
    t = eda.seasonality_table(s, "level")
    assert t["years"] == [2022, 2023]
    assert t["matrix"][0][1:] == [None] * 11


def test_seasonality_table_empty() -> None:
    assert eda.seasonality_table(pd.Series(dtype="float64"))["years"] == []


def test_annual_paths_rebases_each_year_to_100() -> None:
    s = _daily(3 * 252, start="2021-01-04")
    a = eda.annual_paths(s, normalize=True)
    for path in a["paths"].values():
        assert path[0]["value"] == pytest.approx(100.0)
        assert all(1 <= p["doy"] <= 366 for p in path)
    assert a["years"] == sorted(a["years"])


def test_annual_paths_max_years_keeps_most_recent() -> None:
    s = _daily(6 * 252, start="2018-01-02")
    a = eda.annual_paths(s, normalize=False, max_years=2)
    assert len(a["years"]) == 2 and a["years"][-1] == int(s.index.max().year)


# ---------------------------------------------------------------- curves
def _curve() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=10)
    tenors = {"dgs3mo": 5.0, "dgs2": 4.5, "dgs5": 4.2, "dgs10": 4.0, "dgs30": 4.3}
    return pd.DataFrame(
        {k: np.full(10, v) + np.arange(10) * 0.01 for k, v in tenors.items()}, index=idx
    )


TENORS = {"dgs3mo": 0.25, "dgs2": 2.0, "dgs5": 5.0, "dgs10": 10.0, "dgs30": 30.0}


def test_yield_curve_snapshots_uses_nearest_prior_date_never_forward() -> None:
    panel = _curve()
    # Saturday: must resolve to Friday 2024-01-05, not Monday 2024-01-08.
    snaps = eda.yield_curve_snapshots(panel, TENORS, [pd.Timestamp("2024-01-06")])
    assert len(snaps) == 1 and snaps[0]["date"] == "2024-01-05"
    assert [p["id"] for p in snaps[0]["points"]] == list(TENORS)
    assert snaps[0]["points"][-1]["yield"] == pytest.approx(panel.loc["2024-01-05", "dgs30"])


def test_yield_curve_snapshots_skips_dates_before_history() -> None:
    assert eda.yield_curve_snapshots(_curve(), TENORS, [pd.Timestamp("2000-01-01")]) == []


def test_yield_curve_snapshots_missing_tenor_is_none() -> None:
    panel = _curve()
    panel.loc["2024-01-05", "dgs30"] = np.nan
    snap = eda.yield_curve_snapshots(panel, TENORS, [pd.Timestamp("2024-01-05")])[0]
    assert snap["points"][-1]["yield"] is None


def test_curve_shape_history_definitions() -> None:
    h = eda.curve_shape_history(_curve())
    c = _curve()
    pd.testing.assert_series_equal(h["spread_2s10s"], c["dgs10"] - c["dgs2"], check_names=False)
    pd.testing.assert_series_equal(
        h["butterfly_2_5_10"], 2 * c["dgs5"] - c["dgs2"] - c["dgs10"], check_names=False
    )
    assert eda.curve_shape_history(pd.DataFrame()).empty


# ---------------------------------------------------------------- staleness
def test_staleness_thresholds() -> None:
    s = pd.Series([1.0], index=pd.to_datetime(["2026-07-10"]))
    fresh = eda.staleness(s, "x", pd.Timestamp("2026-07-13"), expected_max_gap_days=5)
    late = eda.staleness(s, "x", pd.Timestamp("2026-07-18"), expected_max_gap_days=5)
    stale = eda.staleness(s, "x", pd.Timestamp("2026-09-09"), expected_max_gap_days=5)
    assert (fresh.status, late.status, stale.status) == ("fresh", "late", "stale")
    assert stale.days_since_last == 61 and stale.last_date == "2026-07-10"


def test_staleness_ignores_trailing_nans_and_handles_tz() -> None:
    idx = pd.to_datetime(["2026-07-09", "2026-07-10"]).tz_localize("America/New_York")
    s = pd.Series([1.0, np.nan], index=idx)
    r = eda.staleness(s, "x", pd.Timestamp("2026-07-10"), 5)
    assert r.last_date == "2026-07-09" and r.status == "fresh"


def test_staleness_empty() -> None:
    r = eda.staleness(pd.Series(dtype="float64"), "x", pd.Timestamp("2026-01-01"), 5)
    assert r.status == "empty" and r.last_date is None
