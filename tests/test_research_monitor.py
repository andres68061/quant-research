"""Tests for core.research.monitor - the data-monitor catalog and reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.factors.macro_catalog import FRED_SERIES_CATALOG
from core.data.vendors.commodities import COMMODITIES_CONFIG
from core.research import monitor


def test_catalog_covers_every_commodity_and_fred_series_once() -> None:
    cat = monitor.monitored_series_catalog()
    ids = [s.id for s in cat]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(COMMODITIES_CONFIG) | set(FRED_SERIES_CATALOG)
    groups = [s.group for s in cat]
    order = {g: i for i, g in enumerate(monitor.GROUP_ORDER)}
    assert groups == sorted(groups, key=lambda g: order[g])


def test_catalog_defaults_by_kind() -> None:
    by_id = {s.id: s for s in monitor.monitored_series_catalog()}
    assert by_id["GLD"].default_transform == "log_return"
    assert by_id["dgs10"].default_transform == "diff"
    assert by_id["unrate"].default_transform == "level"
    assert by_id["wti"].default_transform == "log_return"
    # Monthly series must tolerate a full cycle beyond the lag before 'late'.
    assert by_id["cpi_yoy"].expected_max_gap_days > FRED_SERIES_CATALOG["cpi_yoy"].lag_days + 30
    assert by_id["GLD"].expected_max_gap_days <= 7
    assert monitor.find_series("nope") is None


def test_pivot_raw_macro_native_frequency_not_ffilled() -> None:
    raw = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-01-02", "2024-01-03"]
            ),
            "series_id": ["unrate", "unrate", "dgs10", "dgs10"],
            "value": [3.7, 3.9, 4.0, 4.1],
        }
    )
    wide = monitor.pivot_raw_macro(raw)
    assert wide.index.name == "reference_date"
    assert wide["unrate"].dropna().tolist() == [3.7, 3.9]  # two points, not ~40
    assert monitor.pivot_raw_macro(pd.DataFrame()).empty


def _price_series(n: int = 800) -> pd.Series:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=idx, name="GLD")


def test_series_report_sections_and_downsampling() -> None:
    spec = monitor.find_series("GLD")
    assert spec is not None
    values = _price_series()
    rep = monitor.series_report(values, spec, as_of=values.index.max(), max_points=200)
    for key in (
        "series",
        "transform",
        "summary",
        "histogram",
        "levels",
        "seasonality",
        "annual_paths",
        "staleness",
        "methodology",
    ):
        assert key in rep
    assert rep["transform"] == "log_return"
    assert rep["summary"]["n"] == len(values) - 1
    assert 200 <= len(rep["levels"]) <= 202  # thinned, last point kept
    assert rep["levels"][-1]["date"] == str(values.index.max().date())
    assert rep["staleness"]["status"] == "fresh"
    assert rep["annual_paths"]["normalized"] is True
    assert rep["histogram"]["mark"] == rep["summary"]["last_value"]


def test_series_report_transform_override_and_stale() -> None:
    spec = monitor.find_series("dgs10")
    assert spec is not None
    values = pd.Series(
        [4.0, 4.1, 4.05], index=pd.bdate_range("2026-05-01", periods=3), name="dgs10"
    )
    rep = monitor.series_report(values, spec, as_of=pd.Timestamp("2026-09-10"), transform="level")
    assert rep["transform"] == "level"
    assert rep["staleness"]["status"] == "stale"
    assert rep["summary"]["mean"] is None  # too short for moments, no crash


def test_staleness_board_orders_worst_first_and_reports_missing() -> None:
    as_of = pd.Timestamp("2026-09-10")
    fmp = pd.DataFrame(
        {"GLD": [1.0, 2.0], "WTI": [1.0, np.nan]},
        index=pd.to_datetime(["2026-07-09", "2026-09-09"]),
    )
    fred = pd.DataFrame({"dgs10": [4.0]}, index=pd.to_datetime(["2026-09-08"]))
    board = monitor.staleness_board({"fmp": fmp, "fred": fred}, as_of)
    by_id = {r["series_id"]: r for r in board}
    assert by_id["GLD"]["status"] == "fresh"
    assert by_id["WTI"]["status"] == "stale" and by_id["WTI"]["last_date"] == "2026-07-09"
    assert by_id["dgs10"]["status"] == "fresh"
    assert by_id["unrate"]["status"] == "empty"  # not in the panel provided
    statuses = [r["status"] for r in board]
    assert statuses.index("stale") < statuses.index("empty") < statuses.index("fresh")


def test_curve_report_from_raw_pivot() -> None:
    idx = pd.bdate_range("2026-08-01", periods=20)
    wide = pd.DataFrame(
        {"dgs3mo": 3.9, "dgs2": 4.3, "dgs5": 4.5, "dgs10": 4.8, "dgs30": 5.2, "unrate": np.nan},
        index=idx,
    )
    rep = monitor.curve_report(wide, [idx[-1], idx[0], pd.Timestamp("1990-01-01")])
    assert [t["id"] for t in rep["tenors"]] == ["dgs3mo", "dgs2", "dgs5", "dgs10", "dgs30"]
    assert len(rep["snapshots"]) == 2  # 1990 skipped
    assert rep["shape_history"][-1]["spread_2s10s"] == 0.5
    assert "spreads" in rep["methodology"]
