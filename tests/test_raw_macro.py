"""Tests for the raw FRED macro layer and its derived publication-lagged panel."""

from __future__ import annotations

import pandas as pd
import pytest

from core.data.factors.macro import (
    DEFAULT_FRED_SERIES_MAP,
    MACRO_PUBLICATION_LAGS_DAYS,
    RAW_LONG_COLUMNS,
    apply_macro_publication_lag,
    derive_macro_panel_from_raw,
    fetch_raw_fred_series,
)


def _build_raw_long_fixture() -> pd.DataFrame:
    """Two reference dates per series at native frequency, no lag, no ffill.

    Mirrors the schema produced by ``fetch_raw_fred_series`` so the tests run
    offline without hitting FRED.
    """
    rows = []
    for series_id, monthly_dates in {
        "cpi_yoy": ("2020-01-01", "2020-02-01"),
        "unrate": ("2020-01-01", "2020-02-01"),
        "fed_funds": ("2020-01-01", "2020-02-01"),
        "dgs10": ("2020-01-02", "2020-01-03"),
        "t10y2y": ("2020-01-02", "2020-01-03"),
    }.items():
        for value, date in zip([1.0, 2.0], monthly_dates, strict=True):
            rows.append(
                {"reference_date": pd.Timestamp(date), "series_id": series_id, "value": value}
            )
    return pd.DataFrame(rows)


def test_raw_long_columns_constant_matches_schema() -> None:
    """The exported tuple is the contract every helper relies on."""
    assert RAW_LONG_COLUMNS == ("reference_date", "series_id", "value")


def test_default_fred_series_map_covers_publication_lags() -> None:
    """Every canonical series_id has a documented publication lag."""
    assert set(DEFAULT_FRED_SERIES_MAP) == set(MACRO_PUBLICATION_LAGS_DAYS)


def test_fetch_raw_fred_series_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing FRED_API_KEY raises a clear RuntimeError instead of silently returning."""
    monkeypatch.setattr("core.data.factors.macro.FRED_API_KEY", "")
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        fetch_raw_fred_series({"unrate": "UNRATE"})


def test_derive_macro_panel_pivots_lags_and_ffills() -> None:
    """The full pipeline is equivalent to pivot -> apply_lag -> asfreq B -> ffill."""
    raw_long = _build_raw_long_fixture()

    derived = derive_macro_panel_from_raw(raw_long)

    assert set(derived.columns) == set(DEFAULT_FRED_SERIES_MAP)
    assert derived.index.freqstr == "B"
    # Each column should have non-null values once the lag period elapses.
    cpi_lag = MACRO_PUBLICATION_LAGS_DAYS["cpi_yoy"]
    expected_cpi_first_visible = pd.Timestamp("2020-01-01") + pd.Timedelta(days=cpi_lag)
    assert derived.loc[expected_cpi_first_visible, "cpi_yoy"] == 1.0
    # Forward-fill should keep the value alive between lagged release dates.
    expected_cpi_second_release = pd.Timestamp("2020-02-01") + pd.Timedelta(days=cpi_lag)
    assert derived.loc[expected_cpi_second_release, "cpi_yoy"] == 2.0


def test_derive_macro_panel_matches_manual_apply_lag() -> None:
    """Regression: derive_macro_panel_from_raw is exactly pivot -> lag -> bfill alignment.

    This locks the contract that the legacy ``load_default_macro`` produced.
    """
    raw_long = _build_raw_long_fixture()
    derived = derive_macro_panel_from_raw(raw_long)

    wide = raw_long.pivot(index="reference_date", columns="series_id", values="value").sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    wide.columns.name = None
    manual = apply_macro_publication_lag(wide).asfreq("B").ffill()
    manual.index.name = "date"

    pd.testing.assert_frame_equal(derived, manual, check_freq=False)


def test_derive_macro_panel_handles_empty_input() -> None:
    """Empty raw long input returns an empty DataFrame, not an exception."""
    empty = pd.DataFrame(columns=list(RAW_LONG_COLUMNS))
    assert derive_macro_panel_from_raw(empty).empty
