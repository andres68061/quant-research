"""Tests for making period-end vendor metrics point-in-time via filing dates.

The central property under test is a leakage property: a vendor row describing
the quarter ending 2024-03-31 must not become visible until the filing that made
it public, not on 2024-03-31.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.data.factors.statement_metrics import (
    FALLBACK_PUBLICATION_LAG_DAYS,
    resolve_publication_date,
)
from core.data.factors.vendor_metrics import (
    KEY_METRIC_FIELDS,
    VENDOR_METRIC_COLUMNS,
    attach_publication_dates,
    build_filing_date_lookup,
    build_symbol_vendor_metrics,
)
from core.exceptions import DataSchemaError


def _statements(n: int = 8, lag_days: int = 40) -> pd.DataFrame:
    period_ends = pd.date_range("2022-03-31", periods=n, freq="QE")
    return pd.DataFrame(
        {
            "date": period_ends,
            "acceptedDate": period_ends + pd.Timedelta(days=lag_days),
            "revenue": range(n),
        }
    )


def _key_metrics(n: int = 8) -> pd.DataFrame:
    period_ends = pd.date_range("2022-03-31", periods=n, freq="QE")
    return pd.DataFrame(
        {
            "date": period_ends,
            "returnOnInvestedCapital": [0.10 + 0.01 * i for i in range(n)],
            "incomeQuality": 1.2,
            "cashConversionCycle": 30.0,
        }
    )


class TestResolvePublicationDate:
    def test_plausible_dates_pass_through(self) -> None:
        statements = _statements()
        resolved, imputed = resolve_publication_date(
            statements["acceptedDate"], pd.DatetimeIndex(statements["date"])
        )
        assert not imputed.any()
        assert (resolved == statements["acceptedDate"]).all()

    def test_period_end_placeholder_is_replaced(self) -> None:
        """FMP fills acceptedDate with the period end for pre-EDGAR filings."""
        statements = _statements(lag_days=0)
        resolved, imputed = resolve_publication_date(
            statements["acceptedDate"], pd.DatetimeIndex(statements["date"])
        )
        assert imputed.all()
        expected = statements["date"] + pd.Timedelta(days=FALLBACK_PUBLICATION_LAG_DAYS)
        assert (resolved == expected).all()

    def test_date_before_period_end_is_replaced(self) -> None:
        statements = _statements(lag_days=-5)
        _, imputed = resolve_publication_date(
            statements["acceptedDate"], pd.DatetimeIndex(statements["date"])
        )
        assert imputed.all()

    def test_missing_date_is_replaced(self) -> None:
        statements = _statements()
        statements.loc[0, "acceptedDate"] = pd.NaT
        resolved, imputed = resolve_publication_date(
            statements["acceptedDate"], pd.DatetimeIndex(statements["date"])
        )
        assert imputed.iloc[0]
        assert pd.notna(resolved.iloc[0])

    def test_short_but_genuine_lag_is_kept(self) -> None:
        """A few days is plausible for an 8-K release and must not be overwritten."""
        statements = _statements(lag_days=3)
        _, imputed = resolve_publication_date(
            statements["acceptedDate"], pd.DatetimeIndex(statements["date"])
        )
        assert not imputed.any()


class TestFilingDateLookup:
    def test_maps_period_end_to_acceptance(self) -> None:
        lookup = build_filing_date_lookup(_statements())
        assert lookup.loc[pd.Timestamp("2022-03-31")] == pd.Timestamp("2022-05-10")

    def test_amendment_keeps_the_later_acceptance(self) -> None:
        statements = _statements(n=2)
        amendment = statements.iloc[[0]].copy()
        amendment["acceptedDate"] = amendment["acceptedDate"] + pd.Timedelta(days=90)
        combined = pd.concat([statements, amendment], ignore_index=True)

        lookup = build_filing_date_lookup(combined)
        assert lookup.loc[pd.Timestamp("2022-03-31")] == pd.Timestamp("2022-08-08")

    def test_missing_pit_fields_raise(self) -> None:
        with pytest.raises(DataSchemaError):
            build_filing_date_lookup(_statements().drop(columns=["acceptedDate"]))

    def test_empty_input(self) -> None:
        assert build_filing_date_lookup(pd.DataFrame()).empty


class TestAttachPublicationDates:
    def test_reindexes_onto_publication_not_period_end(self) -> None:
        """THE leakage assertion: nothing is visible on its period-end date."""
        lookup = build_filing_date_lookup(_statements())
        dated = attach_publication_dates(_key_metrics(), lookup, KEY_METRIC_FIELDS)

        period_ends = set(pd.date_range("2022-03-31", periods=8, freq="QE"))
        assert not set(dated.index) & period_ends
        assert (dated.index == pd.DatetimeIndex(sorted(period_ends)) + pd.Timedelta(days=40)).all()

    def test_values_travel_with_their_period(self) -> None:
        lookup = build_filing_date_lookup(_statements())
        dated = attach_publication_dates(_key_metrics(), lookup, KEY_METRIC_FIELDS)
        # Q1 2022's ROIC of 0.10 is visible on the Q1 filing date, not before.
        assert abs(dated["return_on_invested_capital"].iloc[0] - 0.10) < 1e-12

    def test_undatable_periods_are_dropped_not_guessed(self) -> None:
        lookup = build_filing_date_lookup(_statements(n=4))
        dated = attach_publication_dates(_key_metrics(n=8), lookup, KEY_METRIC_FIELDS)
        assert len(dated) == 4

    def test_missing_vendor_column_yields_nan_column(self) -> None:
        lookup = build_filing_date_lookup(_statements())
        vendor = _key_metrics().drop(columns=["incomeQuality"])
        dated = attach_publication_dates(vendor, lookup, KEY_METRIC_FIELDS)
        assert dated["income_quality"].isna().all()
        assert dated["return_on_invested_capital"].notna().any()

    def test_empty_inputs_return_typed_empty_frame(self) -> None:
        dated = attach_publication_dates(
            pd.DataFrame(), pd.Series(dtype="datetime64[ns]"), KEY_METRIC_FIELDS
        )
        assert dated.empty
        assert list(dated.columns) == list(KEY_METRIC_FIELDS)


class TestBuildSymbolVendorMetrics:
    def test_combines_datasets_and_emits_all_columns(self) -> None:
        metrics = build_symbol_vendor_metrics({"key_metrics": _key_metrics()}, _statements())
        assert list(metrics.columns) == list(VENDOR_METRIC_COLUMNS)
        assert metrics["return_on_invested_capital"].notna().any()
        # Columns from datasets we did not pass are present but empty.
        assert metrics["quick_ratio"].isna().all()

    def test_no_statements_means_nothing_datable(self) -> None:
        metrics = build_symbol_vendor_metrics({"key_metrics": _key_metrics()}, pd.DataFrame())
        assert metrics.empty
        assert list(metrics.columns) == list(VENDOR_METRIC_COLUMNS)

    def test_price_based_vendor_ratios_are_not_carried(self) -> None:
        """PE/PB/marketCap are stale by construction; we compute them daily."""
        excluded = {"priceToEarningsRatio", "priceToBookRatio", "marketCap", "dividendYield"}
        assert not excluded & set(VENDOR_METRIC_COLUMNS)
