"""Tests for the point-in-time fundamentals layer (leakage is the enemy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.factors.fundamentals import (
    build_pit_fundamentals_panel,
    compute_fundamental_factors,
    extract_pit_metrics,
)
from core.exceptions import DataSchemaError


def _quarterly_statements(n_quarters: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic statements: Q ends Mar/Jun/Sep/Dec, filed ~40 days later."""
    period_ends = pd.date_range("2020-03-31", periods=n_quarters, freq="QE")
    accepted = period_ends + pd.Timedelta(days=40)
    income = pd.DataFrame(
        {
            "date": period_ends,
            "acceptedDate": accepted,
            "netIncome": np.arange(1, n_quarters + 1) * 100.0,
            "revenue": np.arange(1, n_quarters + 1) * 1000.0,
            "weightedAverageShsOutDil": 500.0,
        }
    )
    balance = pd.DataFrame(
        {
            "date": period_ends,
            "acceptedDate": accepted,
            "totalStockholdersEquity": 5000.0 + np.arange(n_quarters) * 100.0,
            "totalAssets": 10000.0 * (1.05 ** np.arange(n_quarters)),
        }
    )
    return income, balance


class TestExtractPitMetrics:
    def test_publication_dated_and_ttm_needs_four_quarters(self) -> None:
        income, balance = _quarterly_statements()
        metrics = extract_pit_metrics(income, balance)
        # Indexed by publication date (accepted + 0, normalized), not period end.
        assert (metrics.index == metrics["reference_date"] + pd.Timedelta(days=40)).all()
        # First 3 rows lack a full TTM window.
        assert metrics["net_income_ttm"].isna().sum() == 3
        # 4th row TTM = 100+200+300+400.
        assert metrics["net_income_ttm"].dropna().iloc[0] == 1000.0

    def test_asset_growth_is_yoy(self) -> None:
        income, balance = _quarterly_statements()
        metrics = extract_pit_metrics(income, balance)
        expected = 1.05**4 - 1.0
        assert abs(metrics["asset_growth_yoy"].dropna().iloc[0] - expected) < 1e-12

    def test_missing_fields_raise(self) -> None:
        income, balance = _quarterly_statements()
        with pytest.raises(DataSchemaError):
            extract_pit_metrics(income.drop(columns=["netIncome"]), balance)


class TestBuildPitPanelNoLeakage:
    def test_metric_never_visible_before_publication(self) -> None:
        income, balance = _quarterly_statements()
        metrics = extract_pit_metrics(income, balance)
        trading_index = pd.bdate_range("2020-01-01", "2022-12-31", tz="America/New_York")
        panel = build_pit_fundamentals_panel({"TEST": metrics}, trading_index)

        for publication_date, row in metrics.iterrows():
            pub = pd.Timestamp(publication_date).tz_localize("America/New_York")
            visible = panel.xs("TEST", level="symbol")["book_equity"]
            # THE leakage assertion: strictly after publication, never on/before.
            on_or_before = visible.loc[:pub]
            assert not (on_or_before == row["book_equity"]).any() or (
                # value may coincide with an EARLIER filing's value; check first appearance
                visible[visible == row["book_equity"]].index.min()
                > pub
            )

    def test_first_visible_day_is_next_trading_day(self) -> None:
        income, balance = _quarterly_statements(n_quarters=4)
        metrics = extract_pit_metrics(income, balance)
        trading_index = pd.bdate_range("2020-01-01", "2021-12-31", tz="America/New_York")
        panel = build_pit_fundamentals_panel({"TEST": metrics}, trading_index)
        series = panel.xs("TEST", level="symbol")["book_equity"].dropna()

        first_pub = pd.Timestamp(metrics.index[0]).tz_localize("America/New_York")
        expected_first = trading_index[trading_index.searchsorted(first_pub, side="right")]
        assert series.index.min() == expected_first

    def test_staleness_cap_kills_old_filings(self) -> None:
        income, balance = _quarterly_statements(n_quarters=4)
        metrics = extract_pit_metrics(income, balance)
        # Trading calendar extends 3 years past the last filing.
        trading_index = pd.bdate_range("2020-01-01", "2024-12-31", tz="America/New_York")
        panel = build_pit_fundamentals_panel({"TEST": metrics}, trading_index)
        series = panel.xs("TEST", level="symbol")["book_equity"]
        assert pd.isna(series.reindex(trading_index).iloc[-1])


class TestComputeFundamentalFactors:
    def test_factor_values_and_negative_book_handling(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2021-01-04"), "GOOD"), (pd.Timestamp("2021-01-04"), "NEGBOOK")],
            names=["date", "symbol"],
        )
        pit_panel = pd.DataFrame(
            {
                "book_equity": [5000.0, -100.0],
                "net_income_ttm": [1000.0, 50.0],
                "revenue_ttm": [10000.0, 500.0],
                "total_assets": [12000.0, 400.0],
                "asset_growth_yoy": [0.10, 0.90],
                "shares_diluted": [500.0, 10.0],
            },
            index=index,
        )
        market_cap = pd.Series([20000.0, 200.0], index=index)
        factors = compute_fundamental_factors(pit_panel, market_cap)

        good = factors.xs("GOOD", level="symbol").iloc[0]
        assert abs(good["book_to_market"] - 0.25) < 1e-6
        assert abs(good["earnings_yield"] - 0.05) < 1e-6
        assert abs(good["roe"] - 0.20) < 1e-6
        assert abs(good["neg_asset_growth"] - (-0.10)) < 1e-6

        negbook = factors.xs("NEGBOOK", level="symbol").iloc[0]
        assert pd.isna(negbook["book_to_market"])
        assert pd.isna(negbook["roe"])
        assert not pd.isna(negbook["earnings_yield"])
        assert "neg_asset_growth" in factors.columns
