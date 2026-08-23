"""Tests for the raw-statement -> publication-dated metric mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.factors.statement_metrics import (
    STATEMENT_METRIC_COLUMNS,
    TTM_QUARTERS,
    build_statement_metrics,
)
from core.exceptions import DataSchemaError


def _statements(n_quarters: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic quarterly statements filed 40 days after each period end."""
    period_ends = pd.date_range("2019-03-31", periods=n_quarters, freq="QE")
    accepted = period_ends + pd.Timedelta(days=40)
    quarter = np.arange(1, n_quarters + 1, dtype=float)

    income = pd.DataFrame(
        {
            "date": period_ends,
            "acceptedDate": accepted,
            "revenue": quarter * 1000.0,
            "netIncome": quarter * 100.0,
            "grossProfit": quarter * 400.0,
            "operatingIncome": quarter * 200.0,
            "ebit": quarter * 210.0,
            "ebitda": quarter * 260.0,
            "researchAndDevelopmentExpenses": quarter * 50.0,
            "sellingGeneralAndAdministrativeExpenses": quarter * 80.0,
            "interestExpense": quarter * 10.0,
            "costOfRevenue": quarter * 600.0,
            "incomeTaxExpense": quarter * 25.0,
            "weightedAverageShsOutDil": 500.0,
            "epsDiluted": quarter * 0.2,
        }
    )
    balance = pd.DataFrame(
        {
            "date": period_ends,
            "acceptedDate": accepted,
            "totalStockholdersEquity": 5000.0 + np.arange(n_quarters) * 100.0,
            "totalAssets": 10000.0 * (1.05 ** np.arange(n_quarters)),
            "totalLiabilities": 4000.0,
            "totalDebt": 2000.0,
            "longTermDebt": 1500.0,
            "cashAndShortTermInvestments": 800.0,
            "totalCurrentAssets": 3000.0,
            "totalCurrentLiabilities": 1500.0,
            "inventory": 600.0,
            "netReceivables": 900.0,
            "propertyPlantEquipmentNet": 2500.0,
            "retainedEarnings": 3000.0,
            "goodwillAndIntangibleAssets": 700.0,
            "minorityInterest": 0.0,
            "preferredStock": 0.0,
        }
    )
    cash_flow = pd.DataFrame(
        {
            "date": period_ends,
            "acceptedDate": accepted,
            "netCashProvidedByOperatingActivities": quarter * 150.0,
            "capitalExpenditure": -quarter * 40.0,
            "freeCashFlow": quarter * 110.0,
            "netDividendsPaid": -quarter * 20.0,
            "commonStockRepurchased": -quarter * 30.0,
            "commonStockIssuance": quarter * 5.0,
            "stockBasedCompensation": quarter * 12.0,
            "depreciationAndAmortization": quarter * 45.0,
        }
    )
    return income, balance, cash_flow


class TestBuildStatementMetrics:
    def test_indexed_by_publication_not_period_end(self) -> None:
        income, balance, cash_flow = _statements()
        metrics = build_statement_metrics(income, balance, cash_flow)
        assert metrics.index.name == "publication_date"
        assert (metrics.index == metrics["reference_date"] + pd.Timedelta(days=40)).all()

    def test_emits_every_declared_column(self) -> None:
        income, balance, cash_flow = _statements()
        metrics = build_statement_metrics(income, balance, cash_flow)
        assert set(STATEMENT_METRIC_COLUMNS).issubset(metrics.columns)
        assert metrics[list(STATEMENT_METRIC_COLUMNS)].dtypes.eq("float64").all()

    def test_ttm_sums_four_quarters(self) -> None:
        income, balance, cash_flow = _statements()
        metrics = build_statement_metrics(income, balance, cash_flow)
        assert metrics["revenue_ttm"].isna().sum() == TTM_QUARTERS - 1
        # Quarters 1..4 revenue = 1000+2000+3000+4000.
        assert metrics["revenue_ttm"].dropna().iloc[0] == 10000.0
        assert metrics["cfo_ttm"].dropna().iloc[0] == 150.0 * (1 + 2 + 3 + 4)

    def test_lagged_columns_are_prior_year_values(self) -> None:
        income, balance, cash_flow = _statements()
        metrics = build_statement_metrics(income, balance, cash_flow)
        assert metrics["total_assets_lag4"].iloc[8] == metrics["total_assets"].iloc[4]
        assert metrics["total_assets_lag8"].iloc[8] == metrics["total_assets"].iloc[0]
        assert metrics["total_assets_lag4"].iloc[:4].isna().all()

    def test_missing_cash_flow_keeps_other_metrics(self) -> None:
        income, balance, _ = _statements()
        metrics = build_statement_metrics(income, balance, cash_flows=None)
        assert not metrics.empty
        assert metrics["cfo_ttm"].isna().all()
        assert metrics["revenue_ttm"].notna().any()

    def test_missing_vendor_column_yields_nan_not_error(self) -> None:
        income, balance, cash_flow = _statements()
        metrics = build_statement_metrics(
            income.drop(columns=["researchAndDevelopmentExpenses"]), balance, cash_flow
        )
        assert metrics["rd_expense_ttm"].isna().all()
        assert metrics["revenue_ttm"].notna().any()

    def test_missing_pit_dates_raise(self) -> None:
        income, balance, cash_flow = _statements()
        with pytest.raises(DataSchemaError):
            build_statement_metrics(income.drop(columns=["acceptedDate"]), balance, cash_flow)

    def test_empty_input_returns_typed_empty_frame(self) -> None:
        metrics = build_statement_metrics(pd.DataFrame(), pd.DataFrame())
        assert metrics.empty
        assert set(STATEMENT_METRIC_COLUMNS).issubset(metrics.columns)

    def test_publication_date_is_latest_of_the_statements(self) -> None:
        income, balance, cash_flow = _statements(n_quarters=6)
        # Balance sheet lands a week after the income statement.
        balance["acceptedDate"] = balance["acceptedDate"] + pd.Timedelta(days=7)
        metrics = build_statement_metrics(income, balance, cash_flow)
        assert (metrics.index == metrics["reference_date"] + pd.Timedelta(days=47)).all()
