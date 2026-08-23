"""Tests for the published fundamental anomaly factors and accounting scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.factors.fundamental_factors import (
    PRICE_DEPENDENT_FACTOR_COLUMNS,
    STATEMENT_FACTOR_COLUMNS,
    compute_price_dependent_factors,
    compute_statement_factors,
)
from core.data.factors.quality_scores import (
    PIOTROSKI_SIGNAL_COLUMNS,
    compute_altman_z_score,
    compute_piotroski_f_score,
    compute_piotroski_signals,
)


def _metrics(**overrides: float) -> pd.DataFrame:
    """One row of statement metrics with sane defaults; override to probe a factor."""
    base = {
        "revenue_ttm": 10_000.0,
        "net_income_ttm": 1_000.0,
        "gross_profit_ttm": 4_000.0,
        "operating_income_ttm": 2_000.0,
        "ebit_ttm": 2_100.0,
        "ebitda_ttm": 2_600.0,
        "rd_expense_ttm": 500.0,
        "sga_expense_ttm": 800.0,
        "interest_expense_ttm": 100.0,
        "cost_of_revenue_ttm": 6_000.0,
        "income_tax_ttm": 250.0,
        "shares_diluted": 500.0,
        "eps_diluted": 2.0,
        "book_equity": 5_000.0,
        "total_assets": 20_000.0,
        "total_liabilities": 15_000.0,
        "total_debt": 6_000.0,
        "long_term_debt": 4_000.0,
        "cash_and_st_investments": 2_000.0,
        "total_current_assets": 8_000.0,
        "total_current_liabilities": 4_000.0,
        "inventory": 1_000.0,
        "net_receivables": 1_500.0,
        "ppe_net": 7_000.0,
        "retained_earnings": 3_000.0,
        "goodwill_and_intangibles": 1_200.0,
        "minority_interest": 0.0,
        "preferred_stock": 0.0,
        "cfo_ttm": 1_500.0,
        "capex_ttm": -400.0,
        "fcf_ttm": 1_100.0,
        "dividends_paid_ttm": -200.0,
        "buybacks_ttm": -300.0,
        "stock_issuance_ttm": 50.0,
        "sbc_ttm": 120.0,
        "depreciation_ttm": 450.0,
        "total_assets_lag4": 16_000.0,
        "book_equity_lag4": 4_500.0,
        "shares_diluted_lag4": 520.0,
        "inventory_lag4": 800.0,
        "ppe_net_lag4": 6_000.0,
        "net_receivables_lag4": 1_300.0,
        "long_term_debt_lag4": 4_400.0,
        "total_current_assets_lag4": 7_000.0,
        "total_current_liabilities_lag4": 4_000.0,
        "revenue_ttm_lag4": 8_000.0,
        "net_income_ttm_lag4": 700.0,
        "gross_profit_ttm_lag4": 3_000.0,
        "cfo_ttm_lag4": 1_200.0,
        "total_assets_lag8": 13_000.0,
    }
    base.update(overrides)
    return pd.DataFrame([base], index=pd.DatetimeIndex(["2024-05-10"], name="publication_date"))


class TestStatementFactors:
    def test_known_values(self) -> None:
        factors = compute_statement_factors(_metrics()).iloc[0]
        assert factors["gross_profitability"] == 4_000.0 / 20_000.0
        assert factors["roa"] == 1_000.0 / 20_000.0
        assert factors["roe"] == 1_000.0 / 5_000.0
        assert factors["gross_margin"] == 4_000.0 / 10_000.0
        assert factors["leverage"] == 6_000.0 / 20_000.0
        assert factors["current_ratio"] == 8_000.0 / 4_000.0
        # Sloan: (earnings - cash flow) / average assets.
        assert factors["accruals"] == (1_000.0 - 1_500.0) / ((20_000.0 + 16_000.0) / 2)
        assert factors["asset_growth"] == 20_000.0 / 16_000.0 - 1.0
        # Capex is a negative vendor outflow; intensity is positive spending.
        assert factors["capex_intensity"] == 400.0 / 20_000.0

    def test_net_share_issuance_is_log_and_signed(self) -> None:
        factors = compute_statement_factors(_metrics()).iloc[0]
        assert factors["net_share_issuance"] == np.log(500.0 / 520.0)
        # Buyback (share count fell) is a positive signal after negation.
        assert factors["neg_net_share_issuance"] > 0

    def test_negative_orientations_mirror_raw_factors(self) -> None:
        factors = compute_statement_factors(_metrics()).iloc[0]
        for name in ("accruals", "asset_growth", "capex_intensity", "inventory_growth"):
            assert factors[f"neg_{name}"] == -factors[name]

    def test_nonpositive_denominators_give_nan_not_extremes(self) -> None:
        factors = compute_statement_factors(_metrics(book_equity=-100.0, total_assets=0.0)).iloc[0]
        assert pd.isna(factors["roe"])
        assert pd.isna(factors["roa"])
        assert pd.isna(factors["gross_profitability"])

    def test_emits_every_declared_column(self) -> None:
        factors = compute_statement_factors(_metrics())
        assert list(factors.columns) == list(STATEMENT_FACTOR_COLUMNS)


class TestPriceDependentFactors:
    def _panel(self) -> tuple[pd.DataFrame, pd.Series]:
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-05-10"), "TEST")], names=["date", "symbol"]
        )
        panel = _metrics().set_index(index)
        return panel, pd.Series([20_000.0], index=index)

    def test_known_values(self) -> None:
        panel, market_cap = self._panel()
        factors = compute_price_dependent_factors(panel, market_cap).iloc[0]
        assert factors["book_to_market"] == 5_000.0 / 20_000.0
        assert factors["earnings_yield"] == 1_000.0 / 20_000.0
        assert factors["sales_to_price"] == 10_000.0 / 20_000.0
        # EV = cap + debt + minority + preferred - cash.
        assert factors["ebitda_to_ev"] == 2_600.0 / (20_000.0 + 6_000.0 - 2_000.0)
        # Buybacks and dividends are negative outflows; payout is what came back.
        assert factors["net_payout_yield"] == 500.0 / 20_000.0

    def test_negative_book_equity_is_nan(self) -> None:
        panel, market_cap = self._panel()
        panel = panel.assign(book_equity=-100.0)
        factors = compute_price_dependent_factors(panel, market_cap).iloc[0]
        assert pd.isna(factors["book_to_market"])
        assert not pd.isna(factors["earnings_yield"])

    def test_net_cash_exceeding_cap_makes_ev_multiple_nan(self) -> None:
        panel, market_cap = self._panel()
        panel = panel.assign(cash_and_st_investments=40_000.0)
        factors = compute_price_dependent_factors(panel, market_cap).iloc[0]
        assert pd.isna(factors["ebitda_to_ev"])

    def test_emits_every_declared_column(self) -> None:
        panel, market_cap = self._panel()
        factors = compute_price_dependent_factors(panel, market_cap)
        assert list(factors.columns) == list(PRICE_DEPENDENT_FACTOR_COLUMNS)


class TestPiotroski:
    def test_all_nine_signals_pass_for_a_healthy_firm(self) -> None:
        signals = compute_piotroski_signals(_metrics()).iloc[0]
        assert list(signals.index) == list(PIOTROSKI_SIGNAL_COLUMNS)
        assert signals.sum() == 9.0
        assert compute_piotroski_f_score(_metrics()).iloc[0] == 9.0

    def test_roa_scaled_by_beginning_of_year_assets(self) -> None:
        # Contemporaneous scaling would call this shrinking ROA (1000/20000 <
        # 700/16000 is false, but 1000/16000 > 700/13000 is true).
        signals = compute_piotroski_signals(_metrics()).iloc[0]
        assert signals["f_roa_improving"] == 1.0

    def test_failing_tests_score_zero_not_nan(self) -> None:
        signals = compute_piotroski_signals(_metrics(net_income_ttm=-500.0)).iloc[0]
        assert signals["f_roa_positive"] == 0.0

    def test_dilution_penalised(self) -> None:
        signals = compute_piotroski_signals(_metrics(shares_diluted=600.0)).iloc[0]
        assert signals["f_no_dilution"] == 0.0

    def test_missing_input_is_nan_not_a_failed_test(self) -> None:
        signals = compute_piotroski_signals(_metrics(total_assets_lag8=np.nan)).iloc[0]
        assert pd.isna(signals["f_roa_improving"])
        assert signals["f_roa_positive"] == 1.0

    def test_score_withheld_when_too_few_tests_evaluable(self) -> None:
        sparse = _metrics(
            total_assets_lag4=np.nan,
            total_assets_lag8=np.nan,
            revenue_ttm_lag4=np.nan,
            gross_profit_ttm_lag4=np.nan,
            shares_diluted_lag4=np.nan,
            total_current_assets_lag4=np.nan,
        )
        assert pd.isna(compute_piotroski_f_score(sparse).iloc[0])


class TestAltmanZ:
    def test_matches_the_published_formula(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-05-10"), "TEST")], names=["date", "symbol"]
        )
        panel = _metrics().set_index(index)
        market_cap = pd.Series([20_000.0], index=index)

        expected = (
            1.2 * ((8_000.0 - 4_000.0) / 20_000.0)
            + 1.4 * (3_000.0 / 20_000.0)
            + 3.3 * (2_100.0 / 20_000.0)
            + 0.6 * (20_000.0 / 15_000.0)
            + 1.0 * (10_000.0 / 20_000.0)
        )
        assert abs(compute_altman_z_score(panel, market_cap).iloc[0] - expected) < 1e-9

    def test_zero_assets_gives_nan(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-05-10"), "TEST")], names=["date", "symbol"]
        )
        panel = _metrics(total_assets=0.0).set_index(index)
        market_cap = pd.Series([20_000.0], index=index)
        assert pd.isna(compute_altman_z_score(panel, market_cap).iloc[0])
