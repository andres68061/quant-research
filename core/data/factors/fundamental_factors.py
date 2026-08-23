"""Published fundamental anomaly factors from the statement metric frame.

Split by what each factor needs:

- :func:`compute_statement_factors` — ratios of statement items only. Computed at
  quarterly publication frequency (cheap), then dailyized once.
- :func:`compute_price_dependent_factors` — anything divided by market
  capitalisation. Must be computed *after* dailyization because market cap moves
  every day while the numerator only moves on filing days.

Every factor is oriented so that **higher = the side the published anomaly says
to go long**, and a ``neg_`` prefix marks a factor whose raw form predicts
negatively (asset growth, accruals, issuance, capex). Both orientations are
emitted so the shared cross-sectional ranker never has to know the sign.

References
----------
- Novy-Marx (2013), gross profitability
- Sloan (1996), accruals
- Cooper, Gulen & Schill (2008), asset growth
- Pontiff & Woodgate (2008), net share issuance
- Hirshleifer, Hou, Teoh & Zhang (2004), net operating assets
- Titman, Wei & Xie (2004), capital investment
- Chan, Lakonishok & Sougiannis (2001), R&D to market
- Boudoukh, Michaely, Richardson & Roberts (2007), net payout yield
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns of :func:`compute_statement_factors` output.
STATEMENT_FACTOR_COLUMNS: tuple[str, ...] = (
    "gross_profitability",
    "operating_profitability",
    "roa",
    "roe",
    "cfo_to_assets",
    "gross_margin",
    "asset_turnover",
    "leverage",
    "debt_to_equity",
    "current_ratio",
    "rd_intensity",
    "accruals",
    "neg_accruals",
    "asset_growth",
    "neg_asset_growth",
    "net_share_issuance",
    "neg_net_share_issuance",
    "net_operating_assets",
    "neg_net_operating_assets",
    "capex_intensity",
    "neg_capex_intensity",
    "inventory_growth",
    "neg_inventory_growth",
    "revenue_growth",
)

# Columns of :func:`compute_price_dependent_factors` output.
PRICE_DEPENDENT_FACTOR_COLUMNS: tuple[str, ...] = (
    "book_to_market",
    "earnings_yield",
    "cash_flow_to_price",
    "sales_to_price",
    "fcf_yield",
    "ebitda_to_ev",
    "rd_to_market",
    "net_payout_yield",
)

# Numerators that must be dailyized so they can be divided by daily market cap.
PRICE_DEPENDENT_INPUT_COLUMNS: tuple[str, ...] = (
    "book_equity",
    "net_income_ttm",
    "cfo_ttm",
    "revenue_ttm",
    "fcf_ttm",
    "ebitda_ttm",
    "ebit_ttm",
    "rd_expense_ttm",
    "buybacks_ttm",
    "dividends_paid_ttm",
    "total_debt",
    "cash_and_st_investments",
    "minority_interest",
    "preferred_stock",
    "total_assets",
    "total_liabilities",
    "total_current_assets",
    "total_current_liabilities",
    "retained_earnings",
)


def _ratio(numerator: pd.Series, denominator: pd.Series, positive_only: bool = True) -> pd.Series:
    """
    Divide two aligned series, returning NaN where the result is meaningless.

    A zero or negative scale denominator (total assets, book equity, revenue) does
    not make the ratio "very large" — it makes it undefined. Masking to NaN is the
    honest answer and keeps a sign flip from masquerading as an extreme rank.

    Args:
        numerator: Numerator series.
        denominator: Denominator series, aligned to ``numerator``.
        positive_only: Mask denominators <= 0 (default). Set False when a negative
            denominator is economically meaningful.

    Returns:
        float64 Series aligned to the inputs.
    """
    denom = denominator.astype("float64")
    denom = denom.where(denom > 0) if positive_only else denom.where(denom != 0)
    return numerator.astype("float64") / denom


def _growth(current: pd.Series, prior: pd.Series) -> pd.Series:
    """Year-over-year growth rate, undefined when the prior level is not positive."""
    return _ratio(current, prior) - 1.0


def compute_statement_factors(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every factor that needs no market price.

    Args:
        metrics: Output of
            :func:`core.data.factors.statement_metrics.build_statement_metrics`,
            or a dailyized panel carrying the same columns. Index is preserved.

    Returns:
        DataFrame indexed like ``metrics`` with :data:`STATEMENT_FACTOR_COLUMNS`.

    Notes:
        Financial-sector companies produce meaningless values for inventory,
        capex and asset-turnover style ratios (a bank has no inventory). Filter
        the universe by sector rather than trying to special-case the formulas.
    """
    total_assets = metrics["total_assets"]
    book_equity = metrics["book_equity"]
    revenue = metrics["revenue_ttm"]

    factors = pd.DataFrame(index=metrics.index)

    factors["gross_profitability"] = _ratio(metrics["gross_profit_ttm"], total_assets)
    factors["operating_profitability"] = _ratio(
        metrics["gross_profit_ttm"] - metrics["sga_expense_ttm"].fillna(0.0), book_equity
    )
    factors["roa"] = _ratio(metrics["net_income_ttm"], total_assets)
    factors["roe"] = _ratio(metrics["net_income_ttm"], book_equity)
    factors["cfo_to_assets"] = _ratio(metrics["cfo_ttm"], total_assets)
    factors["gross_margin"] = _ratio(metrics["gross_profit_ttm"], revenue)
    factors["asset_turnover"] = _ratio(revenue, total_assets)
    factors["leverage"] = _ratio(metrics["total_debt"], total_assets)
    factors["debt_to_equity"] = _ratio(metrics["total_debt"], book_equity)
    factors["current_ratio"] = _ratio(
        metrics["total_current_assets"], metrics["total_current_liabilities"]
    )
    factors["rd_intensity"] = _ratio(metrics["rd_expense_ttm"].fillna(0.0), revenue)

    # Sloan (1996): accruals = (earnings - cash flow) / average total assets.
    average_assets = (total_assets + metrics["total_assets_lag4"]) / 2.0
    factors["accruals"] = _ratio(metrics["net_income_ttm"] - metrics["cfo_ttm"], average_assets)

    factors["asset_growth"] = _growth(total_assets, metrics["total_assets_lag4"])

    # Pontiff & Woodgate (2008): log growth in diluted share count. Log keeps a
    # 2:1 split-driven doubling symmetric with a halving.
    share_ratio = _ratio(metrics["shares_diluted"], metrics["shares_diluted_lag4"])
    factors["net_share_issuance"] = np.log(share_ratio.where(share_ratio > 0))

    # Hirshleifer et al. (2004): operating assets net of operating liabilities,
    # scaled by lagged total assets.
    operating_assets = total_assets - metrics["cash_and_st_investments"].fillna(0.0)
    operating_liabilities = (
        total_assets
        - metrics["total_debt"].fillna(0.0)
        - metrics["minority_interest"].fillna(0.0)
        - metrics["preferred_stock"].fillna(0.0)
        - book_equity
    )
    factors["net_operating_assets"] = _ratio(
        operating_assets - operating_liabilities, metrics["total_assets_lag4"]
    )

    # Vendor reports capex as a negative outflow; express as positive spending.
    factors["capex_intensity"] = _ratio(-metrics["capex_ttm"], total_assets)
    factors["inventory_growth"] = _growth(metrics["inventory"], metrics["inventory_lag4"])
    factors["revenue_growth"] = _growth(revenue, metrics["revenue_ttm_lag4"])

    for factor in ("accruals", "asset_growth", "net_share_issuance", "capex_intensity"):
        factors[f"neg_{factor}"] = -factors[factor]
    factors["neg_net_operating_assets"] = -factors["net_operating_assets"]
    factors["neg_inventory_growth"] = -factors["inventory_growth"]

    return factors[list(STATEMENT_FACTOR_COLUMNS)]


def compute_price_dependent_factors(
    pit_panel: pd.DataFrame,
    market_cap: pd.Series,
) -> pd.DataFrame:
    """
    Compute valuation factors that divide a statement item by market cap.

    Args:
        pit_panel: Dailyized MultiIndex (date, symbol) panel carrying at least
            :data:`PRICE_DEPENDENT_INPUT_COLUMNS`.
        market_cap: MultiIndex (date, symbol) market capitalisation in the same
            currency units as the statements.

    Returns:
        DataFrame indexed like ``pit_panel`` with
        :data:`PRICE_DEPENDENT_FACTOR_COLUMNS`.

    Notes:
        Enterprise value is ``market cap + total debt + minority interest +
        preferred stock - cash and short-term investments``. Firms whose net cash
        exceeds their market cap get a negative EV, which makes ``ebitda_to_ev``
        undefined rather than spuriously attractive.
    """
    cap = market_cap.reindex(pit_panel.index).astype("float64")
    book = pit_panel["book_equity"]

    factors = pd.DataFrame(index=pit_panel.index)
    factors["book_to_market"] = _ratio(book.where(book > 0), cap)
    factors["earnings_yield"] = _ratio(pit_panel["net_income_ttm"], cap)
    factors["cash_flow_to_price"] = _ratio(pit_panel["cfo_ttm"], cap)
    factors["sales_to_price"] = _ratio(pit_panel["revenue_ttm"], cap)
    factors["fcf_yield"] = _ratio(pit_panel["fcf_ttm"], cap)
    factors["rd_to_market"] = _ratio(pit_panel["rd_expense_ttm"].fillna(0.0), cap)

    enterprise_value = (
        cap
        + pit_panel["total_debt"].fillna(0.0)
        + pit_panel["minority_interest"].fillna(0.0)
        + pit_panel["preferred_stock"].fillna(0.0)
        - pit_panel["cash_and_st_investments"].fillna(0.0)
    )
    factors["ebitda_to_ev"] = _ratio(pit_panel["ebitda_ttm"], enterprise_value)

    # Buybacks and dividends are negative outflows; negate to get cash returned.
    net_payout = -(
        pit_panel["buybacks_ttm"].fillna(0.0) + pit_panel["dividends_paid_ttm"].fillna(0.0)
    )
    factors["net_payout_yield"] = _ratio(net_payout, cap)

    return factors[list(PRICE_DEPENDENT_FACTOR_COLUMNS)]
