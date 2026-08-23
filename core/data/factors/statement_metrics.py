"""Map raw FMP statement fields to a publication-dated metric frame (one symbol).

The raw layer stores ~147 vendor columns per symbol across three statements. This
module selects the subset that feeds published factor definitions, converts flow
items to trailing-twelve-month sums, attaches the prior-year value of anything
needed for a growth or Piotroski-style delta, and stamps each row with the
``publication_date`` on which all of it became knowable.

Sign convention: vendor signs are preserved verbatim. FMP reports
``capitalExpenditure``, ``commonStockRepurchased`` and ``netDividendsPaid`` as
negative outflows; factor formulas in
:mod:`core.data.factors.fundamental_factors` handle the sign, not this module.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

TTM_QUARTERS = 4

# --- Vendor field -> metric name -------------------------------------------------

# Flow items: summed over the trailing four quarters.
INCOME_TTM_FIELDS: dict[str, str] = {
    "revenue_ttm": "revenue",
    "net_income_ttm": "netIncome",
    "gross_profit_ttm": "grossProfit",
    "operating_income_ttm": "operatingIncome",
    "ebit_ttm": "ebit",
    "ebitda_ttm": "ebitda",
    "rd_expense_ttm": "researchAndDevelopmentExpenses",
    "sga_expense_ttm": "sellingGeneralAndAdministrativeExpenses",
    "interest_expense_ttm": "interestExpense",
    "cost_of_revenue_ttm": "costOfRevenue",
    "income_tax_ttm": "incomeTaxExpense",
}

CASH_FLOW_TTM_FIELDS: dict[str, str] = {
    "cfo_ttm": "netCashProvidedByOperatingActivities",
    "capex_ttm": "capitalExpenditure",
    "fcf_ttm": "freeCashFlow",
    "dividends_paid_ttm": "netDividendsPaid",
    "buybacks_ttm": "commonStockRepurchased",
    "stock_issuance_ttm": "commonStockIssuance",
    "sbc_ttm": "stockBasedCompensation",
    "depreciation_ttm": "depreciationAndAmortization",
}

# Stock items: point-in-time levels, taken as reported for the quarter.
INCOME_LEVEL_FIELDS: dict[str, str] = {
    "shares_diluted": "weightedAverageShsOutDil",
    "eps_diluted": "epsDiluted",
}

BALANCE_LEVEL_FIELDS: dict[str, str] = {
    "book_equity": "totalStockholdersEquity",
    "total_assets": "totalAssets",
    "total_liabilities": "totalLiabilities",
    "total_debt": "totalDebt",
    "long_term_debt": "longTermDebt",
    "cash_and_st_investments": "cashAndShortTermInvestments",
    "total_current_assets": "totalCurrentAssets",
    "total_current_liabilities": "totalCurrentLiabilities",
    "inventory": "inventory",
    "net_receivables": "netReceivables",
    "ppe_net": "propertyPlantEquipmentNet",
    "retained_earnings": "retainedEarnings",
    "goodwill_and_intangibles": "goodwillAndIntangibleAssets",
    "minority_interest": "minorityInterest",
    "preferred_stock": "preferredStock",
}

# Metrics whose prior-year value is needed for a growth rate or Piotroski delta.
# Emitted as ``{metric}_lag4``.
LAGGED_METRICS: tuple[str, ...] = (
    "total_assets",
    "book_equity",
    "shares_diluted",
    "inventory",
    "ppe_net",
    "net_receivables",
    "long_term_debt",
    "total_current_assets",
    "total_current_liabilities",
    "revenue_ttm",
    "net_income_ttm",
    "gross_profit_ttm",
    "cfo_ttm",
)

# Piotroski scales ROA and asset turnover by *beginning-of-year* total assets, so
# scoring the prior year needs the level two years back. Emitted as ``*_lag8``.
LAGGED_METRICS_TWO_YEAR: tuple[str, ...] = ("total_assets",)

STATEMENT_METRIC_COLUMNS: tuple[str, ...] = (
    *INCOME_TTM_FIELDS,
    *INCOME_LEVEL_FIELDS,
    *BALANCE_LEVEL_FIELDS,
    *CASH_FLOW_TTM_FIELDS,
    *(f"{metric}_lag4" for metric in LAGGED_METRICS),
    *(f"{metric}_lag8" for metric in LAGGED_METRICS_TWO_YEAR),
    "publication_date_imputed",
)

_PIT_DATE_FIELDS = {"date", "acceptedDate"}

# Fallback lag used when the vendor's acceptedDate is not a real filing date.
# FMP fills acceptedDate with the period end for filings that predate EDGAR
# electronic acceptance: 100% of 1980s rows, ~50% of 1990s, ~3% of 2020s
# (21.5% overall). Taking those at face value would make a quarter's results
# knowable on the day the quarter closed — a ~35 day lookahead concentrated in
# the early history, exactly where a backtest has the least out-of-sample data
# to catch it.
#
# 45 days is the SEC's 10-Q deadline for large accelerated filers and sits above
# the observed median real lag (32-39 days), so it errs late. Erring late costs
# a few days of signal; erring early manufactures alpha.
FALLBACK_PUBLICATION_LAG_DAYS = 45


def _select(
    statements: pd.DataFrame,
    ttm_fields: dict[str, str],
    level_fields: dict[str, str],
    label: str,
) -> pd.DataFrame:
    """
    Select and reshape one statement type into metric columns indexed by ``date``.

    Missing vendor columns yield an all-NaN metric column rather than an error:
    FMP's older filings and financial-sector companies legitimately omit fields
    (e.g. ``inventory`` for a bank).

    Args:
        statements: Raw quarterly statement rows for one symbol.
        ttm_fields: ``{metric_name: vendor_field}`` summed over 4 quarters.
        level_fields: ``{metric_name: vendor_field}`` taken as reported.
        label: Statement name, for error messages.

    Returns:
        DataFrame indexed by ``date`` (reference_date) with an ``acceptedDate``
        column plus one column per requested metric.

    Raises:
        DataSchemaError: If the point-in-time date fields are missing.
    """
    if statements.empty:
        return pd.DataFrame()
    if missing := _PIT_DATE_FIELDS - set(statements.columns):
        raise DataSchemaError(f"{label} missing point-in-time fields: {missing}")

    ordered = statements.sort_values("date").copy()
    selected = pd.DataFrame(index=ordered["date"])
    selected["acceptedDate"] = ordered["acceptedDate"].to_numpy()

    for metric, vendor_field in ttm_fields.items():
        if vendor_field not in ordered.columns:
            selected[metric] = np.nan
            continue
        values = pd.to_numeric(ordered[vendor_field], errors="coerce")
        selected[metric] = values.rolling(TTM_QUARTERS).sum().to_numpy()

    for metric, vendor_field in level_fields.items():
        if vendor_field not in ordered.columns:
            selected[metric] = np.nan
            continue
        selected[metric] = pd.to_numeric(ordered[vendor_field], errors="coerce").to_numpy()

    return selected[~selected.index.duplicated(keep="last")]


def resolve_publication_date(
    vendor_accepted: pd.Series,
    reference_date: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    """
    Replace implausible vendor acceptance dates with a conservative filing lag.

    A filing cannot be accepted on or before the end of the period it reports —
    that is an unambiguous bright line, and the only case treated as implausible
    here. A merely short lag (an 8-K earnings release days after quarter end) is
    genuine and left alone.

    Args:
        vendor_accepted: Vendor ``acceptedDate`` per row, normalized.
        reference_date: Matching fiscal period ends.

    Returns:
        ``(publication_date, imputed)`` where ``imputed`` is a boolean Series
        marking rows whose date was substituted. Callers should carry that flag
        so results can be recomputed excluding imputed history.
    """
    reference = pd.Series(pd.DatetimeIndex(reference_date), index=vendor_accepted.index)
    implausible = vendor_accepted.isna() | (vendor_accepted <= reference)
    fallback = reference + pd.Timedelta(days=FALLBACK_PUBLICATION_LAG_DAYS)
    return vendor_accepted.where(~implausible, fallback), implausible


def build_statement_metrics(
    income_statements: pd.DataFrame,
    balance_sheets: pd.DataFrame,
    cash_flows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build one symbol's publication-dated metric rows from its raw statements.

    Income and balance sheet are joined on ``reference_date`` (fiscal period end)
    with an inner join — a metric is not knowable until both statements exist.
    Cash flow is left-joined so a symbol with no cash-flow coverage still gets its
    income/balance metrics rather than dropping out entirely.

    ``publication_date`` is the **latest** ``acceptedDate`` across the statements
    that contributed to the row, normalized to midnight. In practice all three
    arrive in the same 10-Q on the same day; taking the max is the conservative
    choice when they do not.

    Args:
        income_statements: Raw quarterly income statements for one symbol.
        balance_sheets: Raw quarterly balance sheets for one symbol.
        cash_flows: Raw quarterly cash-flow statements for one symbol; optional.

    Returns:
        DataFrame indexed by ``publication_date`` (ascending, unique) with
        :data:`STATEMENT_METRIC_COLUMNS` plus ``reference_date``. TTM columns are
        NaN until four consecutive quarters exist; ``*_lag4`` columns are NaN for
        the first four rows.

    Raises:
        DataSchemaError: If a non-empty statement lacks ``date``/``acceptedDate``.

    Example:
        >>> metrics = build_statement_metrics(income, balance, cash_flow)  # doctest: +SKIP
        >>> metrics.index.name
        'publication_date'
    """
    empty = pd.DataFrame(columns=[*STATEMENT_METRIC_COLUMNS, "reference_date"])
    empty.index.name = "publication_date"
    if income_statements.empty or balance_sheets.empty:
        return empty

    income = _select(income_statements, INCOME_TTM_FIELDS, INCOME_LEVEL_FIELDS, "income statement")
    balance = _select(balance_sheets, {}, BALANCE_LEVEL_FIELDS, "balance sheet")
    merged = income.join(balance, how="inner", rsuffix="_bal")

    if cash_flows is not None and not cash_flows.empty:
        cash_flow = _select(cash_flows, CASH_FLOW_TTM_FIELDS, {}, "cash flow statement")
        merged = merged.join(cash_flow, how="left", rsuffix="_cf")
    for metric in CASH_FLOW_TTM_FIELDS:
        if metric not in merged.columns:
            merged[metric] = np.nan

    if merged.empty:
        return empty

    accepted_columns = [c for c in merged.columns if c.startswith("acceptedDate")]
    vendor_accepted = merged[accepted_columns].max(axis=1).dt.normalize()
    publication_date, imputed = resolve_publication_date(vendor_accepted, merged.index)
    merged["publication_date_imputed"] = imputed.to_numpy().astype("float64")

    for metric in LAGGED_METRICS:
        merged[f"{metric}_lag4"] = merged[metric].shift(TTM_QUARTERS)
    for metric in LAGGED_METRICS_TWO_YEAR:
        merged[f"{metric}_lag8"] = merged[metric].shift(2 * TTM_QUARTERS)

    merged["reference_date"] = merged.index
    result = (
        merged.assign(publication_date=publication_date.to_numpy())
        .set_index("publication_date")[[*STATEMENT_METRIC_COLUMNS, "reference_date"]]
        .astype({metric: "float64" for metric in STATEMENT_METRIC_COLUMNS})
        .sort_index()
    )
    # Amendments filed the same day as an original: keep the later fiscal period.
    return result[~result.index.duplicated(keep="last")]
