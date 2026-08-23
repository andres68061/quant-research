"""Make vendor-computed metrics point-in-time by joining their filing dates.

``key-metrics``, ``ratios``, ``financial-growth`` and ``enterprise-values`` return
rows stamped with the **fiscal period end**, not the date the numbers became
public. A row dated 2024-03-31 was not knowable until the 10-Q was filed, typically
30-45 days later. Using that date directly leaks roughly six weeks of future
information into a backtest, and — because leaked fundamentals make results look
*better* — nothing about the output would prompt a second look. See ADR 0010.

The fix is a join, not a fudge: the raw statements already carry the real
``filingDate``/``acceptedDate`` for the same ``(symbol, period_end)``, so this
module attaches them and re-indexes on publication date.

What is deliberately NOT carried through
----------------------------------------
Vendor ratios that embed a price — ``priceToEarningsRatio``, ``priceToBookRatio``,
``marketCap``, ``enterpriseValue``, ``dividendYield``, and friends — are dropped.
They are computed against the vendor's period-end price, so they are stale by
construction on every day except one, and we already compute their equivalents
against a daily market cap in :mod:`core.data.factors.fundamental_factors`.
Keeping both would put two columns with the same name-shape and different
freshness in front of a researcher.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.data.factors.statement_metrics import resolve_publication_date
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

# Vendor fields worth carrying: fundamentals-only quantities that we do NOT
# already compute ourselves. Grouped by source dataset.
KEY_METRIC_FIELDS: dict[str, str] = {
    "return_on_invested_capital": "returnOnInvestedCapital",
    "return_on_capital_employed": "returnOnCapitalEmployed",
    "return_on_tangible_assets": "returnOnTangibleAssets",
    "income_quality": "incomeQuality",
    "capex_to_operating_cash_flow": "capexToOperatingCashFlow",
    "capex_to_depreciation": "capexToDepreciation",
    "days_sales_outstanding": "daysOfSalesOutstanding",
    "days_inventory_outstanding": "daysOfInventoryOutstanding",
    "days_payables_outstanding": "daysOfPayablesOutstanding",
    "cash_conversion_cycle": "cashConversionCycle",
    "operating_cycle": "operatingCycle",
    "intangibles_to_total_assets": "intangiblesToTotalAssets",
    "sbc_to_revenue": "stockBasedCompensationToRevenue",
    "working_capital": "workingCapital",
    "invested_capital": "investedCapital",
    "tangible_asset_value": "tangibleAssetValue",
    "net_current_asset_value": "netCurrentAssetValue",
}

RATIO_FIELDS: dict[str, str] = {
    "quick_ratio": "quickRatio",
    "cash_ratio": "cashRatio",
    "solvency_ratio": "solvencyRatio",
    "interest_coverage_ratio": "interestCoverageRatio",
    "debt_service_coverage_ratio": "debtServiceCoverageRatio",
    "receivables_turnover": "receivablesTurnover",
    "inventory_turnover": "inventoryTurnover",
    "payables_turnover": "payablesTurnover",
    "fixed_asset_turnover": "fixedAssetTurnover",
    "working_capital_turnover": "workingCapitalTurnoverRatio",
    "effective_tax_rate": "effectiveTaxRate",
    "dividend_payout_ratio": "dividendPayoutRatio",
    "operating_cash_flow_ratio": "operatingCashFlowRatio",
    "financial_leverage_ratio": "financialLeverageRatio",
    "net_income_per_ebt": "netIncomePerEBT",
    "ebt_per_ebit": "ebtPerEbit",
}

GROWTH_FIELDS: dict[str, str] = {
    "eps_growth": "epsgrowth",
    "operating_cash_flow_growth": "operatingCashFlowGrowth",
    "free_cash_flow_growth": "freeCashFlowGrowth",
    "book_value_per_share_growth": "bookValueperShareGrowth",
    "debt_growth": "debtGrowth",
    "rd_expense_growth": "rdexpenseGrowth",
    "receivables_growth": "receivablesGrowth",
    "three_year_revenue_growth_per_share": "threeYRevenueGrowthPerShare",
    "five_year_revenue_growth_per_share": "fiveYRevenueGrowthPerShare",
}

VENDOR_DATASET_FIELDS: dict[str, dict[str, str]] = {
    "key_metrics": KEY_METRIC_FIELDS,
    "ratios": RATIO_FIELDS,
    "financial_growth": GROWTH_FIELDS,
}

VENDOR_METRIC_COLUMNS: tuple[str, ...] = (
    *KEY_METRIC_FIELDS,
    *RATIO_FIELDS,
    *GROWTH_FIELDS,
)


def build_filing_date_lookup(statements: pd.DataFrame) -> pd.Series:
    """
    Map each fiscal period end to the date its statements became public.

    Args:
        statements: Raw quarterly statements for one symbol, with ``date``
            (period end) and ``acceptedDate`` (EDGAR acceptance).

    Returns:
        Series indexed by period-end date, values = normalized publication date.

    Raises:
        DataSchemaError: If the point-in-time fields are missing.
    """
    if statements.empty:
        return pd.Series(dtype="datetime64[ns]")
    if missing := {"date", "acceptedDate"} - set(statements.columns):
        raise DataSchemaError(f"statements missing point-in-time fields: {missing}")

    lookup = statements[["date", "acceptedDate"]].dropna().copy()
    lookup["date"] = pd.to_datetime(lookup["date"])
    lookup["acceptedDate"] = pd.to_datetime(lookup["acceptedDate"]).dt.normalize()

    # Same guard as the statement metrics: FMP fills acceptedDate with the period
    # end for pre-EDGAR filings, which would date a quarter's ratios to the day
    # the quarter closed. Substitute a conservative filing lag instead.
    resolved, _ = resolve_publication_date(lookup["acceptedDate"], pd.DatetimeIndex(lookup["date"]))
    lookup["acceptedDate"] = resolved.to_numpy()

    # An amendment can restate a period; the later acceptance is when the final
    # numbers were public, and that is the conservative choice.
    return (
        lookup.sort_values("acceptedDate")
        .drop_duplicates("date", keep="last")
        .set_index("date")["acceptedDate"]
    )


def attach_publication_dates(
    vendor_frame: pd.DataFrame,
    filing_dates: pd.Series,
    field_map: dict[str, str],
) -> pd.DataFrame:
    """
    Re-index one vendor dataset from fiscal period end onto publication date.

    Periods with no matching filing date are **dropped**, not guessed. A vendor
    row we cannot date is not usable, and inventing a lag would reintroduce
    exactly the error this module exists to prevent.

    Args:
        vendor_frame: Raw rows for one symbol from a ``period_end_only`` dataset.
        filing_dates: Output of :func:`build_filing_date_lookup` for that symbol.
        field_map: ``{output_name: vendor_field}`` to carry through.

    Returns:
        DataFrame indexed by ``publication_date`` (ascending, unique) with the
        mapped columns as float64. Empty when nothing could be dated.
    """
    empty = pd.DataFrame(columns=list(field_map))
    empty.index.name = "publication_date"
    if vendor_frame.empty or filing_dates.empty or "date" not in vendor_frame.columns:
        return empty

    frame = vendor_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])

    publication_date = frame["date"].map(filing_dates)
    frame = frame[publication_date.notna()]
    if frame.empty:
        return empty

    selected = pd.DataFrame(index=pd.DatetimeIndex(publication_date.dropna()))
    selected.index.name = "publication_date"
    for output_name, vendor_field in field_map.items():
        if vendor_field in frame.columns:
            selected[output_name] = pd.to_numeric(frame[vendor_field], errors="coerce").to_numpy()
        else:
            selected[output_name] = float("nan")

    return (
        selected.astype("float64")
        .sort_index()[lambda df: ~df.index.duplicated(keep="last")]
        .replace([float("inf"), float("-inf")], float("nan"))
    )


def build_symbol_vendor_metrics(
    vendor_frames: dict[str, pd.DataFrame],
    statements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine every vendor dataset for one symbol into one publication-dated frame.

    Args:
        vendor_frames: ``{dataset_name: raw frame}`` for the datasets in
            :data:`VENDOR_DATASET_FIELDS`. Missing datasets are tolerated.
        statements: That symbol's raw income statements, used for filing dates.

    Returns:
        DataFrame indexed by ``publication_date`` with
        :data:`VENDOR_METRIC_COLUMNS`. Empty when no row could be dated.
    """
    filing_dates = build_filing_date_lookup(statements)
    if filing_dates.empty:
        empty = pd.DataFrame(columns=list(VENDOR_METRIC_COLUMNS))
        empty.index.name = "publication_date"
        return empty

    dated: list[pd.DataFrame] = []
    for dataset, field_map in VENDOR_DATASET_FIELDS.items():
        frame = vendor_frames.get(dataset)
        if frame is None or frame.empty:
            continue
        attached = attach_publication_dates(frame, filing_dates, field_map)
        if not attached.empty:
            dated.append(attached)

    if not dated:
        empty = pd.DataFrame(columns=list(VENDOR_METRIC_COLUMNS))
        empty.index.name = "publication_date"
        return empty

    combined = pd.concat(dated, axis=1).sort_index()
    for column in VENDOR_METRIC_COLUMNS:
        if column not in combined.columns:
            combined[column] = float("nan")
    return combined[list(VENDOR_METRIC_COLUMNS)]
