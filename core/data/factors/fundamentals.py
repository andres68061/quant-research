"""Point-in-time fundamentals: derive a leakage-free panel from raw FMP statements.

Vocabulary (see CLAUDE.md): each statement row has a ``reference_date`` (fiscal
period end, vendor field ``date``) and a ``publication_date`` (when it became
knowable, vendor field ``acceptedDate``). This module aligns every metric on
its publication date — a Q4 balance sheet is NOT knowable on Dec 31.

Conservative visibility rule: a filing accepted on day T becomes usable on the
first trading day strictly AFTER T (EDGAR acceptance can be after the close;
one extra day costs almost nothing and removes the intraday ambiguity).

Values forward-fill until the next filing, capped at ``MAX_STALENESS_TRADING_DAYS``
so a delisted company's last filing does not live forever.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

TTM_QUARTERS = 4
# ~13 months of trading days: one missed annual cycle kills the signal.
MAX_STALENESS_TRADING_DAYS = 273

PIT_METRIC_COLUMNS = [
    "book_equity",
    "net_income_ttm",
    "revenue_ttm",
    "total_assets",
    "asset_growth_yoy",
    "shares_diluted",
]


def extract_pit_metrics(
    income_statements: pd.DataFrame,
    balance_sheets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one symbol's publication-dated metric rows from raw statements.

    Args:
        income_statements: Raw quarterly income statements (one symbol),
            requiring ``date``, ``acceptedDate``, ``netIncome``, ``revenue``,
            ``weightedAverageShsOutDil``.
        balance_sheets: Raw quarterly balance sheets (one symbol), requiring
            ``date``, ``acceptedDate``, ``totalStockholdersEquity``, ``totalAssets``.

    Returns:
        DataFrame indexed by ``publication_date`` (normalized acceptedDate),
        sorted ascending, with columns ``PIT_METRIC_COLUMNS`` plus
        ``reference_date``. TTM sums need 4 consecutive quarters; YoY growth
        needs a quarter 4 periods back — earlier rows hold NaN for those.

    Raises:
        DataSchemaError: If required vendor fields are missing.
    """
    required_income = {"date", "acceptedDate", "netIncome", "revenue"}
    required_balance = {"date", "acceptedDate", "totalStockholdersEquity", "totalAssets"}
    if not income_statements.empty and (
        missing := required_income - set(income_statements.columns)
    ):
        raise DataSchemaError(f"income statements missing fields: {missing}")
    if not balance_sheets.empty and (missing := required_balance - set(balance_sheets.columns)):
        raise DataSchemaError(f"balance sheets missing fields: {missing}")
    if income_statements.empty or balance_sheets.empty:
        return pd.DataFrame(columns=[*PIT_METRIC_COLUMNS, "reference_date"])

    income = income_statements.sort_values("date").copy()
    income["net_income_ttm"] = income["netIncome"].rolling(TTM_QUARTERS).sum()
    income["revenue_ttm"] = income["revenue"].rolling(TTM_QUARTERS).sum()
    income_metrics = income.set_index("date")[
        ["acceptedDate", "net_income_ttm", "revenue_ttm", "weightedAverageShsOutDil"]
    ].rename(columns={"weightedAverageShsOutDil": "shares_diluted"})

    balance = balance_sheets.sort_values("date").copy()
    balance["asset_growth_yoy"] = (
        balance["totalAssets"] / balance["totalAssets"].shift(TTM_QUARTERS) - 1.0
    )
    balance_metrics = balance.set_index("date")[
        ["acceptedDate", "totalStockholdersEquity", "totalAssets", "asset_growth_yoy"]
    ].rename(columns={"totalStockholdersEquity": "book_equity", "totalAssets": "total_assets"})

    # Join on reference_date (fiscal period end); publication is the LATER of
    # the two filings so no metric is visible before both statements exist.
    merged = income_metrics.join(balance_metrics, how="inner", lsuffix="", rsuffix="_bal")
    merged["publication_date"] = (
        merged[["acceptedDate", "acceptedDate_bal"]].max(axis=1).dt.normalize()
    )
    merged["reference_date"] = merged.index

    result = (
        merged.reset_index(drop=True)
        .set_index("publication_date")[[*PIT_METRIC_COLUMNS, "reference_date"]]
        .sort_index()
    )
    # If two filings publish the same day (amendments), keep the later period.
    return result[~result.index.duplicated(keep="last")]


def build_pit_fundamentals_panel(
    per_symbol_metrics: dict[str, pd.DataFrame],
    trading_index: pd.DatetimeIndex,
    max_staleness_days: int = MAX_STALENESS_TRADING_DAYS,
) -> pd.DataFrame:
    """
    Assemble per-symbol publication-dated metrics into a tradable panel.

    Each metric becomes visible on the first trading day strictly AFTER its
    publication date and forward-fills until the next filing (capped).

    Args:
        per_symbol_metrics: ``{symbol: extract_pit_metrics(...) output}``.
        trading_index: Target trading calendar (tz-aware, from the price panel).
        max_staleness_days: Forward-fill cap in trading days.

    Returns:
        MultiIndex (date, symbol) DataFrame with ``PIT_METRIC_COLUMNS``.

    Example:
        >>> panel = build_pit_fundamentals_panel({"AAPL": aapl_metrics}, prices.index)
        ... # doctest: +SKIP
    """
    tz = trading_index.tz
    symbol_frames = []
    for symbol, metrics in per_symbol_metrics.items():
        if metrics.empty:
            continue
        publication = metrics.index
        if tz is not None and publication.tz is None:
            publication = publication.tz_localize(tz)
        # First trading day strictly after publication.
        positions = trading_index.searchsorted(publication, side="right")
        valid = positions < len(trading_index)
        if not valid.any():
            continue
        visible = metrics.iloc[np.flatnonzero(valid)]
        visible_dates = trading_index[positions[valid]]

        symbol_metrics = visible[PIT_METRIC_COLUMNS].copy()
        symbol_metrics.index = visible_dates
        # Same visible day for consecutive filings: keep the latest.
        symbol_metrics = symbol_metrics[~symbol_metrics.index.duplicated(keep="last")]
        aligned = symbol_metrics.reindex(trading_index).ffill(limit=max_staleness_days)
        aligned["symbol"] = symbol
        symbol_frames.append(aligned)

    if not symbol_frames:
        return pd.DataFrame(columns=PIT_METRIC_COLUMNS)

    panel = pd.concat(symbol_frames)
    panel.index.name = "date"
    panel = panel.set_index("symbol", append=True).sort_index()
    return panel.dropna(how="all")


def compute_fundamental_factors(
    pit_panel: pd.DataFrame,
    market_cap: pd.Series,
) -> pd.DataFrame:
    """
    Compute valuation/quality factors from the PIT panel and market caps.

    Args:
        pit_panel: Output of :func:`build_pit_fundamentals_panel`
            (MultiIndex (date, symbol)).
        market_cap: MultiIndex (date, symbol) market capitalization series in
            the same currency units as the statements.

    Returns:
        DataFrame indexed like ``pit_panel`` with columns:
        - ``book_to_market``: book equity / market cap (high = cheap)
        - ``earnings_yield``: TTM net income / market cap (E/P; high = cheap)
        - ``roe``: TTM net income / book equity (high = quality)
        - ``asset_growth``: YoY total asset growth (raw; high = growth)
        - ``neg_asset_growth``: ``-asset_growth`` so the shared ranker can long
          the low-growth premium (Cooper, Gulen & Schill 2008)

    Notes:
        Negative book equity makes book_to_market and roe meaningless — those
        rows get NaN, not a sign flip.
    """
    eps = 1e-10
    cap = market_cap.reindex(pit_panel.index)
    book = pit_panel["book_equity"].where(pit_panel["book_equity"] > 0)

    factors = pd.DataFrame(index=pit_panel.index)
    factors["book_to_market"] = book / (cap + eps)
    factors["earnings_yield"] = pit_panel["net_income_ttm"] / (cap + eps)
    factors["roe"] = pit_panel["net_income_ttm"] / (book + eps)
    factors["asset_growth"] = pit_panel["asset_growth_yoy"]
    factors["neg_asset_growth"] = -pit_panel["asset_growth_yoy"]
    return factors
