"""Registry of per-symbol FMP datasets and a single generic fetcher.

Adding a new FMP dataset should mean adding one :class:`SymbolDataset` entry, not
writing another bespoke script. Each spec records the endpoint, its parameters,
which columns are dates, and — most importantly — the dataset's **point-in-time
status**, because that determines whether a backtest may use it at all.

Point-in-time status
--------------------

``POINT_IN_TIME``
    Rows carry the date the information became public (a filing date, an
    announcement date, or an observation date). Usable directly in a backtest.

``PERIOD_END_ONLY``
    Rows are stamped with the fiscal period they describe but **not** with when
    they were published. Vendor-computed ratios are the common case: a value
    dated 2024-03-31 was not knowable until the 10-Q landed weeks later. To use
    these, join the matching ``filingDate``/``acceptedDate`` from the raw
    statements on ``(symbol, date)`` and shift accordingly — see
    :mod:`core.data.factors.statement_metrics` for the pattern.

``SNAPSHOT``
    A single current-value row with no history. Useful for reconciliation and
    for live screening, never for backtesting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from core.data.vendors.fmp.client import fmp_get
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

POINT_IN_TIME = "point_in_time"
PERIOD_END_ONLY = "period_end_only"
SNAPSHOT = "snapshot"

# ~50 years of quarters; the API returns fewer when history is shorter.
_QUARTERS_LIMIT = 200


@dataclass(frozen=True)
class SymbolDataset:
    """
    One per-symbol FMP endpoint and how to store it.

    Attributes:
        name: Directory name under ``data/raw/fmp/`` and the CLI identifier.
        endpoint: Path relative to the stable base URL.
        params: Query parameters, excluding ``symbol`` and ``apikey``.
        date_columns: Columns to parse as datetimes.
        primary_date: Column to sort by; None for snapshot datasets.
        pit_status: One of :data:`POINT_IN_TIME`, :data:`PERIOD_END_ONLY`,
            :data:`SNAPSHOT`.
        notes: Why the dataset is worth having, and its traps.
    """

    name: str
    endpoint: str
    pit_status: str
    params: dict[str, Any] = field(default_factory=dict)
    date_columns: tuple[str, ...] = ()
    primary_date: Optional[str] = None
    notes: str = ""


SYMBOL_DATASETS: dict[str, SymbolDataset] = {
    "key_metrics": SymbolDataset(
        name="key_metrics",
        endpoint="key-metrics",
        pit_status=PERIOD_END_ONLY,
        params={"period": "quarter", "limit": _QUARTERS_LIMIT},
        date_columns=("date",),
        primary_date="date",
        notes="ROIC, EV multiples, cash conversion cycle, Graham number, working capital.",
    ),
    "ratios": SymbolDataset(
        name="ratios",
        endpoint="ratios",
        pit_status=PERIOD_END_ONLY,
        params={"period": "quarter", "limit": _QUARTERS_LIMIT},
        date_columns=("date",),
        primary_date="date",
        notes="~60 vendor ratios: margins, turnover, coverage, per-share values.",
    ),
    "financial_growth": SymbolDataset(
        name="financial_growth",
        endpoint="financial-growth",
        pit_status=PERIOD_END_ONLY,
        params={"period": "quarter", "limit": _QUARTERS_LIMIT},
        date_columns=("date",),
        primary_date="date",
        notes="YoY and 3/5/10-year per-share growth rates across the statements.",
    ),
    "enterprise_values": SymbolDataset(
        name="enterprise_values",
        endpoint="enterprise-values",
        pit_status=PERIOD_END_ONLY,
        params={"period": "quarter", "limit": _QUARTERS_LIMIT},
        date_columns=("date",),
        primary_date="date",
        notes="Vendor EV build-up; cross-check for our own EV in fundamental_factors.",
    ),
    "financial_scores": SymbolDataset(
        name="financial_scores",
        endpoint="financial-scores",
        pit_status=SNAPSHOT,
        date_columns=(),
        notes="Current Altman Z and Piotroski F. Reconciliation target for our own.",
    ),
    "earnings": SymbolDataset(
        name="earnings",
        endpoint="earnings",
        pit_status=POINT_IN_TIME,
        params={"limit": 500},
        date_columns=("date",),
        primary_date="date",
        notes="Announcement-dated actual vs estimated EPS and revenue. Feeds PEAD.",
    ),
    "dividends": SymbolDataset(
        name="dividends",
        endpoint="dividends",
        pit_status=POINT_IN_TIME,
        params={"limit": 1000},
        date_columns=("date", "recordDate", "paymentDate", "declarationDate"),
        primary_date="date",
        notes="Ex-dates and declaration dates; audits the price adjustment factors.",
    ),
    "splits": SymbolDataset(
        name="splits",
        endpoint="splits",
        pit_status=POINT_IN_TIME,
        params={"limit": 500},
        date_columns=("date",),
        primary_date="date",
        notes="Split ratios; audits adjusted-price continuity and detects bad prints.",
    ),
    "analyst_estimates": SymbolDataset(
        name="analyst_estimates",
        endpoint="analyst-estimates",
        pit_status=SNAPSHOT,
        params={"period": "annual", "limit": 100},
        date_columns=("date",),
        primary_date="date",
        notes=(
            "Consensus for FUTURE fiscal periods as of today — not a revision "
            "history. Dates are period ends in the future, so this cannot be "
            "replayed historically; snapshot it repeatedly to build a revision series."
        ),
    ),
    "grades_historical": SymbolDataset(
        name="grades_historical",
        endpoint="grades-historical",
        pit_status=POINT_IN_TIME,
        params={"limit": 1000},
        date_columns=("date",),
        primary_date="date",
        notes="Monthly analyst rating counts (strong buy .. strong sell) by date.",
    ),
    "ratings_historical": SymbolDataset(
        name="ratings_historical",
        endpoint="ratings-historical",
        pit_status=POINT_IN_TIME,
        params={"limit": 1000},
        date_columns=("date",),
        primary_date="date",
        notes="Vendor composite rating and component scores, observation-dated.",
    ),
    "price_target_summary": SymbolDataset(
        name="price_target_summary",
        endpoint="price-target-summary",
        pit_status=SNAPSHOT,
        notes="Rolling 1m/1q/1y average price targets as of today.",
    ),
    "shares_float": SymbolDataset(
        name="shares_float",
        endpoint="shares-float",
        pit_status=SNAPSHOT,
        date_columns=("date",),
        primary_date="date",
        notes="Free float vs shares outstanding — the correct denominator for liquidity.",
    ),
    "employee_count": SymbolDataset(
        name="employee_count",
        endpoint="employee-count",
        pit_status=POINT_IN_TIME,
        params={"limit": 100},
        date_columns=("filingDate", "periodOfReport", "acceptanceTime"),
        primary_date="filingDate",
        notes="Headcount from 10-K filings, with the filing date. Sales-per-employee.",
    ),
    "insider_trading": SymbolDataset(
        name="insider_trading",
        endpoint="insider-trading/search",
        pit_status=POINT_IN_TIME,
        params={"limit": 1000},
        date_columns=("filingDate", "transactionDate"),
        primary_date="filingDate",
        notes="Form 4 transactions. Use filingDate, not transactionDate, for visibility.",
    ),
    "stock_peers": SymbolDataset(
        name="stock_peers",
        endpoint="stock-peers",
        pit_status=SNAPSHOT,
        notes="Vendor peer group; a starting point for pairs/relative-value screens.",
    ),
}


def fetch_symbol_dataset(
    symbol: str,
    dataset: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch one dataset for one symbol and return it as a typed DataFrame.

    Args:
        symbol: Ticker as listed on FMP.
        dataset: Key into :data:`SYMBOL_DATASETS`.
        api_key: Optional key override.

    Returns:
        DataFrame with vendor columns preserved verbatim, date columns parsed to
        datetime64, and rows sorted ascending by the spec's ``primary_date``.
        Empty DataFrame when FMP has no coverage for the symbol.

    Raises:
        DataSchemaError: For an unknown dataset name or a non-list payload.

    Example:
        >>> earnings = fetch_symbol_dataset("AAPL", "earnings")  # doctest: +SKIP
    """
    if dataset not in SYMBOL_DATASETS:
        raise DataSchemaError(f"Unknown dataset {dataset!r}; known: {sorted(SYMBOL_DATASETS)}")
    spec = SYMBOL_DATASETS[dataset]

    rows = fmp_get(spec.endpoint, {**spec.params, "symbol": symbol}, api_key=api_key)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise DataSchemaError(f"Unexpected FMP payload for {symbol}/{dataset}: {type(rows)}")
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    for column in spec.date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    if spec.primary_date and spec.primary_date in frame.columns:
        frame = frame.sort_values(spec.primary_date).reset_index(drop=True)
    return frame


def describe_datasets() -> pd.DataFrame:
    """Return the registry as a DataFrame — used by docs and the fetch script's --list."""
    return pd.DataFrame(
        [
            {
                "dataset": spec.name,
                "endpoint": spec.endpoint,
                "pit_status": spec.pit_status,
                "notes": spec.notes,
            }
            for spec in SYMBOL_DATASETS.values()
        ]
    )
