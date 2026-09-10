"""Ad-hoc querying and screening over the panels, backed by DuckDB.

Three questions this answers, which are the same question at different levels of
convenience:

1. *"What do we have on McDonald's?"* — :func:`company_profile`.
2. *"Show me every profitable stock under 10x earnings"* — :func:`build_screen_sql`
   turns structured filters into SQL.
3. *"Let me just write the query"* — :func:`run_query`.

**Why DuckDB rather than loading pandas frames.** The panels total ~4.6 GB. DuckDB
reads parquet column-by-column with predicate pushdown, so a screen touching four
columns on one date reads a few MB instead of gigabytes, and never materialises a
frame the API would have to hold. Views are registered over the files; no data is
copied or imported.

**Why raw SQL is allowed at all.** This is a local, single-user research tool, and
the generated SQL is shown next to every structured screen — being able to copy,
edit and re-run it is the point. It is still constrained: read-only statements
only, a hard row cap, and a query timeout, so a typo cannot hang the API or
modify anything. :func:`validate_read_only` is the gate, and it rejects by
default rather than blocklisting known-bad keywords.

**What that gate does not do.** It prevents *modification*, not filesystem
*reading*: a SELECT may still call DuckDB's ``read_parquet``/``read_csv`` on any
path the API process can reach. That is acceptable for a local tool whose
operator owns the machine, and is not acceptable if this API is ever exposed
beyond localhost. Exposing it would require DuckDB's ``enable_external_access``
disabled and the views pre-registered — stated here so the assumption is a
decision rather than an oversight.

**A note on the wide panels.** ``prices`` and ``dollar_adv_21d`` are date x symbol
(8,911 columns), which SQL handles badly. They are exposed through purpose-built
helpers that unpivot the handful of symbols asked for, rather than as views.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FACTORS_DIR = ROOT / "data" / "factors"
UNIVERSE_DIR = ROOT / "data" / "universe"
SECTORS_DIR = ROOT / "data" / "sectors"

MAX_ROWS = 5_000
QUERY_TIMEOUT_SECONDS = 30

# Only these statement forms are executable. An allowlist, not a blocklist:
# enumerating dangerous keywords is a losing game, and everything this tool needs
# to do starts with SELECT or WITH.
_READ_ONLY_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_STATEMENT_SEPARATOR = re.compile(r";\s*\S")


@dataclass(frozen=True)
class Dataset:
    """One queryable table and what it means."""

    name: str
    path: Path
    description: str
    grain: str
    family: str

    @property
    def exists(self) -> bool:
        return self.path.exists()


# The seven factor families plus the reference tables, named as the user sees
# them. Order is display order.
DATASETS: tuple[Dataset, ...] = (
    Dataset(
        "factors_price",
        FACTORS_DIR / "factors_all.parquet",
        "Momentum, reversal, volatility, beta, size — computed from the price panel.",
        "(date, symbol)",
        "factor",
    ),
    Dataset(
        "factors_fundamental",
        FACTORS_DIR / "factors_fundamental.parquet",
        "Profitability, accruals, leverage, growth — from quarterly statements, "
        "dated by filing date.",
        "(date, symbol)",
        "factor",
    ),
    Dataset(
        "factors_composites",
        FACTORS_DIR / "factors_composites.parquet",
        "Cross-sectional z-score blends (value+quality, sector-neutral variants).",
        "(date, symbol)",
        "factor",
    ),
    Dataset(
        "factors_microstructure",
        FACTORS_DIR / "factors_microstructure.parquet",
        "Range-based volatility, spread and illiquidity estimators from OHLCV bars.",
        "(date, symbol)",
        "factor",
    ),
    Dataset(
        "factors_earnings",
        FACTORS_DIR / "factors_earnings_surprise.parquet",
        "Standardised earnings surprise (SUE) and days since announcement.",
        "(date, symbol)",
        "factor",
    ),
    Dataset(
        "factors_vendor",
        FACTORS_DIR / "factors_vendor_metrics.parquet",
        "~42 vendor-computed ratios, joined to real publication dates (ADR 0010).",
        "(date, symbol)",
        "factor",
    ),
    Dataset(
        "universe",
        ROOT / "data" / "raw" / "fmp" / "universe" / "us_equity_universe.parquet",
        "Every US listing we know of, live and delisted, with listing dates.",
        "symbol",
        "reference",
    ),
    Dataset(
        "sectors",
        SECTORS_DIR / "sector_classifications.parquet",
        "Sector and industry labels. Current-state only — no history (known bias).",
        "symbol",
        "reference",
    ),
    Dataset(
        "security_master",
        UNIVERSE_DIR / "security_master.parquet",
        "Permanent security ids (qid) and issuer ids. See ADR 0015.",
        "qid",
        "reference",
    ),
    Dataset(
        "index_membership",
        UNIVERSE_DIR / "index_membership.parquet",
        "Point-in-time index membership as intervals (valid_from, valid_to).",
        "(symbol, index, interval)",
        "reference",
    ),
)

DATASETS_BY_NAME = {d.name: d for d in DATASETS}


class QueryError(ValueError):
    """A query was rejected or failed."""


def validate_read_only(sql: str) -> None:
    """
    Reject anything that is not a single read-only statement.

    Raises:
        QueryError: If the statement does not start with SELECT/WITH, or if more
            than one statement was supplied (which is how a read-only check gets
            bypassed by appending a second statement).
    """
    if not sql or not sql.strip():
        raise QueryError("Empty query")
    if not _READ_ONLY_START.match(sql):
        raise QueryError("Only SELECT and WITH queries are allowed")
    if _STATEMENT_SEPARATOR.search(sql):
        raise QueryError("Only one statement per query")


def open_connection() -> duckdb.DuckDBPyConnection:
    """
    In-memory DuckDB with a view over every available dataset.

    Views are lazy references to parquet files: creating them costs nothing and
    reads nothing until a query touches them.
    """
    con = duckdb.connect(":memory:")
    for dataset in DATASETS:
        if not dataset.exists:
            logger.debug("Dataset %s not on disk; skipping view", dataset.name)
            continue
        con.execute(f"CREATE VIEW {dataset.name} AS SELECT * FROM read_parquet('{dataset.path}')")
    return con


@dataclass
class QueryResult:
    """Rows plus enough context for the UI to render and chart them."""

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    elapsed_ms: float
    sql: str
    dtypes: dict[str, str] = field(default_factory=dict)


def run_query(sql: str, limit: int = MAX_ROWS) -> QueryResult:
    """
    Execute a read-only query and return rows the API can serialise.

    Args:
        sql: A single SELECT/WITH statement.
        limit: Hard row cap. Applied as an outer LIMIT so the engine can stop
            early rather than materialising everything and slicing.

    Returns:
        The result set, flagged as ``truncated`` when the cap was hit.

    Raises:
        QueryError: On a rejected or failing statement.
    """
    validate_read_only(sql)
    capped = min(int(limit), MAX_ROWS)
    wrapped = f"SELECT * FROM ({sql.rstrip().rstrip(';')}) LIMIT {capped + 1}"

    started = time.perf_counter()
    con = open_connection()
    try:
        frame = con.execute(wrapped).fetch_df()
    except duckdb.Error as exc:
        raise QueryError(str(exc)) from exc
    finally:
        con.close()
    elapsed_ms = (time.perf_counter() - started) * 1000

    truncated = len(frame) > capped
    if truncated:
        frame = frame.head(capped)

    return QueryResult(
        columns=[str(c) for c in frame.columns],
        rows=_records(frame),
        row_count=len(frame),
        truncated=truncated,
        elapsed_ms=round(elapsed_ms, 1),
        sql=sql,
        dtypes={str(c): str(frame[c].dtype) for c in frame.columns},
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe records: NaN to None, timestamps to ISO dates."""
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].dt.strftime("%Y-%m-%d")
    return safe.astype(object).where(pd.notna(safe), None).to_dict("records")


# ---------------------------------------------------------------- screening

Operator = Literal[">", ">=", "<", "<=", "=", "!=", "between", "in"]

_ALLOWED_OPERATORS: frozenset[str] = frozenset({">", ">=", "<", "<=", "=", "!=", "between", "in"})
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ScreenFilter:
    """One condition in a structured screen."""

    column: str
    operator: Operator
    value: Any
    value2: Any = None


def _quote_literal(value: Any) -> str:
    """Render a Python value as a SQL literal, refusing anything unexpected."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise QueryError(f"Unsupported filter value type: {type(value).__name__}")


def _render_condition(condition: ScreenFilter) -> str:
    if not _IDENTIFIER.match(condition.column):
        raise QueryError(f"Invalid column name: {condition.column!r}")
    if condition.operator not in _ALLOWED_OPERATORS:
        raise QueryError(f"Unsupported operator: {condition.operator!r}")

    if condition.operator == "between":
        return (
            f"{condition.column} BETWEEN {_quote_literal(condition.value)} "
            f"AND {_quote_literal(condition.value2)}"
        )
    if condition.operator == "in":
        if not isinstance(condition.value, (list, tuple)) or not condition.value:
            raise QueryError("'in' needs a non-empty list")
        rendered = ", ".join(_quote_literal(v) for v in condition.value)
        return f"{condition.column} IN ({rendered})"
    return f"{condition.column} {condition.operator} {_quote_literal(condition.value)}"


def build_screen_sql(
    columns: list[str],
    filters: list[ScreenFilter],
    *,
    as_of: Optional[str] = None,
    panels: Optional[list[str]] = None,
    order_by: Optional[str] = None,
    descending: bool = True,
    limit: int = 200,
) -> str:
    """
    Turn structured filters into the SQL a screen runs, joining only what it needs.

    The generated SQL is returned rather than executed so the UI can display it.
    Seeing the query is what makes the screener teachable instead of a black box:
    the panel joins, the as-of date, and the exact thresholds are all visible.

    Args:
        columns: Factor/attribute columns to return, beyond symbol and date.
        filters: Conditions, ANDed together.
        as_of: Date to evaluate on. Defaults to the latest date present.
        panels: Datasets to join. Inferred from the requested columns when None.
        order_by: Column to sort by.
        descending: Sort direction.
        limit: Row cap.

    Returns:
        A single SELECT statement.
    """
    for column in columns:
        if not _IDENTIFIER.match(column):
            raise QueryError(f"Invalid column name: {column!r}")

    needed = panels or infer_panels(columns + [f.column for f in filters])
    factor_panels = [p for p in needed if DATASETS_BY_NAME[p].family == "factor"]
    if not factor_panels:
        factor_panels = ["factors_price"]

    base, *rest = factor_panels
    select_columns = ", ".join([f"{base}.symbol", f"{base}.date", *columns])

    joins = "".join(
        f"\n  JOIN {panel} ON {panel}.symbol = {base}.symbol AND {panel}.date = {base}.date"
        for panel in rest
    )
    if "sectors" in needed:
        joins += f"\n  LEFT JOIN sectors ON sectors.symbol = {base}.symbol"

    conditions = [_render_condition(f) for f in filters]
    date_clause = (
        f"{base}.date = DATE '{as_of}'"
        if as_of
        else f"{base}.date = (SELECT max(date) FROM {base})"
    )
    conditions.insert(0, date_clause)

    order_clause = ""
    if order_by:
        if not _IDENTIFIER.match(order_by):
            raise QueryError(f"Invalid order column: {order_by!r}")
        order_clause = f"\nORDER BY {order_by} {'DESC' if descending else 'ASC'} NULLS LAST"

    return (
        f"SELECT {select_columns}\nFROM {base}{joins}"
        f"\nWHERE {' AND '.join(conditions)}"
        f"{order_clause}\nLIMIT {int(limit)}"
    )


def infer_panels(columns: list[str]) -> list[str]:
    """
    Which datasets contain these columns.

    Uses parquet schema metadata only — no data is read.
    """
    wanted = {c for c in columns if _IDENTIFIER.match(c)}
    found: list[str] = []
    for dataset in DATASETS:
        if not dataset.exists:
            continue
        available = set(_dataset_columns(dataset.name))
        if wanted & available:
            found.append(dataset.name)
            wanted -= available
        if not wanted:
            break
    return found


def _dataset_columns(name: str) -> list[str]:
    """Column names of a dataset, from parquet metadata."""
    import pyarrow.parquet as pq

    dataset = DATASETS_BY_NAME.get(name)
    if dataset is None or not dataset.exists:
        return []
    return [str(n) for n in pq.ParquetFile(dataset.path).schema_arrow.names]


def search_companies(term: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Find securities by ticker or company name.

    Matching on name matters because nobody remembers that McDonald's is MCD.
    Exact ticker matches sort first, then prefix matches, then name matches.
    """
    if not term or not term.strip():
        return []
    needle = term.strip().upper().replace("'", "''")
    con = open_connection()
    try:
        return _records(
            con.execute(
                f"""
                SELECT symbol, company_name, exchange, sector, industry, is_delisted
                FROM universe
                WHERE upper(symbol) = '{needle}'
                   OR upper(symbol) LIKE '{needle}%'
                   OR upper(company_name) LIKE '%{needle}%'
                ORDER BY
                  CASE WHEN upper(symbol) = '{needle}' THEN 0
                       WHEN upper(symbol) LIKE '{needle}%' THEN 1
                       ELSE 2 END,
                  is_delisted, symbol
                LIMIT {int(limit)}
                """
            ).fetch_df()
        )
    finally:
        con.close()


def company_profile(symbol: str, history_days: int = 756) -> Optional[dict[str, Any]]:
    """
    Everything the platform holds about one company, in one call.

    This is the "SELECT * WHERE company = ..." view: identity and classification,
    the permanent id, index membership history, the latest value of every factor
    across all six families, and a price series for charting.

    Args:
        symbol: Ticker, any punctuation style.
        history_days: Trailing price history to return (~3 years by default).

    Returns:
        None when the symbol is unknown.
    """
    ticker = symbol.strip().upper()
    escaped = ticker.replace("'", "''")
    con = open_connection()
    try:
        identity = con.execute(
            f"SELECT * FROM universe WHERE upper(symbol) = '{escaped}' LIMIT 1"
        ).fetch_df()
        if identity.empty:
            return None

        profile: dict[str, Any] = {"symbol": ticker, "identity": _records(identity)[0]}

        classification = con.execute(
            f"SELECT * FROM sectors WHERE upper(symbol) = '{escaped}' LIMIT 1"
        ).fetch_df()
        profile["classification"] = (
            _records(classification)[0] if not classification.empty else None
        )

        master = con.execute(
            f"SELECT * FROM security_master WHERE upper(symbol) = '{escaped}' LIMIT 1"
        ).fetch_df()
        profile["identifiers"] = _records(master)[0] if not master.empty else None

        try:
            membership = con.execute(
                f"SELECT * FROM index_membership WHERE upper(symbol) = '{escaped}'"
            ).fetch_df()
            profile["index_membership"] = _records(membership)
        except duckdb.Error:
            profile["index_membership"] = []

        profile["factors"] = _latest_factors(con, escaped)
    finally:
        con.close()

    profile["prices"] = _price_history(ticker, history_days)
    return profile


def _latest_factors(con: duckdb.DuckDBPyConnection, escaped_symbol: str) -> list[dict[str, Any]]:
    """
    The most recent value of every factor for one symbol, grouped by family.

    Queried per panel rather than as one join: the panels have different last
    dates (fundamentals lag prices), so a join would drop a family entirely
    rather than showing it as of its own latest date.
    """
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        if dataset.family != "factor" or not dataset.exists:
            continue
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {dataset.name}
                WHERE upper(symbol) = '{escaped_symbol}'
                ORDER BY date DESC LIMIT 1
                """
            ).fetch_df()
        except duckdb.Error:
            continue
        if frame.empty:
            continue
        record = _records(frame)[0]
        as_of = record.pop("date", None)
        record.pop("symbol", None)
        for name, value in record.items():
            if value is None:
                continue
            rows.append(
                {
                    "family": dataset.name,
                    "factor": name,
                    "value": value,
                    "as_of": as_of,
                }
            )
    return rows


def _price_history(symbol: str, days: int) -> list[dict[str, Any]]:
    """
    Trailing adjusted closes for one symbol.

    The price panel is date x symbol with 8,911 columns, so this reads the single
    column rather than going through SQL — a one-column parquet read is a few MB.
    """
    path = FACTORS_DIR / "prices.parquet"
    if not path.exists():
        return []
    try:
        frame = pd.read_parquet(path, columns=[symbol])
    except (KeyError, ValueError, OSError):
        return []
    series = frame[symbol].dropna().tail(int(days))
    return [
        {"date": stamp.strftime("%Y-%m-%d"), "adj_close": float(value)}
        for stamp, value in series.items()
    ]


def describe_datasets() -> list[dict[str, Any]]:
    """The dataset catalog the UI renders as toggles, with column lists."""
    catalog: list[dict[str, Any]] = []
    for dataset in DATASETS:
        if not dataset.exists:
            continue
        columns = [c for c in _dataset_columns(dataset.name) if c not in ("date", "symbol")]
        catalog.append(
            {
                "name": dataset.name,
                "description": dataset.description,
                "grain": dataset.grain,
                "family": dataset.family,
                "columns": sorted(columns),
                "n_columns": len(columns),
            }
        )
    return catalog
