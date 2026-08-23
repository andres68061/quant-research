"""Data explorer endpoints: catalog, company lookup, structured screen, raw query.

Thin handlers over :mod:`core.data.explorer`. All query construction, validation
and execution lives in core; these translate HTTP and shape errors.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.data.explorer import (
    MAX_ROWS,
    QueryError,
    ScreenFilter,
    build_screen_sql,
    company_profile,
    describe_datasets,
    run_query,
    search_companies,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/explorer", tags=["explorer"])


class FilterSpec(BaseModel):
    """One screen condition."""

    column: str
    operator: str = Field(..., description="> >= < <= = != between in")
    value: Any
    value2: Optional[Any] = None


class ScreenRequest(BaseModel):
    """A structured screen: columns to show, conditions to apply."""

    columns: list[str] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    panels: Optional[list[str]] = None
    as_of: Optional[str] = None
    order_by: Optional[str] = None
    descending: bool = True
    limit: int = Field(200, ge=1, le=MAX_ROWS)
    preview_only: bool = Field(
        False,
        description="Return the generated SQL without executing it.",
    )


class QueryRequest(BaseModel):
    """A raw read-only SQL statement."""

    sql: str
    limit: int = Field(500, ge=1, le=MAX_ROWS)


@router.get("/datasets")
def get_datasets() -> dict[str, Any]:
    """The queryable dataset catalog, with every column name."""
    return {"datasets": describe_datasets()}


@router.get("/search")
def get_search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)) -> dict:
    """Find securities by ticker or company name."""
    return {"results": search_companies(q, limit=limit)}


@router.get("/company/{symbol}")
def get_company(symbol: str, history_days: int = Query(756, ge=20, le=20_000)) -> dict[str, Any]:
    """Everything held about one company: identity, ids, factors, prices."""
    profile = company_profile(symbol, history_days=history_days)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown symbol '{symbol}'")
    return profile


@router.post("/screen")
def post_screen(req: ScreenRequest) -> dict[str, Any]:
    """
    Run a structured screen.

    The generated SQL is always returned alongside the rows — the screen is meant
    to be readable and editable, not a black box.
    """
    try:
        sql = build_screen_sql(
            columns=req.columns,
            filters=[
                ScreenFilter(f.column, f.operator, f.value, f.value2)  # type: ignore[arg-type]
                for f in req.filters
            ],
            as_of=req.as_of,
            panels=req.panels,
            order_by=req.order_by,
            descending=req.descending,
            limit=req.limit,
        )
    except QueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if req.preview_only:
        return {"sql": sql, "columns": [], "rows": [], "row_count": 0, "truncated": False}

    try:
        result = run_query(sql, limit=req.limit)
    except QueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "sql": result.sql,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "elapsed_ms": result.elapsed_ms,
        "dtypes": result.dtypes,
    }


@router.post("/query")
def post_query(req: QueryRequest) -> dict[str, Any]:
    """Execute a read-only SQL statement against the dataset views."""
    try:
        result = run_query(req.sql, limit=req.limit)
    except QueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "sql": result.sql,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "elapsed_ms": result.elapsed_ms,
        "dtypes": result.dtypes,
    }
