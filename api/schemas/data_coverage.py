"""Pydantic schemas for the data coverage endpoint."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class DatasetInfo(BaseModel):
    """One dataset file: provenance, shape, freshness."""

    name: str
    source: str
    path: str
    layer: str  # "raw" or "derived"
    rows: int
    columns: int
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    size_mb: float
    description: str


class YearCoverage(BaseModel):
    """Symbols with price data in a given year vs S&P 500 members that year."""

    year: int
    symbols_with_data: int
    sp500_members: int
    coverage_pct: float


class QuarantineEntry(BaseModel):
    """One quality finding on one symbol."""

    symbol: str
    check: str
    value: float
    detail: str
    status: str
    review_note: str = ""


class DataCoverageResponse(BaseModel):
    """Full data inventory: datasets, per-year universe coverage, quality findings."""

    datasets: List[DatasetInfo]
    coverage_by_year: List[YearCoverage]
    quarantine: List[QuarantineEntry]
    quarantined_symbol_count: int
    flagged_symbol_count: int
    total_symbols_loaded: int
    survivorship_note: str = ""
