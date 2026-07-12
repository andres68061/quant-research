"""Pydantic schemas for the Fama-French factor endpoint."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class FactorStats(BaseModel):
    """Annualized summary statistics for one factor."""

    factor: str
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float


class FF5SeriesResponse(BaseModel):
    """Cumulative growth series and stats for the FF5 daily factors."""

    dates: List[str]
    # factor name -> cumulative growth of $1, aligned with ``dates``
    growth: Dict[str, List[float]]
    stats: List[FactorStats]
    first_date: str
    last_date: str
