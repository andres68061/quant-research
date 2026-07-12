"""Fama-French 5-factor data endpoint (backs the frontend explainer page)."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas.fama_french import FactorStats, FF5SeriesResponse
from config.settings import PROJECT_ROOT
from core.data.factors.fama_french import load_ff5_parquet, prepare_ff5_view

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/fama-french", tags=["fama-french"])

FF5_PATH = PROJECT_ROOT / "data" / "factors" / "fama_french_5.parquet"


@router.get("/series", response_model=FF5SeriesResponse)
def ff5_series(
    start: Optional[str] = Query(None, description="ISO start date, e.g. 1990-01-01"),
) -> FF5SeriesResponse:
    """Cumulative growth of $1 and annualized stats for the five factors."""
    ff5_daily = load_ff5_parquet(FF5_PATH)
    if ff5_daily is None or ff5_daily.empty:
        raise HTTPException(status_code=404, detail="fama_french_5.parquet not found or empty")

    growth, stats = prepare_ff5_view(ff5_daily, start=start)

    return FF5SeriesResponse(
        dates=[d.strftime("%Y-%m-%d") for d in growth.index],
        growth={col: [round(float(v), 4) for v in growth[col]] for col in growth.columns},
        stats=[
            FactorStats(
                factor=str(factor),
                annualized_return=round(float(row["annualized_return"]), 4),
                annualized_volatility=round(float(row["annualized_volatility"]), 4),
                sharpe_ratio=round(float(row["sharpe_ratio"]), 2),
            )
            for factor, row in stats.iterrows()
        ],
        first_date=growth.index.min().strftime("%Y-%m-%d"),
        last_date=growth.index.max().strftime("%Y-%m-%d"),
    )
