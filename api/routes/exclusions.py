"""
Stock exclusion analysis endpoints.

Identifies stocks excluded from portfolio simulations due to price
filters and provides detailed per-symbol diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/exclusions", tags=["exclusions"])

DATA_DIR = Path("data/factors")


def _load_prices() -> pd.DataFrame:
    candidates = [DATA_DIR, Path("../data/factors")]
    for p in candidates:
        fp = p / "prices.parquet"
        if fp.exists():
            return pd.read_parquet(fp)
    raise FileNotFoundError("prices.parquet not found in data/factors/")


class ExclusionStat(BaseModel):
    symbol: str
    min_price: float
    max_price: float
    current_price: float
    days_below: int
    pct_below: float


class ExclusionSummaryResponse(BaseModel):
    total: int
    valid: int
    excluded: int
    threshold: float
    stats: List[ExclusionStat]


class StockDetailResponse(BaseModel):
    symbol: str
    prices: List[dict]
    min_price: float
    max_price: float
    current_price: float
    days_below: int
    pct_below: float
    annualized_vol: float
    max_daily_gain: float
    max_daily_loss: float
    extreme_gains: int
    extreme_losses: int


@router.get("/summary", response_model=ExclusionSummaryResponse)
def exclusion_summary(
    price_threshold: float = Query(5.0, ge=0.5, le=50.0),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Return summary of stocks excluded by a price threshold."""
    try:
        df = _load_prices()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    mask = (df.fillna(np.inf) >= price_threshold).all(axis=0)
    valid_syms = df.columns[mask].tolist()
    excluded_syms = df.columns[~mask].tolist()

    stats: list[ExclusionStat] = []
    for sym in excluded_syms:
        s = df[sym].dropna()
        if len(s) == 0:
            continue
        stats.append(
            ExclusionStat(
                symbol=sym,
                min_price=round(float(s.min()), 4),
                max_price=round(float(s.max()), 4),
                current_price=round(float(s.iloc[-1]), 4),
                days_below=int((s < price_threshold).sum()),
                pct_below=round(float((s < price_threshold).sum() / len(s) * 100), 2),
            )
        )

    stats.sort(key=lambda x: x.min_price)

    return ExclusionSummaryResponse(
        total=len(df.columns),
        valid=len(valid_syms),
        excluded=len(excluded_syms),
        threshold=price_threshold,
        stats=stats,
    )


@router.get("/detail/{symbol}", response_model=StockDetailResponse)
def exclusion_detail(
    symbol: str,
    price_threshold: float = Query(5.0, ge=0.5, le=50.0),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Return detailed price/return data for a single excluded stock."""
    try:
        df = _load_prices()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if symbol not in df.columns:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    s = df[symbol].dropna()
    if len(s) == 0:
        raise HTTPException(status_code=404, detail="No data for symbol in range")

    rets = s.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(252)) if len(rets) > 0 else 0.0

    prices_list = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(v), 4), "below": bool(v < price_threshold)}
        for d, v in s.items()
    ]

    return StockDetailResponse(
        symbol=symbol,
        prices=prices_list,
        min_price=round(float(s.min()), 4),
        max_price=round(float(s.max()), 4),
        current_price=round(float(s.iloc[-1]), 4),
        days_below=int((s < price_threshold).sum()),
        pct_below=round(float((s < price_threshold).sum() / len(s) * 100), 2),
        annualized_vol=round(vol * 100, 2),
        max_daily_gain=round(float(rets.max() * 100), 4) if len(rets) > 0 else 0.0,
        max_daily_loss=round(float(rets.min() * 100), 4) if len(rets) > 0 else 0.0,
        extreme_gains=int((rets > 0.5).sum()) if len(rets) > 0 else 0,
        extreme_losses=int((rets < -0.5).sum()) if len(rets) > 0 else 0,
    )
