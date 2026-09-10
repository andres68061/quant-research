"""Replay / animation frame endpoints."""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.dependencies import (
    get_dollar_adv,
    get_factor_frame,
    get_factor_store,
    get_prices,
)
from api.schemas.strategy import InvestedCoverage
from core.backtest.portfolio import sp500_universe_filter
from core.backtest.replay import precompute_backtest_frames
from core.strategies import run_factor_cross_section_backtest_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/replay", tags=["replay"])


@router.get("/frames")
def get_replay_frames(
    factor: Optional[str] = None,
    rebalance_freq: str = "ME",
    transaction_cost_bps: float = Query(10.0, ge=0),
    top_pct: float = Query(0.20, gt=0, le=1),
    bottom_pct: float = Query(0.20, ge=0, le=1),
    long_only: bool = False,
    tail: int = Query(500, ge=1, le=5000),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    survivorship_free: bool = True,
    min_stocks: int = Query(20, ge=2),
    signal_lag_days: int = Query(1, ge=0),
) -> dict:
    """
    Precompute and return frame-by-frame replay data for a factor-based
    backtest.  Each frame includes date, daily PnL, cumulative PnL,
    drawdown, rolling Sortino, position from long/short headcounts, and
    invested-coverage disclosure for flat (cash) stretches.
    """
    store = get_factor_store()
    prices = get_prices()
    if store is None or prices is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    available = store.available_factors
    factor_col = factor or (available[0] if available else None)
    if factor_col is None or factor_col not in store.column_to_panel:
        raise HTTPException(status_code=400, detail=f"Factor '{factor_col}' not found")
    factors = get_factor_frame(factor_col)

    start = (
        pd.Timestamp(start_date)
        if start_date
        else pd.Timestamp(date.today() - timedelta(days=5 * 365))
    )
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(date.today())

    uf = sp500_universe_filter() if survivorship_free else None

    detail = run_factor_cross_section_backtest_detail(
        factors,
        prices,
        factor_col=factor_col,
        start=start,
        end=end,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        long_only=long_only,
        rebalance_freq=rebalance_freq,
        transaction_cost=transaction_cost_bps / 10_000,
        universe_filter=uf,
        min_stocks=min_stocks,
        signal_lag_days=signal_lag_days,
        dollar_adv=get_dollar_adv(),
    )
    frames = precompute_backtest_frames(
        detail["net_return"],
        n_long=detail["n_long"],
        n_short=detail["n_short"],
    )

    frame_slice = frames[-tail:]
    coverage = InvestedCoverage(**detail["coverage"])

    return {
        "total_frames": len(frames),
        "returned_frames": len(frame_slice),
        "frames": frame_slice,
        "coverage": coverage.model_dump(),
    }
