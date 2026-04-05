"""Replay / animation frame endpoints."""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_factors, get_prices
from core.backtest.portfolio import sp500_universe_filter
from core.strategies import run_factor_cross_section_backtest
from core.replay.precompute import precompute_backtest_frames

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/replay", tags=["replay"])


@router.get("/frames")
def get_replay_frames(
    factor: Optional[str] = None,
    rebalance_freq: str = "ME",
    transaction_cost_bps: float = 10.0,
    top_pct: float = 0.20,
    tail: int = Query(500, ge=1, le=5000),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    survivorship_free: bool = True,
) -> dict:
    """
    Precompute and return frame-by-frame replay data for a factor-based
    backtest.  Each frame includes date, daily PnL, cumulative PnL,
    drawdown, rolling Sharpe, and position.
    """
    factors = get_factors()
    prices = get_prices()
    if factors is None or prices is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    factor_col = factor or factors.columns[0]
    if factor_col not in factors.columns:
        raise HTTPException(status_code=400, detail=f"Factor '{factor_col}' not found")

    start = (
        pd.Timestamp(start_date)
        if start_date
        else pd.Timestamp(date.today() - timedelta(days=5 * 365))
    )
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(date.today())

    uf = sp500_universe_filter() if survivorship_free else None

    net_returns = run_factor_cross_section_backtest(
        factors,
        prices,
        factor_col=factor_col,
        start=start,
        end=end,
        top_pct=top_pct,
        rebalance_freq=rebalance_freq,
        transaction_cost=transaction_cost_bps / 10_000,
        universe_filter=uf,
    )
    frames = precompute_backtest_frames(net_returns)

    frame_slice = frames[-tail:]

    return {
        "total_frames": len(frames),
        "returned_frames": len(frame_slice),
        "frames": frame_slice,
    }
