"""Rolling multi-pair stat-arb index HTTP endpoint."""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import get_dollar_adv, get_prices, get_sectors
from api.schemas.metrics import EquityCurvePoint, PerformanceMetrics
from api.schemas.pairs_index import (
    PairsIndexBacktestRequest,
    PairsIndexBacktestResponse,
    PairsIndexPairRow,
    PairsIndexPeriodRow,
)
from core.metrics.performance import calculate_cumulative_returns, calculate_performance_metrics
from core.strategies.pairs_index import run_pairs_stat_arb_index

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pairs"])


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


@router.post("/run-pairs-index-backtest", response_model=PairsIndexBacktestResponse)
def run_pairs_index_backtest(req: PairsIndexBacktestRequest) -> PairsIndexBacktestResponse:
    """Roll a same-sector pairs basket forward (Gatev SSD formation) and blend returns."""
    prices = get_prices()
    sectors = get_sectors()
    if prices is None or prices.empty:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    if sectors is None or sectors.empty:
        raise HTTPException(status_code=503, detail="Sector data not loaded")

    start = (
        pd.Timestamp(req.start_date, tz="America/New_York")
        if req.start_date
        else pd.Timestamp(date.today() - timedelta(days=14 * 365), tz="America/New_York")
    )
    end = (
        pd.Timestamp(req.end_date, tz="America/New_York")
        if req.end_date
        else pd.Timestamp(date.today(), tz="America/New_York")
    )

    try:
        out = run_pairs_stat_arb_index(
            prices,
            sectors,
            sector_names=req.sector_names,
            start=start,
            end=end,
            dollar_adv=get_dollar_adv() if req.use_adv else None,
            formation_months=req.formation_months,
            trading_months=req.trading_months,
            top_n_pairs=req.top_n_pairs,
            max_symbols_per_sector=req.max_symbols_per_sector,
            hedge_window=req.hedge_window,
            zscore_window=req.zscore_window,
            entry_z=req.entry_z,
            exit_z=req.exit_z,
            transaction_cost=req.transaction_cost_bps / 10_000.0,
            signal_lag_days=req.signal_lag_days,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    net = out["net_returns"]
    if net.empty:
        raise HTTPException(
            status_code=400,
            detail="No index days produced; widen the date range or add sectors.",
        )

    metrics = calculate_performance_metrics(net)
    cum_wealth = calculate_cumulative_returns(net)
    equity = [
        EquityCurvePoint(date=str(d.date()), cumulative_return=float(v - 1.0))
        for d, v in cum_wealth.items()
    ]

    periods = [
        PairsIndexPeriodRow(
            formation_start=p["formation_start"],
            formation_end=p["formation_end"],
            trading_start=p["trading_start"],
            trading_end=p["trading_end"],
            n_candidates_formed=p["n_candidates_formed"],
            n_pairs_selected=p["n_pairs_selected"],
            avg_active_pairs=p["avg_active_pairs"],
            blended_sharpe=_finite_or_none(p["blended_sharpe"]),
            selected_pairs=[
                PairsIndexPairRow(
                    symbol_y=row["symbol_y"],
                    symbol_x=row["symbol_x"],
                    sector=row["sector"],
                    formation_ssd=row["formation_ssd"],
                    formation_adf_pvalue=_finite_or_none(row["formation_adf_pvalue"]),
                    period_sharpe=row["period_sharpe"],
                    period_n_days=row["period_n_days"],
                )
                for row in p["selected_pairs"]
            ],
        )
        for p in out["periods"]
    ]

    return PairsIndexBacktestResponse(
        metrics=PerformanceMetrics(**metrics),
        equity_curve=equity,
        total_days=len(net),
        universe=out["universe"],
        periods=periods,
    )
