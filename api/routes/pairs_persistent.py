"""Cointegration-persistence pairs index HTTP endpoint."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import get_dollar_adv, get_prices, get_sectors
from api.schemas.metrics import EquityCurvePoint, PerformanceMetrics
from api.schemas.pairs_persistent import (
    PairsPersistentBacktestRequest,
    PairsPersistentBacktestResponse,
    PairsPersistentPairRow,
    PairsPersistentScreenRow,
)
from core.metrics.performance import calculate_cumulative_returns, calculate_performance_metrics
from core.strategies.pairs_persistent import run_pairs_persistent_index

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pairs"])


@router.post("/run-pairs-persistent-backtest", response_model=PairsPersistentBacktestResponse)
def run_pairs_persistent_backtest(
    req: PairsPersistentBacktestRequest,
) -> PairsPersistentBacktestResponse:
    """Screen crossing+cointegrated pairs, trade each until its cointegration breaks."""
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
        out = run_pairs_persistent_index(
            prices,
            sectors,
            sector_names=req.sector_names,
            start=start,
            end=end,
            dollar_adv=get_dollar_adv() if req.use_adv else None,
            formation_months=req.formation_months,
            rescreen_months=req.rescreen_months,
            top_n_pairs=req.top_n_pairs,
            max_symbols_per_sector=req.max_symbols_per_sector,
            max_adf_pvalue=req.max_adf_pvalue,
            min_crossings=req.min_crossings,
            hedge_window=req.hedge_window,
            zscore_window=req.zscore_window,
            entry_z=req.entry_z,
            exit_z=req.exit_z,
            transaction_cost=req.transaction_cost_bps / 10_000.0,
            signal_lag_days=req.signal_lag_days,
            monitor_window=req.monitor_window,
            check_every_days=req.check_every_days,
            max_pvalue=req.stop_max_pvalue,
            persistence_checks=req.persistence_checks,
            freeze_hedge_in_trade=req.freeze_hedge_in_trade,
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

    screens = [PairsPersistentScreenRow(**f) for f in out["formations"]]
    pair_history = [
        PairsPersistentPairRow(
            symbol_y=h["symbol_y"],
            symbol_x=h["symbol_x"],
            sector=h["sector"],
            formation_adf_pvalue=float(h["formation_adf_pvalue"]),
            formation_crossings=int(h["formation_crossings"]),
            trading_start=str(h["trading_start"].date()),
            stop_date=str(h["stop_date"].date()) if h["stop_date"] is not None else None,
            stopped_early=bool(h["stopped_early"]),
            n_days=int(h["n_days"]),
        )
        for h in out["pair_history"]
    ]

    return PairsPersistentBacktestResponse(
        metrics=PerformanceMetrics(**metrics),
        equity_curve=equity,
        total_days=len(net),
        screens=screens,
        pair_history=pair_history,
    )
