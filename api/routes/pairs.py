"""Pairs / cointegration strategy HTTP endpoints."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import get_dollar_adv, get_prices, get_sectors
from api.schemas.metrics import EquityCurvePoint, PerformanceMetrics
from api.schemas.pairs import (
    EngleGrangerDiagnostics,
    PairsBacktestRequest,
    PairsBacktestResponse,
    PairsDiagnostics,
    PairsScreenRequest,
    PairsScreenResponse,
    PairsScreenRow,
    SpreadPoint,
)
from core.metrics.performance import (
    calculate_cumulative_returns,
    calculate_performance_metrics,
)
from core.strategies.pairs_gatev import resolve_liquid_symbols, screen_pairs_gatev
from core.strategies.pairs_runner import run_pairs_cointegration_backtest
from core.strategies.pairs_screener import resolve_sector_symbols, screen_pairs_walk_forward

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pairs"])


@router.post("/run-pairs-backtest", response_model=PairsBacktestResponse)
def run_pairs_backtest(req: PairsBacktestRequest) -> PairsBacktestResponse:
    """Run Engle–Granger pairs mean-reversion on two symbols from the price panel."""
    if req.entry_z <= req.exit_z:
        raise HTTPException(
            status_code=400,
            detail="entry_z must be strictly greater than exit_z",
        )

    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    symbol_y = req.symbol_y.strip().upper()
    symbol_x = req.symbol_x.strip().upper()
    if symbol_y == symbol_x:
        raise HTTPException(status_code=400, detail="symbol_y and symbol_x must differ")

    start = (
        pd.Timestamp(req.start_date, tz="America/New_York")
        if req.start_date
        else pd.Timestamp(date.today() - timedelta(days=5 * 365), tz="America/New_York")
    )
    end = (
        pd.Timestamp(req.end_date, tz="America/New_York")
        if req.end_date
        else pd.Timestamp(date.today(), tz="America/New_York")
    )

    try:
        result = run_pairs_cointegration_backtest(
            prices,
            symbol_y=symbol_y,
            symbol_x=symbol_x,
            start=start,
            end=end,
            hedge_window=req.hedge_window,
            zscore_window=req.zscore_window,
            entry_z=req.entry_z,
            exit_z=req.exit_z,
            transaction_cost=req.transaction_cost_bps / 10_000,
            signal_lag_days=req.signal_lag_days,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    net = result["net_returns"]
    if net.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid trading days after warm-up; widen the date range.",
        )

    metrics = calculate_performance_metrics(net)
    cum_wealth = calculate_cumulative_returns(net)
    equity = [
        EquityCurvePoint(date=str(d.date()), cumulative_return=float(v - 1.0))
        for d, v in cum_wealth.items()
    ]

    z = result["spread_z"]
    pos = result["position"]
    spread_series = [
        SpreadPoint(
            date=str(ts.date()),
            zscore=float(z.loc[ts]),
            position=float(pos.loc[ts]) if pd.notna(pos.loc[ts]) else 0.0,
        )
        for ts in z.index[-500:]
    ]

    eg_raw = result["diagnostics"]["engle_granger"]
    diagnostics = PairsDiagnostics(
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        engle_granger=EngleGrangerDiagnostics(**eg_raw),
        hedge_window=result["diagnostics"]["hedge_window"],
        zscore_window=result["diagnostics"]["zscore_window"],
        entry_z=result["diagnostics"]["entry_z"],
        exit_z=result["diagnostics"]["exit_z"],
        transaction_cost=result["diagnostics"]["transaction_cost"],
        signal_lag_days=result["diagnostics"]["signal_lag_days"],
        n_days=result["diagnostics"]["n_days"],
        pct_days_in_trade=result["diagnostics"]["pct_days_in_trade"],
    )

    return PairsBacktestResponse(
        metrics=PerformanceMetrics(**metrics),
        equity_curve=equity[-500:],
        total_days=len(net),
        diagnostics=diagnostics,
        spread_series=spread_series,
    )


@router.post("/screen-pairs", response_model=PairsScreenResponse)
def screen_pairs(req: PairsScreenRequest) -> PairsScreenResponse:
    """
    Walk-forward pairs screen.

    ``method=gatev``: formation SSD (Gatev et al.), OOS z-score backtest.
    ``method=engle_granger``: train EG filter, OOS backtest.
    Sector universes default to ADV-ranked liquid names when ``use_adv``.
    """
    if req.entry_z <= req.exit_z:
        raise HTTPException(
            status_code=400,
            detail="entry_z must be strictly greater than exit_z",
        )
    method = req.method.strip().lower()
    if method not in {"gatev", "engle_granger"}:
        raise HTTPException(
            status_code=400,
            detail="method must be 'gatev' or 'engle_granger'",
        )

    prices = get_prices()
    if prices is None or prices.empty:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    symbols: list[str]
    if req.symbols:
        symbols = [s.strip().upper() for s in req.symbols if s.strip()]
        missing = [s for s in symbols if s not in prices.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Symbols not in price panel: {', '.join(missing[:8])}",
            )
    elif req.sector:
        sectors = get_sectors()
        if sectors is None or sectors.empty:
            raise HTTPException(status_code=503, detail="Sector data not loaded")
        try:
            if req.use_adv:
                symbols = resolve_liquid_symbols(
                    sectors,
                    req.sector.strip(),
                    price_columns=list(prices.columns),
                    dollar_adv=get_dollar_adv(),
                    max_symbols=req.max_symbols,
                )
            else:
                symbols = resolve_sector_symbols(
                    sectors,
                    req.sector.strip(),
                    price_columns=list(prices.columns),
                    max_symbols=req.max_symbols,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either sector or symbols",
        )

    if len(symbols) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 symbols in the screen universe",
        )

    start = pd.Timestamp(req.start_date or "2018-01-01", tz="America/New_York")
    end = (
        pd.Timestamp(req.end_date, tz="America/New_York")
        if req.end_date
        else pd.Timestamp(date.today() - timedelta(days=1), tz="America/New_York")
    )
    panel = prices.loc[start:end, symbols].dropna(how="all")
    if panel.empty or len(panel) < 120:
        raise HTTPException(
            status_code=400,
            detail="Insufficient overlapping history for the selected universe",
        )

    try:
        if method == "gatev":
            raw = screen_pairs_gatev(
                panel,
                symbols,
                formation_frac=req.train_frac,
                top_n=req.max_oos_backtests,
                hedge_window=req.hedge_window,
                zscore_window=req.zscore_window,
                entry_z=req.entry_z,
                exit_z=req.exit_z,
                transaction_cost=req.transaction_cost_bps / 10_000.0,
            )
        else:
            raw = screen_pairs_walk_forward(
                panel,
                symbols,
                train_frac=req.train_frac,
                min_train_corr=req.min_train_corr,
                max_train_adf_pvalue=req.max_train_adf_pvalue,
                max_oos_backtests=req.max_oos_backtests,
                hedge_window=req.hedge_window,
                zscore_window=req.zscore_window,
                entry_z=req.entry_z,
                exit_z=req.exit_z,
                transaction_cost=req.transaction_cost_bps / 10_000.0,
            )
            raw["method"] = "engle_granger"
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PairsScreenResponse(
        symbols=list(raw["symbols"]),
        split_date=str(raw["split_date"]),
        train_frac=float(raw["train_frac"]),
        method=str(raw.get("method", method)),
        n_pairs_tested=int(raw["n_pairs_tested"]),
        n_pairs_passed_train=int(raw["n_pairs_passed_train"]),
        results=[PairsScreenRow(**row) for row in raw["results"]],
    )
