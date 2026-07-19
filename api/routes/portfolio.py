"""Portfolio optimization endpoints (efficient frontier, tangency, simulation)."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_prices
from api.time_utils import bound_timestamp, slice_by_dates
from core.metrics.performance import calculate_cumulative_returns, calculate_performance_metrics
from core.optimization.portfolio import (
    calculate_cal_points,
    calculate_efficient_frontier,
    find_min_variance_portfolio,
    find_tangency_portfolio,
    run_walk_forward_tangency,
    simulate_rebalanced_portfolio,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

MIN_OPTIMIZER_PRICE_ROWS = 60


class OptimizeRequest(BaseModel):
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    risk_free_rate: float = 0.08
    borrowing_rate: float = 0.11


class SimulateRequest(BaseModel):
    symbols: List[str]
    weights: Dict[str, float]
    freq: str = "Annual"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class JointHistoryRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class WalkForwardOptimizeRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=2)
    start_date: str
    end_date: Optional[str] = None
    lookback_months: int = Field(24, ge=3, le=60)
    rebalance_months: int = Field(6, ge=1, le=12)
    risk_free_rate: float = 0.0
    portfolio_kind: str = Field("tangency", description="'tangency' or 'min_variance'")


def _aligned_price_panel(
    symbols: List[str], start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"Symbols not found: {missing}")

    df = prices[symbols].dropna()
    return slice_by_dates(df, start, end)


def _solo_row_counts(
    symbols: List[str], start: Optional[str], end: Optional[str]
) -> Dict[str, int]:
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    out: Dict[str, int] = {}
    for s in symbols:
        ser = prices[s].dropna()
        ser = slice_by_dates(ser, start, end)
        out[s] = int(len(ser))
    return out


def _raise_if_insufficient_history(
    df: pd.DataFrame,
    symbols: List[str],
    start: Optional[str],
    end: Optional[str],
) -> None:
    n_joint = len(df)
    if n_joint >= MIN_OPTIMIZER_PRICE_ROWS:
        return

    solo = _solo_row_counts(symbols, start, end)
    solo_sorted = sorted(solo.items(), key=lambda x: x[1])
    solo_bits = ", ".join(f"{sym}={c}" for sym, c in solo_sorted[:12])
    if len(solo_sorted) > 12:
        solo_bits += ", …"

    raise HTTPException(
        status_code=400,
        detail=(
            f"Insufficient joint history: {n_joint} overlapping trading days where every symbol "
            f"has a price (need >= {MIN_OPTIMIZER_PRICE_ROWS}). "
            f"After your date filter, solo usable rows per symbol: {solo_bits}. "
            "Drop symbols with short history or choose an earlier start date."
        ),
    )


def _get_returns(symbols: List[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = _aligned_price_panel(symbols, start, end)
    _raise_if_insufficient_history(df, symbols, start, end)
    return df.pct_change().dropna()


@router.get("/price-row-counts")
def price_row_counts(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Per-symbol coverage summary, used by the UI to dim ineligible / late-start /
    delisted symbols in the picker.

    For every symbol in the loaded price panel returns:
        - ``count``: number of days with a non-null price inside ``[start, end]``
          (intersected with the symbol's own native trading window).
        - ``first`` / ``last``: ISO dates of the symbol's first and last trade in
          the *full* panel (ignoring ``[start, end]``). These let the frontend
          render "started 2014" or "delisted 2020" cues.

    Also returns the panel's ``last_panel_date`` so the frontend can decide what
    counts as "still trading" (e.g. last trade within ~30 days of the panel tail).
    """
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    last_panel_date: Optional[str] = str(prices.index.max().date()) if len(prices.index) else None

    symbols: Dict[str, Dict[str, object]] = {}
    for col in prices.columns:
        ser_full = prices[col].dropna()
        if ser_full.empty:
            continue
        ser_window = ser_full
        ser_window = slice_by_dates(ser_window, start_date, end_date)
        symbols[str(col)] = {
            "count": int(len(ser_window)),
            "first": str(ser_full.index.min().date()),
            "last": str(ser_full.index.max().date()),
        }

    return {
        "start_date": start_date,
        "end_date": end_date,
        "min_required": MIN_OPTIMIZER_PRICE_ROWS,
        "last_panel_date": last_panel_date,
        "symbols": symbols,
    }


@router.post("/joint-history")
def joint_history(req: JointHistoryRequest) -> dict:
    """Rows where every requested symbol has a price (intersection), plus solo counts."""
    df = _aligned_price_panel(req.symbols, req.start_date, req.end_date)
    solo = _solo_row_counts(req.symbols, req.start_date, req.end_date)
    n_joint = len(df)
    return {
        "joint_rows": n_joint,
        "min_required": MIN_OPTIMIZER_PRICE_ROWS,
        "eligible": n_joint >= MIN_OPTIMIZER_PRICE_ROWS,
        "solo_row_counts": solo,
    }


@router.post("/optimize")
def optimize(req: OptimizeRequest) -> dict:
    """Compute efficient frontier, tangency, and min-vol portfolios."""
    returns = _get_returns(req.symbols, req.start_date, req.end_date)

    mean_ret = (returns.mean() * 252).values
    cov = (returns.cov() * 252).values
    std = (returns.std() * np.sqrt(252)).values

    tangency = find_tangency_portfolio(mean_ret, cov, req.risk_free_rate)
    min_vol = find_min_variance_portfolio(mean_ret, cov)
    frontier = calculate_efficient_frontier(mean_ret, cov, req.risk_free_rate)
    cal = calculate_cal_points(
        tangency["ret"],
        tangency["volatility"],
        req.risk_free_rate,
        req.borrowing_rate,
    )

    tangency["weights"] = {s: w for s, w in zip(req.symbols, tangency["weights"], strict=True)}
    min_vol["weights"] = {s: w for s, w in zip(req.symbols, min_vol["weights"], strict=True)}

    individual = [
        {"symbol": s, "ret": float(mean_ret[i]), "volatility": float(std[i])}
        for i, s in enumerate(req.symbols)
    ]

    return {
        "tangency": tangency,
        "min_vol": min_vol,
        "frontier": frontier,
        "cal": cal,
        "individual": individual,
    }


@router.post("/simulate")
def simulate(req: SimulateRequest) -> dict:
    """Simulate portfolio NAV with periodic rebalancing."""
    symbols = req.symbols
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"Symbols not found: {missing}")

    df = prices[symbols].dropna()
    df = slice_by_dates(df, req.start_date, req.end_date)

    weights = np.array([req.weights.get(s, 0.0) for s in symbols])
    weights = weights / weights.sum()

    nav = simulate_rebalanced_portfolio(df, weights, req.freq)
    daily_ret = nav.pct_change().dropna()
    metrics = calculate_performance_metrics(daily_ret)

    nav_list = [{"date": str(d.date()), "value": float(v)} for d, v in nav.items()]

    return {"nav": nav_list, "metrics": metrics}


@router.post("/walk-forward-optimize")
def walk_forward_optimize(req: WalkForwardOptimizeRequest) -> dict:
    """
    Roll tangency/min-variance weights forward with no lookahead.

    Unlike ``/optimize`` + ``/simulate`` (which fit weights on
    ``[start_date, end_date]`` and then evaluate those same weights over
    the identical window — in-sample look-ahead), this re-fits weights on
    a trailing ``lookback_months`` window every ``rebalance_months`` and
    only ever reports realized returns from *after* each fit. See
    ``core.optimization.portfolio.run_walk_forward_tangency`` for the
    full rationale.
    """
    if req.portfolio_kind not in {"tangency", "min_variance"}:
        raise HTTPException(
            status_code=400, detail="portfolio_kind must be 'tangency' or 'min_variance'"
        )

    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    missing = [s for s in req.symbols if s not in prices.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"Symbols not found: {missing}")

    start = bound_timestamp(req.start_date, prices.index)
    end = bound_timestamp(req.end_date, prices.index) if req.end_date else prices.index.max()

    try:
        out = run_walk_forward_tangency(
            prices,
            req.symbols,
            start=start,
            end=end,
            lookback_months=req.lookback_months,
            rebalance_months=req.rebalance_months,
            risk_free_rate=req.risk_free_rate,
            portfolio_kind=req.portfolio_kind,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    net = out["net_returns"]
    if net.empty:
        raise HTTPException(
            status_code=400,
            detail="No realized out-of-sample days; widen the date range or shorten lookback_months.",
        )

    metrics = calculate_performance_metrics(net)
    cum_wealth = calculate_cumulative_returns(net)
    equity = [
        {"date": str(d.date()), "cumulative_return": float(v - 1.0)} for d, v in cum_wealth.items()
    ]

    return {
        "metrics": metrics,
        "equity_curve": equity,
        "total_days": int(len(net)),
        "periods": out["periods"],
    }
