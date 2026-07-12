"""Portfolio optimization endpoints (efficient frontier, tangency, simulation)."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_prices
from core.metrics.performance import calculate_performance_metrics
from core.optimization.portfolio import (
    calculate_cal_points,
    calculate_efficient_frontier,
    find_min_variance_portfolio,
    find_tangency_portfolio,
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


def _aligned_price_panel(symbols: List[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"Symbols not found: {missing}")

    df = prices[symbols].dropna()
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


def _solo_row_counts(symbols: List[str], start: Optional[str], end: Optional[str]) -> Dict[str, int]:
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    out: Dict[str, int] = {}
    for s in symbols:
        ser = prices[s].dropna()
        if start:
            ser = ser[ser.index >= start]
        if end:
            ser = ser[ser.index <= end]
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

    last_panel_date: Optional[str] = (
        str(prices.index.max().date()) if len(prices.index) else None
    )

    symbols: Dict[str, Dict[str, object]] = {}
    for col in prices.columns:
        ser_full = prices[col].dropna()
        if ser_full.empty:
            continue
        ser_window = ser_full
        if start_date:
            ser_window = ser_window[ser_window.index >= start_date]
        if end_date:
            ser_window = ser_window[ser_window.index <= end_date]
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
    if req.start_date:
        df = df[df.index >= req.start_date]
    if req.end_date:
        df = df[df.index <= req.end_date]

    weights = np.array([req.weights.get(s, 0.0) for s in symbols])
    weights = weights / weights.sum()

    nav = simulate_rebalanced_portfolio(df, weights, req.freq)
    daily_ret = nav.pct_change().dropna()
    metrics = calculate_performance_metrics(daily_ret)

    nav_list = [
        {"date": str(d.date()), "value": float(v)}
        for d, v in nav.items()
    ]

    return {"nav": nav_list, "metrics": metrics}
