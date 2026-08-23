"""
Event-driven backtest simulation API (v0).

Exposes :func:`core.backtest.events.simulator.simulate_equal_weight_rebalances` over HTTP.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.schemas.events_backtest import EventIn, EventSimulateRequest, EventSimulateResponse
from core.backtest.events import Event, EventType, simulate_equal_weight_rebalances
from core.data.universe_filters import load_non_operating_symbols
from core.exceptions import DataSchemaError
from core.research.caveats import SURFACE_PEAD, as_dicts, caveats_for_surface

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest/events", tags=["backtest-events"])


def _parse_ts(raw: str) -> pd.Timestamp:
    """Parse ISO timestamp; naive values are interpreted as UTC."""
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _dto_to_event(dto: EventIn) -> Event:
    """Map Pydantic DTO to core Event."""
    try:
        et = EventType(dto.event_type)
    except ValueError as exc:
        raise DataSchemaError(f"Invalid event_type: {dto.event_type!r}") from exc
    return Event(
        ts=_parse_ts(dto.ts),
        event_type=et,
        symbol=dto.symbol,
        payload=dto.payload,
    )


def _prices_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a wide price DataFrame: index = tz-aware dates, columns = symbols."""
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        raise DataSchemaError("Each price row must include a 'date' field")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if df["date"].duplicated().any():
        raise DataSchemaError("Duplicate dates in price_rows")
    df = df.set_index("date").sort_index()
    if df.shape[1] == 0:
        raise DataSchemaError("price_rows need at least one price column in addition to 'date'")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().any().any():
        raise DataSchemaError("price_rows contain invalid or missing numeric values")
    return df


@lru_cache(maxsize=4)
def _cached_pead_study(
    signal_col: str, horizon_days: int, n_quantiles: int, min_price: float
) -> dict:
    """PEAD study over all stored announcements (~70k events; ~2s compute, cached)."""
    from pathlib import Path

    from api.dependencies import get_prices
    from config.settings import PROJECT_ROOT
    from core.backtest.event_study import extract_events_from_surprise_panel, run_event_study

    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price panel not loaded")
    panel_path = Path(PROJECT_ROOT) / "data" / "factors" / "factors_earnings_surprise.parquet"
    if not panel_path.exists():
        raise HTTPException(status_code=503, detail="Run scripts/build_event_factors.py first")

    surprise_panel = pd.read_parquet(panel_path, columns=[signal_col, "days_since_earnings"])
    events = extract_events_from_surprise_panel(surprise_panel, signal_col=signal_col)
    # A pre-merger SPAC has no earnings to be surprised by.
    excluded = load_non_operating_symbols()
    events = events[~events["symbol"].isin(excluded)]
    result = run_event_study(
        events,
        prices,
        horizon_days=horizon_days,
        n_quantiles=n_quantiles,
        min_price=min_price,
    )

    car_paths = result["car_paths"]
    return {
        "signal": signal_col,
        "horizon_days": horizon_days,
        "n_quantiles": n_quantiles,
        "min_price": min_price,
        "universe": (
            "full canonical panel, non-operating vehicles excluded, "
            f"returns require price >= ${min_price:.2f} and |ret| <= 300%"
        ),
        "n_events": int(sum(result["event_counts"].values())),
        "event_counts": result["event_counts"],
        "first_event": str(events["event_date"].min().date()),
        "last_event": str(events["event_date"].max().date()),
        "spread_t_stat": round(result["spread_t_stat"], 2),
        "spread_final_pct": round(float(result["spread_path"].iloc[-1]) * 100, 3),
        "quantile_paths": [
            {
                "quantile": column,
                "car_pct": [round(float(v) * 100, 4) for v in car_paths[column]],
            }
            for column in car_paths.columns
        ],
        "event_days": [int(d) for d in car_paths.index],
        "caveats": as_dicts(caveats_for_surface(SURFACE_PEAD)),
    }


@router.get("/pead-study")
def pead_study(
    signal: str = Query(
        "sue_price_scaled", pattern="^(sue_price_scaled|sue_std_scaled|revenue_surprise_pct)$"
    ),
    horizon_days: int = Query(60, ge=10, le=120),
    n_quantiles: int = Query(5, ge=3, le=10),
    min_price: float = Query(1.0, ge=0.0, le=50.0),
) -> dict:
    """Event-time PEAD study: average abnormal drift after earnings, by surprise quantile."""
    return _cached_pead_study(signal, horizon_days, n_quantiles, min_price)


@router.post("/simulate", response_model=EventSimulateResponse)
def simulate_events(body: EventSimulateRequest) -> EventSimulateResponse:
    """
    Run equal-weight rebalance simulation on a supplied price panel and event list.

    Inputs must satisfy :class:`~core.backtest.events.log.EventLog` rules (strictly
    increasing tz-aware timestamps). Only ``rebalance`` events with ``symbols`` in
    the payload are used by the v0 simulator.
    """
    try:
        prices = _prices_from_rows(body.price_rows)
        events = [_dto_to_event(e) for e in body.events]
        out = simulate_equal_weight_rebalances(
            prices,
            events,
            transaction_cost=body.transaction_cost,
        )
    except DataSchemaError as exc:
        logger.info("event_simulate_schema_error", extra={"detail": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    idx = out.index
    dates = [t.isoformat() for t in idx]
    returns = [float(x) for x in out.values]
    return EventSimulateResponse(dates=dates, returns=returns)
