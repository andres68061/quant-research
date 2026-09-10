"""Sector breakdown analytics endpoints."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from config.settings import PROJECT_ROOT
from core.data.universe.sector_classification import (
    get_sector_summary,
    load_sector_classifications,
)
from core.research.caveats import SURFACE_SECTOR_PERFORMANCE, as_dicts, caveats_for_surface
from core.strategies.sector_index import MIN_MEMBERS_DEFAULT, compute_sector_indices

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sectors", tags=["sectors"])


@router.get("/summary")
def sector_summary() -> dict:
    """Return sector distribution summary (counts + percentages)."""
    df = load_sector_classifications()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Sector data not available")

    summary = get_sector_summary()
    total = int(summary["count"].sum()) if "count" in summary.columns else len(df)

    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "sector": str(row.get("sector", row.name)),
                "count": int(row.get("count", 0)),
                "pct": round(float(row.get("count", 0)) / total * 100, 1) if total > 0 else 0.0,
            }
        )

    return {"total_symbols": total, "sectors": rows}


@router.get("/breakdown")
def sector_breakdown(sector: Optional[str] = Query(None)) -> dict:
    """Return symbols grouped by sector/industry."""
    df = load_sector_classifications()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Sector data not available")

    if sector:
        df = df[df["sector"] == sector]

    symbols = []
    for _, row in df.iterrows():
        symbols.append(
            {
                "symbol": str(row.get("symbol", row.name)),
                "sector": str(row.get("sector", "Unknown")),
                "industry": str(row.get("industry", "Unknown")),
                "type": str(row.get("quoteType", "EQUITY")),
            }
        )

    return {"symbols": symbols}


@lru_cache(maxsize=8)
def _cached_sector_indices(weighting: str, sp500_only: bool, start: str) -> dict:
    """Compute-once cache: the calculation walks 26 years x 774 symbols (~20s)."""
    from api.dependencies import get_prices

    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price panel not loaded")
    classifications = load_sector_classifications()
    if classifications is None or classifications.empty:
        raise HTTPException(status_code=503, detail="Sector data not available")
    symbol_to_sector = classifications.set_index("symbol")["sector"]

    market_cap = None
    if weighting == "cap":
        caps_path = Path(PROJECT_ROOT) / "data" / "market_caps" / "historical_market_caps.parquet"
        caps = pd.read_parquet(caps_path)["market_cap"]
        caps.index = caps.index.set_names(["date", "symbol"])
        market_cap = caps.unstack("symbol")
        if market_cap.index.tz is None and prices.index.tz is not None:
            market_cap.index = market_cap.index.tz_localize(prices.index.tz)

    membership = None
    if sp500_only:
        from core.backtest.portfolio import sp500_universe_filter

        membership = sp500_universe_filter()

    result = compute_sector_indices(
        prices,
        symbol_to_sector,
        market_cap=market_cap,
        weighting=weighting,
        membership_filter=membership,
        start=pd.Timestamp(start, tz=prices.index.tz),
    )
    # Weekly downsampling: 11 sectors x ~6,700 daily points is a heavy payload
    # and indistinguishable from weekly at page scale.
    levels = result["levels"].resample("W-FRI").last().dropna(how="all")
    members = result["members"].resample("W-FRI").last().reindex(levels.index)

    years = (levels.index[-1] - levels.index[0]).days / 365.25
    sectors_payload = []
    for sector_name in levels.columns:
        series = levels[sector_name]
        sectors_payload.append(
            {
                "sector": sector_name,
                "ann_return_pct": round((float(series.iloc[-1]) ** (1 / years) - 1) * 100, 2),
                "members_latest": int(members[sector_name].iloc[-1]),
                "series": [
                    {"date": str(d.date()), "level": round(float(v), 4)} for d, v in series.items()
                ],
            }
        )
    sectors_payload.sort(key=lambda s: -s["ann_return_pct"])
    return {
        "weighting": weighting,
        "sp500_membership_filter": sp500_only,
        "start": start,
        "granularity": "weekly",
        "methodology": {
            "rebalance": (
                "daily — weights recomputed every trading day from the prior day's market "
                "caps (cap) or equally across that day's members (equal); no discrete "
                "reconstitution and no turnover cost"
            ),
            "membership": (
                "resolved per day: a stock contributes only on days it traded"
                + (
                    " AND was an S&P 500 member on that date (index_membership.parquet)"
                    if sp500_only
                    else " (no index filter — the full labeled universe)"
                )
            ),
            "returns": "dividend-adjusted closes, gross of costs and taxes",
            "min_members_per_day": MIN_MEMBERS_DEFAULT,
            "universe_labeled_symbols": int(len(symbol_to_sector.dropna())),
        },
        "sectors": sectors_payload,
        "caveats": as_dicts(caveats_for_surface(SURFACE_SECTOR_PERFORMANCE)),
    }


@router.get("/performance")
def sector_performance(
    weighting: str = Query("cap", pattern="^(cap|equal)$"),
    sp500_only: bool = Query(True),
    start: str = Query("2000-01-03"),
) -> dict:
    """
    Sector index levels over time with day-by-day membership.

    Survivorship-aware (delisted names contribute until their last traded day)
    and weight-lookahead-free (prior-day caps). Remaining biases are listed in
    ``caveats`` — surface them in any UI that shows this data.
    """
    return _cached_sector_indices(weighting, sp500_only, start)
