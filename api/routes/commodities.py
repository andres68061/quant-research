"""Commodities / Metals analytics endpoints."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from core.data.commodities import COMMODITIES_CONFIG, CommodityDataFetcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/commodities", tags=["commodities"])

_fetcher: Optional[CommodityDataFetcher] = None


def _get_fetcher() -> CommodityDataFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = CommodityDataFetcher()
    return _fetcher


def _load_prices() -> pd.DataFrame:
    fetcher = _get_fetcher()
    df = fetcher.load_prices()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Commodity price data not available. Run fetch script first.")
    return df


@router.get("/list")
def list_commodities() -> dict:
    """Return available commodities with metadata."""
    items = [
        {
            "symbol": sym,
            "name": cfg["name"],
            "category": cfg["category"],
            "unit": cfg["unit"],
        }
        for sym, cfg in COMMODITIES_CONFIG.items()
    ]
    return {"commodities": items}


@router.get("/prices")
def commodity_prices(
    symbols: List[str] = Query(...),
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Return price series for selected commodities."""
    df = _load_prices()
    valid = [s for s in symbols if s in df.columns]
    if not valid:
        raise HTTPException(status_code=404, detail="None of the requested symbols found")

    sub = df[valid].dropna(how="all")
    if start:
        sub = sub[sub.index >= start]
    if end:
        sub = sub[sub.index <= end]

    series: dict = {}
    for col in valid:
        s = sub[col].dropna()
        series[col] = [
            {"date": str(d.date()) if hasattr(d, "date") else str(d), "price": round(float(v), 4)}
            for d, v in s.items()
        ]
    return {"series": series}


@router.get("/returns")
def commodity_returns(
    symbols: List[str] = Query(...),
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Return daily returns + summary statistics for selected commodities."""
    df = _load_prices()
    valid = [s for s in symbols if s in df.columns]
    if not valid:
        raise HTTPException(status_code=404, detail="None of the requested symbols found")

    sub = df[valid].dropna(how="all")
    if start:
        sub = sub[sub.index >= start]
    if end:
        sub = sub[sub.index <= end]

    returns = sub.pct_change().dropna()

    stats = []
    for col in valid:
        r = returns[col].dropna()
        if r.empty:
            continue
        latest = float(sub[col].dropna().iloc[-1]) if not sub[col].dropna().empty else 0.0
        stats.append({
            "symbol": col,
            "mean": float(r.mean()),
            "annualized": float(r.mean() * 252),
            "volatility": float(r.std() * np.sqrt(252)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
            "skew": float(r.skew()),
            "kurtosis": float(r.kurtosis()),
            "latest_price": latest,
        })

    series_data = []
    for idx, row in returns.iterrows():
        point: dict = {"date": str(idx.date()) if hasattr(idx, "date") else str(idx)}
        for col in valid:
            point[col] = round(float(row[col]), 6) if pd.notna(row[col]) else None
        series_data.append(point)

    return {"stats": stats, "series": series_data[-500:]}


@router.get("/correlation")
def commodity_correlation(
    symbols: List[str] = Query(...),
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Return correlation matrix for selected commodities."""
    df = _load_prices()
    valid = [s for s in symbols if s in df.columns]
    if len(valid) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 symbols")

    sub = df[valid].dropna()
    if start:
        sub = sub[sub.index >= start]
    if end:
        sub = sub[sub.index <= end]

    returns = sub.pct_change().dropna()
    corr = returns.corr()

    return {
        "symbols": valid,
        "matrix": corr.values.tolist(),
    }


@router.get("/seasonality")
def commodity_seasonality(
    symbol: str = Query(...),
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Return monthly seasonality analysis for a commodity."""
    df = _load_prices()
    if symbol not in df.columns:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    s = df[symbol].dropna()
    if start:
        s = s[s.index >= start]
    if end:
        s = s[s.index <= end]

    returns = s.pct_change().dropna()
    returns_df = pd.DataFrame({"ret": returns})
    returns_df["month"] = returns_df.index.month
    returns_df["year"] = returns_df.index.year

    monthly_avg = (
        returns_df.groupby("month")["ret"]
        .mean()
        .reset_index()
        .rename(columns={"ret": "avg_return"})
    )

    pivot = returns_df.pivot_table(index="year", columns="month", values="ret", aggfunc="mean")
    heatmap = []
    for year in pivot.index:
        for month in pivot.columns:
            val = pivot.loc[year, month]
            if pd.notna(val):
                heatmap.append({"year": int(year), "month": int(month), "ret": round(float(val), 6)})

    return {
        "symbol": symbol,
        "monthly_avg": monthly_avg.to_dict(orient="records"),
        "heatmap": heatmap,
    }
