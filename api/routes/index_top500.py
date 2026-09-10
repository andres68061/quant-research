"""
Simple quarterly top-500 cap-weighted index + Cid-1 relevance study.

Thin handlers only: data comes from ``api.dependencies`` (prices, factors)
plus the market-cap panel (lazy-loaded here, cached at module level); all
math is delegated to ``core.strategies.top500_index`` and ``core.metrics``. Results are
cached per parameter set — the underlying panels are static between data
refreshes.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_factor_store, get_prices, get_sectors
from config.settings import PROJECT_ROOT
from core.metrics.cross_section import calculate_trailing_cid1_cross_section
from core.metrics.performance import calculate_performance_metrics
from core.research.cid1_study import CONTROL_COLUMNS, run_cid1_relevance_study
from core.strategies.top500_index import (
    CapWeightedIndexResult,
    build_quarterly_rebalance_dates,
    compute_cap_weighted_index,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/index/top500", tags=["index-top500"])

MARKET_CAPS_PATH = PROJECT_ROOT / "data" / "market_caps" / "historical_market_caps.parquet"
NY_TZ = "America/New_York"

_market_caps_wide: Optional[pd.DataFrame] = None


def _clean(value: object) -> Optional[float]:
    """NaN/inf are not valid JSON — map them to null at the boundary."""
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _get_market_caps() -> pd.DataFrame:
    global _market_caps_wide
    if _market_caps_wide is None:
        if not MARKET_CAPS_PATH.exists():
            raise HTTPException(status_code=503, detail="Market-cap panel not available")
        long_caps = pd.read_parquet(MARKET_CAPS_PATH)
        symbol_level = "symbol" if "symbol" in long_caps.index.names else "ticker"
        _market_caps_wide = long_caps["market_cap"].unstack(symbol_level)
        logger.info("Loaded market caps: %s", _market_caps_wide.shape)
    return _market_caps_wide


@lru_cache(maxsize=8)
def _build_index(start_iso: str, top_n: int) -> CapWeightedIndexResult:
    prices = get_prices()
    if prices is None:
        raise HTTPException(status_code=503, detail="Price data not loaded")
    caps = _get_market_caps()
    caps = caps[[c for c in caps.columns if c in prices.columns]]
    start = pd.Timestamp(start_iso, tz=NY_TZ)
    # Include the quarter-end just before `start` so performance begins at
    # `start`, not one quarter later.
    rebalances = build_quarterly_rebalance_dates(prices.index, start=start - pd.Timedelta(days=100))
    result = compute_cap_weighted_index(prices, caps, rebalances, top_n=top_n)
    result.daily_returns = result.daily_returns[result.daily_returns.index >= start]
    return result


@lru_cache(maxsize=4)
def _build_study(start_iso: str, top_n: int, window_days: int) -> dict:
    prices = get_prices()
    store = get_factor_store()
    if prices is None or store is None:
        raise HTTPException(status_code=503, detail="Price/factor data not loaded")
    try:
        # Only the study's control columns, not every factor panel.
        factors = store.load_factors(CONTROL_COLUMNS)
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=f"Missing control factors: {exc}") from exc
    caps = _get_market_caps()
    caps = caps[[c for c in caps.columns if c in prices.columns]]
    equities = prices[[c for c in prices.columns if not c.startswith("^")]]
    return run_cid1_relevance_study(
        equities,
        caps,
        factors,
        start=pd.Timestamp(start_iso, tz=NY_TZ),
        top_n=top_n,
        window_days=window_days,
    )


class RebalanceSummary(BaseModel):
    date: str
    n_constituents: int
    turnover: Optional[float]
    top_weight: float
    top10_weight_share: float


class IndexPerformanceResponse(BaseModel):
    dates: List[str]
    index_cumulative: List[float]
    benchmark_cumulative: Optional[List[float]]
    metrics: Dict[str, Optional[float]]
    benchmark_metrics: Optional[Dict[str, Optional[float]]]
    correlation_vs_benchmark: Optional[float]
    tracking_error_ann: Optional[float]
    rebalances: List[RebalanceSummary]


class HoldingRow(BaseModel):
    symbol: str
    sector: Optional[str]
    market_cap: float
    weight: float
    total_return: Optional[float] = Field(None, description="Trailing-window return")
    cost_basis_pain: Optional[float]
    cid1_ratio: Optional[float] = Field(None, description="null = pain-free (+inf)")
    cid1_angle: Optional[float]


class HoldingsResponse(BaseModel):
    date: str
    window_days: int
    holdings: List[HoldingRow]


class StatSummary(BaseModel):
    mean: Optional[float]
    tstat: Optional[float]
    pvalue: Optional[float]
    n_obs: int


class StudyDateRow(BaseModel):
    date: str
    n_symbols: int
    ic: Optional[float]
    persistence_vs_prev: Optional[float]
    median_cid1_angle: Optional[float]
    share_pain_free: Optional[float]
    quantile_spread: Optional[float]


class SensitivityRow(BaseModel):
    start_year: int
    n_quarters: int
    mean_ic: Optional[float]
    ic_tstat: Optional[float]
    mean_spread: Optional[float]
    spread_tstat: Optional[float]


class Cid1StudyResponse(BaseModel):
    per_date: List[StudyDateRow]
    quantile_avg_forward_returns: List[Optional[float]]
    ic_summary: StatSummary
    persistence_summary: StatSummary
    quantile_spread_summary: StatSummary
    fama_macbeth_univariate: StatSummary
    fama_macbeth_multivariate: Dict[str, StatSummary]
    start_year_sensitivity: List[SensitivityRow]
    config: Dict[str, object]


def _stat(d: Dict[str, float]) -> StatSummary:
    return StatSummary(
        mean=_clean(d.get("mean")),
        tstat=_clean(d.get("tstat")),
        pvalue=_clean(d.get("pvalue")),
        n_obs=int(d.get("n_obs", 0)),
    )


@router.get("/performance", response_model=IndexPerformanceResponse)
def index_performance(start: str = "2005-01-01", top_n: int = 500) -> IndexPerformanceResponse:
    """Cumulative performance of the simple index vs ^GSPC, plus rebalance stats."""
    result = _build_index(start, top_n)
    returns = result.daily_returns
    cumulative = (1.0 + returns).cumprod()

    prices = get_prices()
    benchmark_cumulative: Optional[List[float]] = None
    benchmark_metrics: Optional[Dict[str, Optional[float]]] = None
    correlation: Optional[float] = None
    tracking_error: Optional[float] = None
    if prices is not None and "^GSPC" in prices.columns:
        gspc = prices["^GSPC"].pct_change().reindex(returns.index)
        benchmark_cumulative = [_clean(v) or 1.0 for v in (1.0 + gspc.fillna(0.0)).cumprod()]
        benchmark_metrics = {
            k: _clean(v) for k, v in calculate_performance_metrics(gspc.dropna()).items()
        }
        excess = (returns - gspc).dropna()
        if len(excess) > 30:
            correlation = _clean(returns.corr(gspc))
            tracking_error = _clean(excess.std() * (252**0.5))

    rebalances: List[RebalanceSummary] = []
    for date, holdings in result.holdings.items():
        rebalances.append(
            RebalanceSummary(
                date=str(date.date()),
                n_constituents=int(len(holdings)),
                turnover=_clean(result.turnover.get(date)),
                top_weight=float(holdings["weight"].iloc[0]),
                top10_weight_share=float(holdings["weight"].iloc[:10].sum()),
            )
        )

    return IndexPerformanceResponse(
        dates=[str(d.date()) for d in returns.index],
        index_cumulative=[float(v) for v in cumulative],
        benchmark_cumulative=benchmark_cumulative,
        metrics={k: _clean(v) for k, v in calculate_performance_metrics(returns).items()},
        benchmark_metrics=benchmark_metrics,
        correlation_vs_benchmark=correlation,
        tracking_error_ann=tracking_error,
        rebalances=rebalances,
    )


@router.get("/holdings", response_model=HoldingsResponse)
def index_holdings(
    date: str, start: str = "2005-01-01", top_n: int = 500, window_days: int = 252
) -> HoldingsResponse:
    """Constituents at one rebalance date with trailing Cid-1 components."""
    result = _build_index(start, top_n)
    target = pd.Timestamp(date).date()
    match = next((d for d in result.holdings if d.date() == target), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No rebalance on {date}")
    holdings = result.holdings[match]

    prices = get_prices()
    assert prices is not None  # _build_index already required it
    cid1 = calculate_trailing_cid1_cross_section(
        prices[list(holdings.index)], [match], window_days=window_days
    ).xs(match, level="date")

    sectors = get_sectors()
    sector_map: Dict[str, str] = {}
    if sectors is not None and {"symbol", "sector"}.issubset(sectors.columns):
        sector_map = dict(zip(sectors["symbol"], sectors["sector"], strict=True))

    rows: List[HoldingRow] = []
    for symbol, row in holdings.iterrows():
        stats = cid1.loc[symbol] if symbol in cid1.index else None
        rows.append(
            HoldingRow(
                symbol=str(symbol),
                sector=sector_map.get(str(symbol)),
                market_cap=float(row["market_cap"]),
                weight=float(row["weight"]),
                total_return=_clean(stats["total_return"]) if stats is not None else None,
                cost_basis_pain=_clean(stats["cost_basis_pain"]) if stats is not None else None,
                cid1_ratio=_clean(stats["cid1_ratio"]) if stats is not None else None,
                cid1_angle=_clean(stats["cid1_angle"]) if stats is not None else None,
            )
        )
    return HoldingsResponse(date=str(match.date()), window_days=window_days, holdings=rows)


@router.get("/cid1-study", response_model=Cid1StudyResponse)
def cid1_study(
    start: str = "2005-01-01", top_n: int = 500, window_days: int = 252
) -> Cid1StudyResponse:
    """Persistence, IC, Fama-MacBeth, and quantile-sort evidence for Cid-1."""
    study = _build_study(start, top_n, window_days)
    per_date = study["per_date"]

    rows: List[StudyDateRow] = []
    for date, row in per_date.iterrows():
        rows.append(
            StudyDateRow(
                date=str(date.date()),
                n_symbols=int(row["n_symbols"]),
                ic=_clean(row.get("ic")),
                persistence_vs_prev=_clean(row.get("persistence_vs_prev")),
                median_cid1_angle=_clean(row.get("median_cid1_angle")),
                share_pain_free=_clean(row.get("share_pain_free")),
                quantile_spread=_clean(row.get("quantile_spread")),
            )
        )

    quantile_avg = [
        _clean(v) for _, v in study["quantile_mean_returns"].mean(axis=0).sort_index().items()
    ]
    sensitivity = [
        SensitivityRow(
            start_year=int(r["start_year"]),
            n_quarters=int(r["n_quarters"]),
            mean_ic=_clean(r["mean_ic"]),
            ic_tstat=_clean(r["ic_tstat"]),
            mean_spread=_clean(r["mean_spread"]),
            spread_tstat=_clean(r["spread_tstat"]),
        )
        for _, r in study["start_year_sensitivity"].iterrows()
    ]

    return Cid1StudyResponse(
        per_date=rows,
        quantile_avg_forward_returns=quantile_avg,
        ic_summary=_stat(study["ic_summary"]),
        persistence_summary=_stat(study["persistence_summary"]),
        quantile_spread_summary=_stat(study["quantile_spread_summary"]),
        fama_macbeth_univariate=_stat(study["fama_macbeth"]["univariate_cid1"]),
        fama_macbeth_multivariate={
            k: _stat(v) for k, v in study["fama_macbeth"]["multivariate"].items()
        },
        start_year_sensitivity=sensitivity,
        config=study["config"],
    )
