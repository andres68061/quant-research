"""
Benchmark return endpoints.

Exposes benchmark calculation logic for portfolio comparison.
"""

from datetime import date
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_prices
from core.backtest.benchmarks import calculate_benchmark_returns
from core.metrics.performance import calculate_performance_metrics

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


class BenchmarkResponse(BaseModel):
    """Benchmark return series with performance metrics."""

    benchmark_name: str = Field(..., description="Descriptive name of the benchmark")
    dates: list[str] = Field(..., description="ISO date strings")
    returns: list[float] = Field(..., description="Daily returns")
    cumulative_returns: list[float] = Field(
        ..., description="Cumulative return series (1 + cumulative)"
    )
    total_return: float = Field(..., description="Total return over period")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio (assuming 0% RF)")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    calmar_ratio: float = Field(..., description="Calmar ratio")


@router.get("/returns", response_model=BenchmarkResponse)
def get_benchmark_returns(
    benchmark_type: str = Query(
        ...,
        description=(
            "Benchmark type: 'S&P 500 (^GSPC)', 'S&P 500 Reconstructed (2020+)', "
            "'Equal Weight Universe', or 'Synthetic (Custom Mix)'"
        ),
    ),
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    sp500_weighting: str = Query(
        "Equal Weight",
        description="For reconstructed S&P 500: 'Equal Weight' or 'Cap-Weighted'",
    ),
    component1: Optional[str] = Query(
        None,
        description="First component for synthetic benchmark",
    ),
    component2: Optional[str] = Query(
        None,
        description="Second component for synthetic benchmark",
    ),
    weight1: float = Query(
        60.0,
        ge=0.0,
        le=100.0,
        description="Weight for component1 as percentage (0-100)",
    ),
) -> BenchmarkResponse:
    """Calculate benchmark returns for a given period.

    Supports multiple benchmark types including live S&P 500 data,
    point-in-time reconstructed constituents, equal-weight universe,
    and custom synthetic blends.
    """
    df_prices = get_prices()
    if df_prices is None or df_prices.empty:
        raise HTTPException(status_code=503, detail="Price data not loaded")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df_slice = df_prices.loc[start_ts:end_ts]

    if df_slice.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No price data for period {start_date} to {end_date}",
        )

    try:
        benchmark_returns, benchmark_name = calculate_benchmark_returns(
            benchmark_type=benchmark_type,
            df_prices=df_slice,
            component1=component1,
            component2=component2,
            weight1=weight1,
            sp500_weighting=sp500_weighting,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark calculation failed: {exc}",
        ) from exc

    benchmark_returns = benchmark_returns.fillna(0.0)
    cumulative = (1 + benchmark_returns).cumprod()

    metrics = calculate_performance_metrics(
        returns=benchmark_returns,
        benchmark_returns=None,
        risk_free_rate=0.0,
    )

    return BenchmarkResponse(
        benchmark_name=benchmark_name,
        dates=[d.strftime("%Y-%m-%d") for d in benchmark_returns.index],
        returns=benchmark_returns.tolist(),
        cumulative_returns=cumulative.tolist(),
        total_return=metrics["total_return"],
        annualized_return=metrics["annualized_return"],
        volatility=metrics["annualized_volatility"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        calmar_ratio=metrics["calmar_ratio"],
    )
