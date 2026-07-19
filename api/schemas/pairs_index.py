"""Pydantic schemas for the rolling multi-pair stat-arb index endpoint."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from api.schemas.metrics import EquityCurvePoint, PerformanceMetrics


class PairsIndexBacktestRequest(BaseModel):
    """Roll a same-sector pairs basket forward and blend returns into one index."""

    sector_names: list[str] = Field(..., min_length=1, max_length=10)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    formation_months: int = Field(12, ge=3, le=36)
    trading_months: int = Field(6, ge=1, le=12)
    top_n_pairs: int = Field(10, ge=1, le=20)
    max_symbols_per_sector: int = Field(12, ge=2, le=15)
    use_adv: bool = Field(True, description="Rank each sector universe by dollar ADV")
    hedge_window: int = Field(252, ge=30, le=1000)
    zscore_window: int = Field(60, ge=10, le=500)
    entry_z: float = Field(2.0, gt=0.1, le=5.0)
    exit_z: float = Field(0.5, ge=0.0, le=3.0)
    transaction_cost_bps: float = Field(10.0, ge=0.0, le=200.0)
    signal_lag_days: int = Field(1, ge=0, le=5)


class PairsIndexPairRow(BaseModel):
    symbol_y: str
    symbol_x: str
    sector: str
    formation_ssd: float
    formation_adf_pvalue: Optional[float] = None
    period_sharpe: float
    period_n_days: int


class PairsIndexPeriodRow(BaseModel):
    formation_start: str
    formation_end: str
    trading_start: str
    trading_end: str
    n_candidates_formed: int
    n_pairs_selected: int
    avg_active_pairs: float
    blended_sharpe: Optional[float] = None
    selected_pairs: list[PairsIndexPairRow]


class PairsIndexBacktestResponse(BaseModel):
    metrics: PerformanceMetrics
    equity_curve: list[EquityCurvePoint]
    total_days: int
    universe: list[str]
    periods: list[PairsIndexPeriodRow]
