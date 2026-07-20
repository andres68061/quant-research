"""Pydantic schemas for the cointegration-persistence pairs index endpoint."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from api.schemas.metrics import EquityCurvePoint, PerformanceMetrics


class PairsPersistentBacktestRequest(BaseModel):
    """Event-driven pairs basket: trade each pair until its cointegration breaks."""

    sector_names: list[str] = Field(..., min_length=1, max_length=10)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    formation_months: int = Field(60, ge=6, le=72, description="Screening lookback window")
    rescreen_months: int = Field(12, ge=3, le=72, description="How often free slots are re-filled")
    top_n_pairs: int = Field(10, ge=1, le=20)
    max_symbols_per_sector: int = Field(12, ge=2, le=15)
    use_adv: bool = Field(True, description="Rank each sector universe by dollar ADV")
    min_crossings: int = Field(3, ge=1, le=50)
    max_adf_pvalue: float = Field(0.05, gt=0.0, le=0.20)
    hedge_window: int = Field(252, ge=30, le=1000)
    zscore_window: int = Field(60, ge=10, le=500)
    entry_z: float = Field(2.0, gt=0.1, le=5.0)
    exit_z: float = Field(0.5, ge=0.0, le=3.0)
    transaction_cost_bps: float = Field(10.0, ge=0.0, le=200.0)
    signal_lag_days: int = Field(1, ge=0, le=5)
    monitor_window: int = Field(252, ge=60, le=1000)
    check_every_days: int = Field(21, ge=5, le=126)
    stop_max_pvalue: float = Field(0.10, gt=0.0, le=0.50)
    persistence_checks: int = Field(4, ge=1, le=12)
    freeze_hedge_in_trade: bool = Field(
        False,
        description="Freeze execution weights at entry instead of re-hedging beta drift daily",
    )


class PairsPersistentPairRow(BaseModel):
    symbol_y: str
    symbol_x: str
    sector: str
    formation_adf_pvalue: float
    formation_crossings: int
    trading_start: str
    stop_date: Optional[str] = None
    stopped_early: bool
    n_days: int


class PairsPersistentScreenRow(BaseModel):
    formation_start: str
    formation_end: str
    active_before: int
    free_slots: int
    n_candidates_found: int
    n_selected: int


class PairsPersistentBacktestResponse(BaseModel):
    metrics: PerformanceMetrics
    equity_curve: list[EquityCurvePoint]
    total_days: int
    screens: list[PairsPersistentScreenRow]
    pair_history: list[PairsPersistentPairRow]
