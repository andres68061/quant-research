"""Pydantic schemas for pairs / cointegration strategy endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from api.schemas.metrics import EquityCurvePoint, PerformanceMetrics


class PairsBacktestRequest(BaseModel):
    """Request body for a single-pair cointegration mean-reversion backtest."""

    symbol_y: str = Field(..., description="Dependent leg (spread long = long this)")
    symbol_x: str = Field(..., description="Hedge leg")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    hedge_window: int = Field(252, ge=30, le=1000)
    zscore_window: int = Field(60, ge=10, le=500)
    entry_z: float = Field(2.0, gt=0.1, le=5.0)
    exit_z: float = Field(0.5, ge=0.0, le=3.0)
    transaction_cost_bps: float = Field(10.0, ge=0.0, le=200.0)
    signal_lag_days: int = Field(1, ge=0, le=5)
    train_frac: Optional[float] = Field(
        None,
        ge=0.2,
        le=0.8,
        description=(
            "When set, enforces train/held-out separation: splits "
            "[start_date, end_date] at this fraction, computes the "
            "cointegration diagnostic on the train slice only, and computes "
            "every returned metric/curve on the held-out slice only. When "
            "omitted (default), behaves exactly as before — a single "
            "backtest over the full [start_date, end_date] range, which is "
            "correct only if you did not choose this pair or this date "
            "range using knowledge of how it performs over it."
        ),
    )


class EngleGrangerDiagnostics(BaseModel):
    hedge_ratio: float
    intercept: float
    adf_stat: float
    adf_pvalue: float
    n_obs: float


class PairsDiagnostics(BaseModel):
    symbol_y: str
    symbol_x: str
    engle_granger: EngleGrangerDiagnostics
    hedge_window: int
    zscore_window: int
    entry_z: float
    exit_z: float
    transaction_cost: float
    signal_lag_days: int
    n_days: int
    pct_days_in_trade: float


class SpreadPoint(BaseModel):
    date: str
    zscore: float
    position: float


class PairsBacktestResponse(BaseModel):
    metrics: PerformanceMetrics
    equity_curve: list[EquityCurvePoint]
    total_days: int
    diagnostics: PairsDiagnostics
    spread_series: list[SpreadPoint]
    is_held_out: bool = Field(
        False,
        description=(
            "True when train_frac was set: metrics/equity_curve/diagnostics/"
            "spread_series above describe the held-out slice ONLY."
        ),
    )
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    held_out_start_date: Optional[str] = None
    train_diagnostics: Optional[EngleGrangerDiagnostics] = Field(
        None,
        description="Engle-Granger test on the train slice only (held-out mode).",
    )


class PairsScreenRequest(BaseModel):
    """Walk-forward screen over a sector or an explicit symbol list."""

    sector: Optional[str] = Field(
        None, description="Exact sector label from sector_classifications"
    )
    symbols: Optional[list[str]] = Field(
        None, description="Explicit tickers; overrides sector when provided"
    )
    method: str = Field(
        "gatev",
        description="gatev (SSD formation) or engle_granger (train EG filter)",
    )
    use_adv: bool = Field(True, description="Rank sector universe by dollar ADV when available")
    max_symbols: int = Field(10, ge=2, le=15)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    train_frac: float = Field(0.67, ge=0.2, le=0.85)
    min_train_corr: float = Field(0.5, ge=0.0, le=0.99)
    max_train_adf_pvalue: float = Field(0.05, gt=0.0, le=0.2)
    max_oos_backtests: int = Field(15, ge=1, le=40)
    hedge_window: int = Field(252, ge=30, le=1000)
    zscore_window: int = Field(60, ge=10, le=500)
    entry_z: float = Field(2.0, gt=0.1, le=5.0)
    exit_z: float = Field(0.5, ge=0.0, le=3.0)
    transaction_cost_bps: float = Field(10.0, ge=0.0, le=200.0)


class PairsScreenRow(BaseModel):
    symbol_y: str
    symbol_x: str
    formation_ssd: Optional[float] = None
    train_corr: Optional[float] = None
    train_adf_pvalue: Optional[float] = None
    train_hedge_ratio: Optional[float] = None
    oos_sharpe: float
    oos_annualized_return: float
    oos_max_drawdown: float
    oos_n_days: int
    oos_pct_days_in_trade: float


class PairsScreenResponse(BaseModel):
    symbols: list[str]
    split_date: str
    train_frac: float
    method: str = "engle_granger"
    n_pairs_tested: int
    n_pairs_passed_train: int
    results: list[PairsScreenRow]
