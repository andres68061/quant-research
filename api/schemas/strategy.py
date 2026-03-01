"""Request/response models for strategy endpoints."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class WalkForwardConfig(BaseModel):
    initial_train_days: int = Field(63, ge=10)
    test_days: int = Field(5, ge=1)
    max_splits: int = Field(50, ge=1)


class BacktestRequest(BaseModel):
    """Parameters for POST /run-backtest."""

    universe: str = Field("SP500", description="'SP500', 'commodities', or 'custom'")
    strategy_type: str = Field(
        "factor", description="'factor', 'ml_directional', 'regime'"
    )
    factor_col: Optional[str] = Field(
        None, description="Factor column for factor-based strategies"
    )
    model_type: Optional[str] = Field(
        None, description="'xgboost', 'rf', 'logistic'"
    )
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    rebalance_freq: str = Field("M", description="'D', 'W', 'M', 'Q'")
    transaction_cost_bps: float = Field(10.0, ge=0)
    top_pct: float = Field(0.20, gt=0, le=1)
    bottom_pct: float = Field(0.20, ge=0, le=1)
    long_only: bool = False
    walkforward: Optional[WalkForwardConfig] = None


class MLStrategyRequest(BaseModel):
    """Parameters for POST /run-ml-strategy."""

    symbol: str
    model_type: str = Field("xgboost", description="'xgboost' or 'lstm'")
    initial_train_days: int = Field(63, ge=10)
    test_days: int = Field(5, ge=1)
    max_splits: int = Field(50, ge=1)
