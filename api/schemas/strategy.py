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
    strategy_type: str = Field("factor", description="'factor', 'ml_directional', 'regime'")
    factor_col: Optional[str] = Field(None, description="Factor column for factor-based strategies")
    model_type: Optional[str] = Field(None, description="'xgboost', 'rf', 'logistic'")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    rebalance_freq: str = Field(
        "ME",
        description="Pandas resample offset: 'D', 'W', 'ME' (month end), 'QE' (quarter end). "
        "Legacy 'M'/'Q' are accepted and normalized server-side.",
    )
    transaction_cost_bps: float = Field(10.0, ge=0)
    top_pct: float = Field(0.20, gt=0, le=1)
    bottom_pct: float = Field(0.20, ge=0, le=1)
    long_only: bool = False
    survivorship_free: bool = Field(
        True,
        description=(
            "When True (default), restrict the investable universe at each rebalance "
            "date to stocks that were in the S&P 500 on that date, eliminating "
            "survivorship bias. Set False for legacy behavior (full panel, no filter)."
        ),
    )
    walkforward: Optional[WalkForwardConfig] = None


class MLStrategyRequest(BaseModel):
    """Parameters for POST /run-ml-strategy."""

    symbol: str
    model_type: str = Field(
        "xgboost",
        description="'xgboost', 'random_forest', 'logistic', or 'lstm'",
    )
    initial_train_days: int = Field(63, ge=10)
    test_days: int = Field(5, ge=1)
    max_splits: int = Field(50, ge=1)
