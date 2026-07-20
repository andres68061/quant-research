"""Request/response models for strategy endpoints."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class WalkForwardConfig(BaseModel):
    initial_train_days: int = Field(63, ge=10)
    test_days: int = Field(5, ge=1)
    max_splits: int = Field(50, ge=1)
    label_horizon_days: int = Field(
        1,
        ge=1,
        description=(
            "Forward horizon of the target label in trading days (the shipped "
            "ML features use next-day direction = 1). horizon - 1 rows are "
            "purged from the end of each training window so no training label "
            "overlaps the test window."
        ),
    )
    embargo_days: int = Field(
        0,
        ge=0,
        description="Extra training rows dropped before each test window.",
    )


class InvestedCoverage(BaseModel):
    """How often the factor book held stock exposure vs sat in cash."""

    pct_days_invested: float = Field(..., ge=0, le=1)
    n_days: int = Field(..., ge=0)
    n_days_invested: int = Field(..., ge=0)
    n_days_flat: int = Field(..., ge=0)
    longest_flat_streak_days: int = Field(..., ge=0)
    min_stocks: int = Field(..., ge=1)
    cash_earns_zero: bool = True
    warning: Optional[str] = Field(
        None,
        description=(
            "Present when flat stretches are material. Flat usually means fewer "
            "than min_stocks names had a valid factor on a rebalance date."
        ),
    )


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
    min_stocks: int = Field(
        20,
        ge=2,
        description=(
            "Minimum number of stocks with valid factor values on a given date for "
            "ranking to produce any signal. Dates with fewer valid names yield zero "
            "signals and the portfolio is flat (cash) through them. Lower this to "
            "trade sparser factor panels; raise it to demand more breadth."
        ),
    )
    signal_lag_days: int = Field(
        1,
        ge=0,
        description=(
            "Trading-day lag between factor observation and execution. Default 1 "
            "(standard close-to-close: factor at close(t-1) drives the weight that "
            "earns the return from close(t-1) to close(t)). Set to 0 to reproduce "
            "legacy MOC-style execution (NOT realistic — assumes you can trade at "
            "the same close used to compute the factor)."
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
    label_horizon_days: int = Field(
        1,
        ge=1,
        description=(
            "Forward horizon of the target label in trading days (the shipped "
            "ML features use next-day direction = 1). horizon - 1 rows are "
            "purged from the end of each training window so no training label "
            "overlaps the test window."
        ),
    )
    embargo_days: int = Field(
        0,
        ge=0,
        description="Extra training rows dropped before each test window.",
    )
