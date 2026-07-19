"""Response models for performance metrics."""

from typing import Optional

from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    pain_index: float = 0.0
    pain_ratio: float = 0.0
    ulcer_index: float = 0.0
    martin_ratio: float = 0.0
    cid1_ratio: float = 0.0
    typical_period_return: float = 0.0
    cid2_ratio: float = 0.0
    n_periods: int
    information_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None


class VarResult(BaseModel):
    var: float
    cvar: float


class AllVarResult(BaseModel):
    historical: VarResult
    parametric: VarResult
    monte_carlo: VarResult
    confidence: int


class EquityCurvePoint(BaseModel):
    date: str
    cumulative_return: float = Field(
        ...,
        description="Cumulative net return since the first point (decimal), i.e. wealth_index - 1",
    )


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
