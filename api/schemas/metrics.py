"""Response models for performance metrics."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class PerformanceMetrics(BaseModel):
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
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
    cumulative_return: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
