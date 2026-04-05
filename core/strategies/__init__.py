"""
Named strategy registry and shared factor backtest runner.

Use :func:`list_strategies` / :func:`get_strategy` for catalogs;
:func:`run_factor_cross_section_backtest` for the cross-sectional factor pipeline.
"""

from core.strategies.factor_runner import run_factor_cross_section_backtest
from core.strategies.registry import STRATEGIES, get_strategy, list_strategies
from core.strategies.types import StrategyKind, StrategyMetadata

__all__ = [
    "STRATEGIES",
    "StrategyKind",
    "StrategyMetadata",
    "get_strategy",
    "list_strategies",
    "run_factor_cross_section_backtest",
]
