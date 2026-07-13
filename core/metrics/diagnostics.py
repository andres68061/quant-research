"""Assemble backtest risk diagnostics for API / UI consumption.

Pure functions over a net-return series: rolling Sharpe/Sortino/vol, drawdown,
return histogram, and three VaR/CVaR methodologies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.backtest.portfolio import calculate_rolling_metrics
from core.metrics.performance import calculate_drawdown
from core.metrics.risk import calculate_all_var

__all__ = ["build_backtest_diagnostics"]


def build_backtest_diagnostics(
    net_returns: pd.Series,
    *,
    rolling_window: int = 63,
    histogram_bins: int = 40,
    var_confidence: int = 95,
) -> dict[str, Any]:
    """
    Build a JSON-serializable diagnostics payload from daily net returns.

    Args:
        net_returns: Strategy daily net returns (decimal).
        rolling_window: Trailing window for rolling Sharpe/Sortino/vol.
        histogram_bins: Number of bins for the return distribution.
        var_confidence: VaR/CVaR confidence level (e.g. 95).

    Returns:
        Dict with keys ``rolling``, ``drawdown``, ``histogram``, ``var``.
    """
    clean = net_returns.dropna()
    if clean.empty:
        return {
            "rolling": [],
            "drawdown": [],
            "histogram": {"bin_edges": [], "counts": []},
            "var": {},
            "rolling_window": rolling_window,
            "var_confidence": var_confidence,
        }

    window = min(rolling_window, max(len(clean) // 2, 5))
    rolling = calculate_rolling_metrics(clean, window=window)
    rolling_points = [
        {
            "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
            "sharpe": _finite_or_none(row.sharpe_ratio),
            "sortino": _finite_or_none(row.sortino_ratio),
            "volatility": _finite_or_none(row.annualized_volatility),
        }
        for idx, row in rolling.iterrows()
        if pd.notna(row.annualized_volatility)
    ]

    dd = calculate_drawdown(clean)
    drawdown_points = [
        {
            "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
            "drawdown": float(val),
        }
        for idx, val in dd.items()
        if pd.notna(val)
    ]

    counts, edges = np.histogram(clean.to_numpy(dtype=float), bins=histogram_bins)
    histogram = {
        "bin_edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }

    var_block = calculate_all_var(clean.to_numpy(dtype=float), confidence=var_confidence)

    return {
        "rolling": rolling_points,
        "drawdown": drawdown_points,
        "histogram": histogram,
        "var": var_block,
        "rolling_window": window,
        "var_confidence": var_confidence,
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
