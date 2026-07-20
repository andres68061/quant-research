"""Invested-coverage diagnostics for factor / portfolio backtests.

When fewer than ``min_stocks`` names have a valid factor on a rebalance date,
``create_signals_from_factor`` emits zeros and the book goes to cash (0% return
in this simulator). These helpers make that visible so flat equity stretches
are not mistaken for a regime filter or a silent bug.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

__all__ = ["summarize_invested_coverage", "position_label_from_counts"]


def position_label_from_counts(n_long: int, n_short: int) -> str:
    """Map daily long/short headcounts to a replay position label."""
    if n_long > 0 and n_short > 0:
        return "long_short"
    if n_long > 0:
        return "long"
    if n_short > 0:
        return "short"
    return "flat"


def summarize_invested_coverage(
    n_long: pd.Series,
    n_short: pd.Series,
    *,
    min_stocks: int,
) -> Dict[str, Any]:
    """
    Summarize how often the portfolio held any stock exposure.

    Args:
        n_long: Daily count of long names (from ``calculate_portfolio_returns``).
        n_short: Daily count of short names.
        min_stocks: Threshold used when building signals (for disclosure copy).

    Returns:
        Dict with ``pct_days_invested``, flat-day counts, longest flat streak,
        and an optional ``warning`` when material cash stretches exist.

    Example:
        >>> cov = summarize_invested_coverage(n_long, n_short, min_stocks=20)
        >>> cov["cash_earns_zero"]
        True
    """
    if min_stocks < 1:
        raise ValueError(f"min_stocks must be >= 1, got {min_stocks}")

    long_counts = n_long.fillna(0).astype(int)
    short_counts = n_short.reindex(long_counts.index).fillna(0).astype(int)
    invested = (long_counts + short_counts) > 0

    n_days = int(len(invested))
    n_invested = int(invested.sum())
    n_flat = n_days - n_invested
    pct_days_invested = float(n_invested / n_days) if n_days > 0 else 0.0
    longest_flat = _longest_true_streak(~invested)

    warning: Optional[str] = None
    if n_flat > 0 and pct_days_invested < 0.95:
        pct_flat = 100.0 * (1.0 - pct_days_invested)
        warning = (
            f"Portfolio was flat (cash, 0% return) on {n_flat}/{n_days} days "
            f"({pct_flat:.1f}%). Flat stretches usually mean fewer than "
            f"min_stocks={min_stocks} names had a valid factor value on a "
            f"rebalance date — common for sparse fundamentals factors. "
            f"Longest flat streak: {longest_flat} trading days. Cash earns "
            f"0% in this simulator (not T-bills)."
        )

    return {
        "pct_days_invested": pct_days_invested,
        "n_days": n_days,
        "n_days_invested": n_invested,
        "n_days_flat": n_flat,
        "longest_flat_streak_days": longest_flat,
        "min_stocks": int(min_stocks),
        "cash_earns_zero": True,
        "warning": warning,
    }


def _longest_true_streak(mask: pd.Series) -> int:
    longest = 0
    current = 0
    for is_flat in mask.tolist():
        if is_flat:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)
