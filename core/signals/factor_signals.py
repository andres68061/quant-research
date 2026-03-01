"""
Factor-based signal generation.

Thin convenience layer that delegates to ``core.backtest.portfolio`` for
the actual ranking logic but provides a clean signal-oriented API.
"""

import pandas as pd

from core.backtest.portfolio import create_signals_from_factor


def long_short_from_factor(
    factors_df: pd.DataFrame,
    factor_col: str,
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
    min_stocks: int = 20,
) -> pd.DataFrame:
    """
    Generate long/short signals by ranking on a factor.

    Args:
        factors_df: MultiIndex (date, symbol) DataFrame with factor columns
        factor_col: Column to rank on
        top_pct: Fraction of universe to go long
        bottom_pct: Fraction of universe to short
        min_stocks: Minimum valid stocks per date

    Returns:
        DataFrame with 'signal' column (1 / -1 / 0)
    """
    return create_signals_from_factor(
        factors_df,
        factor_col,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        long_only=False,
        min_stocks=min_stocks,
    )


def long_only_from_factor(
    factors_df: pd.DataFrame,
    factor_col: str,
    top_pct: float = 0.20,
    min_stocks: int = 20,
) -> pd.DataFrame:
    """
    Generate long-only signals (top *top_pct* by factor).

    Args:
        factors_df: MultiIndex (date, symbol) DataFrame with factor columns
        factor_col: Column to rank on
        top_pct: Fraction of universe to go long
        min_stocks: Minimum valid stocks per date

    Returns:
        DataFrame with 'signal' column (1 / 0)
    """
    return create_signals_from_factor(
        factors_df,
        factor_col,
        top_pct=top_pct,
        bottom_pct=0.0,
        long_only=True,
        min_stocks=min_stocks,
    )
