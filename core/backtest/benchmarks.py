"""
Benchmark return computation.

Builds benchmark return series for comparison against portfolio strategies.
Supports S&P 500 (live and reconstructed), equal-weight universe, and
custom synthetic mixes.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_component_returns(
    component_name: str, df_prices: pd.DataFrame
) -> pd.Series:
    """Resolve a named component to a return series."""
    symbol_map = {
        "S&P 500": "^GSPC",
        "S&P 500 (^GSPC)": "^GSPC",
        "NASDAQ Composite": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
    }
    ticker = symbol_map.get(component_name)
    if ticker and ticker in df_prices.columns:
        return df_prices[ticker].pct_change()
    return df_prices.pct_change().mean(axis=1)


def calculate_benchmark_returns(
    benchmark_type: str,
    df_prices: pd.DataFrame,
    component1: Optional[str] = None,
    component2: Optional[str] = None,
    weight1: float = 60.0,
    sp500_weighting: str = "Equal Weight",
) -> Tuple[pd.Series, str]:
    """
    Calculate benchmark returns based on type.

    Args:
        benchmark_type: Type of benchmark (e.g. 'S&P 500 (^GSPC)',
            'S&P 500 Reconstructed (2020+)', 'Equal Weight Universe',
            'Synthetic (Custom Mix)')
        df_prices: Wide-format price DataFrame (date x symbols)
        component1: First component for synthetic benchmark
        component2: Second component for synthetic benchmark
        weight1: Weight for component1 as percentage (0--100)
        sp500_weighting: 'Equal Weight' or 'Cap-Weighted' for reconstructed S&P 500

    Returns:
        Tuple of (benchmark_returns Series, benchmark_name string)
    """
    if benchmark_type == "S&P 500 (^GSPC)":
        if "^GSPC" in df_prices.columns:
            return df_prices["^GSPC"].pct_change(), "S&P 500 (^GSPC)"
        return (
            df_prices.pct_change().mean(axis=1),
            "Equal Weight Universe (S&P 500 not available)",
        )

    if benchmark_type == "S&P 500 Reconstructed (2020+)":
        return _reconstructed_sp500(df_prices, sp500_weighting)

    if benchmark_type == "Equal Weight Universe":
        return df_prices.pct_change().mean(axis=1), "Equal Weight Universe"

    if benchmark_type == "Synthetic (Custom Mix)":
        returns1 = _get_component_returns(component1 or "S&P 500", df_prices)
        returns2 = _get_component_returns(component2 or "S&P 500", df_prices)
        w1 = weight1 / 100.0
        returns = w1 * returns1 + (1 - w1) * returns2
        name = (
            f"{int(weight1)}% {component1} + {int(100 - weight1)}% {component2}"
        )
        return returns, name

    return df_prices.pct_change().mean(axis=1), "Equal Weight All Stocks"


def _reconstructed_sp500(
    df_prices: pd.DataFrame,
    weighting: str,
) -> Tuple[pd.Series, str]:
    """Build a point-in-time reconstructed S&P 500 benchmark."""
    try:
        from core.data.sp500_constituents import SP500Constituents

        sp500 = SP500Constituents()
        sp500.load()
        all_returns = df_prices.pct_change()

        if weighting == "Equal Weight":
            daily_returns = []
            for date in df_prices.index:
                constituents = sp500.get_constituents_on_date(pd.Timestamp(date))
                available = [c for c in constituents if c in all_returns.columns]
                if available:
                    daily_returns.append(all_returns.loc[date, available].mean())
                else:
                    daily_returns.append(np.nan)

            returns = pd.Series(daily_returns, index=df_prices.index)

            all_constituents: set = set()
            for date in df_prices.index[::252]:
                all_constituents.update(
                    sp500.get_constituents_on_date(pd.Timestamp(date))
                )
            name = f"S&P 500 Reconstructed (EW, ~{len(all_constituents)} tickers)"
            return returns, name

        # Cap-weighted path
        from core.data.market_caps import MarketCapCalculator

        calc = MarketCapCalculator()
        market_caps = calc.load_market_caps()

        if market_caps is None or market_caps.empty:
            logger.warning("Market cap data unavailable, falling back to EW.")
            return _reconstructed_sp500(df_prices, "Equal Weight")

        daily_returns = []
        for date in df_prices.index:
            constituents = sp500.get_constituents_on_date(pd.Timestamp(date))
            date_ts = pd.Timestamp(date).tz_localize(None)
            try:
                date_caps = calc.get_market_cap_on_date(date_ts, constituents)
                available = list(date_caps.index)
                if available:
                    weights = date_caps / date_caps.sum()
                    day_ret = (all_returns.loc[date, available] * weights).sum()
                    daily_returns.append(day_ret)
                else:
                    daily_returns.append(np.nan)
            except Exception:
                daily_returns.append(np.nan)

        returns = pd.Series(daily_returns, index=df_prices.index)
        return returns, "S&P 500 Reconstructed (Cap-Weighted)"

    except Exception as exc:
        logger.warning("Reconstructed S&P 500 unavailable: %s", exc)
        if "^GSPC" in df_prices.columns:
            return df_prices["^GSPC"].pct_change(), "S&P 500 (^GSPC)"
        return df_prices.pct_change().mean(axis=1), "Equal Weight Universe"
