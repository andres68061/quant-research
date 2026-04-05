"""
Portfolio simulation and backtesting utilities.

This module provides functions for simulating portfolio strategies,
calculating returns with transaction costs, and managing portfolio positions.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pandas 2.2+ deprecates legacy resample aliases (e.g. M -> ME). Map before resample().
_LEGACY_RESAMPLE_FREQ: dict[str, str] = {
    "M": "ME",
    "Q": "QE",
    "A": "YE",
    "Y": "YE",
}


def _infer_abs_bound(factor_col: str) -> float:
    """
    Default maximum absolute factor value for outlier filtering before ranking.

    Momentum columns are cumulative-return style (clip extreme names). Volatility
    uses annualized vol scale. Beta and log market cap rely on ``notna()`` and
    ``isfinite()`` only (no absolute cap).

    Args:
        factor_col: Column name in the factor panel (e.g. ``mom_12_1``, ``beta_60d``).

    Returns:
        Upper bound on ``abs(factor)`` for a row to count as valid; ``inf`` means
        no cap beyond non-finite exclusion.

    Example:
        >>> _infer_abs_bound("mom_12_1")
        10.0
        >>> _infer_abs_bound("beta_60d") == float("inf")
        True
    """
    if factor_col == "log_market_cap":
        return float("inf")
    if factor_col.startswith("mom_"):
        return 10.0
    if factor_col.startswith("vol_"):
        return 5.0
    if factor_col.startswith("beta_"):
        return float("inf")
    return 10.0


def _normalize_rebalance_freq(freq: str) -> str:
    """
    Map deprecated pandas offset strings to current resample aliases.

    Args:
        freq: User or API input (e.g. ``M``, ``ME``, ``W``, ``D``).

    Returns:
        String safe for ``Series.resample`` without FutureWarning for known legacy forms.

    Example:
        >>> _normalize_rebalance_freq("M")
        'ME'
    """
    if freq in _LEGACY_RESAMPLE_FREQ:
        return _LEGACY_RESAMPLE_FREQ[freq]
    return freq


def create_signals_from_factor(
    factors_df: pd.DataFrame,
    factor_col: str,
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
    long_only: bool = False,
    min_stocks: int = 20,
    universe_filter: Optional[Callable[[pd.Timestamp], set[str]]] = None,
    max_abs_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Create long/short signals based on a factor.

    Args:
        factors_df: DataFrame with MultiIndex (date, symbol) and factor columns
        factor_col: Factor column to rank on
        top_pct: Percentage of stocks to go long (0.20 = top 20%)
        bottom_pct: Percentage of stocks to short (0.20 = bottom 20%)
        long_only: If True, only create long signals
        min_stocks: Minimum number of valid stocks required per date
        universe_filter: Optional callable ``(date) -> set[str]`` that returns the
            set of symbols eligible for trading on that date. Stocks outside the
            returned set receive signal = 0 regardless of their factor value. Use
            :func:`sp500_universe_filter` for survivorship-bias-free S&P 500 backtests.
        max_abs_value: Maximum ``abs(factor)`` for a row to be valid; if ``None``,
            inferred from ``factor_col`` via :func:`_infer_abs_bound`.

    Returns:
        DataFrame with 'signal' column: 1 (long), -1 (short), 0 (neutral)

    Example:
        >>> signals = create_signals_from_factor(
        ...     factors_df, 'mom_12_1', top_pct=0.20, bottom_pct=0.20
        ... )
    """
    df = factors_df.copy()
    df["signal"] = 0

    bound = max_abs_value if max_abs_value is not None else _infer_abs_bound(factor_col)
    valid = df[factor_col].notna() & np.isfinite(df[factor_col]) & (df[factor_col].abs() < bound)
    valid_df = df.loc[valid, [factor_col]].copy()

    if universe_filter is not None:
        dates = valid_df.index.get_level_values("date").unique()
        mask_parts = []
        for dt in dates:
            eligible = universe_filter(dt)
            dt_slice = valid_df.loc[dt]
            symbols_in = dt_slice.index.get_level_values("symbol")
            keep = symbols_in.isin(eligible)
            idx = dt_slice.index[keep]
            mask_parts.append(
                pd.MultiIndex.from_arrays(
                    [[dt] * len(idx), idx.get_level_values("symbol")],
                    names=["date", "symbol"],
                )
            )
        if mask_parts:
            eligible_idx = mask_parts[0].append(mask_parts[1:])
            valid_df = valid_df.loc[valid_df.index.isin(eligible_idx)]
        else:
            valid_df = valid_df.iloc[:0]

    grp = valid_df.groupby(level="date")[factor_col]
    counts = grp.transform("count")

    enough = counts >= min_stocks
    insufficient = ~enough
    if insufficient.any():
        dropped_dates = valid_df.index.get_level_values("date")[insufficient].unique().tolist()
        preview = dropped_dates[:20]
        suffix = " ..." if len(dropped_dates) > 20 else ""
        logger.debug(
            "create_signals_from_factor(%s): %d dates dropped (valid count < min_stocks=%d): "
            "%s%s",
            factor_col,
            len(dropped_dates),
            min_stocks,
            preview,
            suffix,
        )
    valid_df = valid_df.loc[enough]
    counts = counts.loc[enough]

    ranks = valid_df.groupby(level="date")[factor_col].rank(ascending=False, method="first")
    n_long = (counts * top_pct).clip(lower=1).astype(int)

    df.loc[ranks[ranks <= n_long].index, "signal"] = 1

    if not long_only:
        n_short = (counts * bottom_pct).clip(lower=1).astype(int)
        df.loc[ranks[ranks > (counts - n_short)].index, "signal"] = -1

    return df


def sp500_universe_filter() -> Callable[[pd.Timestamp], set[str]]:
    """
    Return a callable that maps a date to the set of S&P 500 members on that date.

    Uses :class:`~core.data.sp500_constituents.SP500Constituents` with the default
    historical CSV. The constituents are loaded once; subsequent calls are a dict
    lookup (fast).

    Returns:
        A function ``(pd.Timestamp) -> set[str]`` suitable for the
        ``universe_filter`` parameter of :func:`create_signals_from_factor`.

    Example:
        >>> uf = sp500_universe_filter()
        >>> members = uf(pd.Timestamp("2020-06-15"))
        >>> "AAPL" in members
        True
    """
    from core.data.sp500_constituents import SP500Constituents

    sp500 = SP500Constituents()
    sp500.load()
    _cache: dict[str, set[str]] = {}

    def _filter(date: pd.Timestamp) -> set[str]:
        key = str(date.date()) if hasattr(date, "date") else str(date)
        if key not in _cache:
            members = sp500.get_constituents_on_date(date)
            _cache[key] = set(members) if members else set()
        return _cache[key]

    return _filter


def calculate_portfolio_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_freq: str = "ME",
    transaction_cost: float = 0.001,
    long_only: bool = False,
) -> pd.DataFrame:
    """
    Calculate portfolio returns from signals and prices.

    Handles delistings, rebalancing on specified dates, and transaction costs.

    Args:
        signals: DataFrame with 'signal' column (MultiIndex: date, symbol)
        prices: DataFrame with prices (wide format: date x symbols)
        rebalance_freq: Pandas offset for resampling (prefer ``ME``, ``QE``; ``M``/``Q``
            are normalized to ``ME``/``QE`` for pandas 2.2+ compatibility).
        transaction_cost: Cost per trade as decimal (0.001 = 10 bps)
        long_only: If True, ignore short signals

    Returns:
        DataFrame with columns: gross_return, transaction_cost,
        net_return, turnover, n_long, n_short, cash

    Example:
        >>> results = calculate_portfolio_returns(
        ...     signals, prices, rebalance_freq='ME', transaction_cost=0.001
        ... )
    """
    returns = prices.pct_change()

    signals_wide = signals["signal"].unstack(fill_value=0)

    common_dates = returns.index.intersection(signals_wide.index)
    returns = returns.loc[common_dates]
    signals_wide = signals_wide.loc[common_dates]

    freq = _normalize_rebalance_freq(rebalance_freq)
    rebalance_dates = returns.resample(freq).last().index

    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    cash_position = pd.Series(0.0, index=returns.index)

    for i, rebal_date in enumerate(rebalance_dates):
        if rebal_date not in signals_wide.index:
            continue

        sigs = signals_wide.loc[rebal_date]
        if long_only:
            sigs = sigs.clip(lower=0)

        available_stocks = prices.loc[rebal_date].dropna().index
        sigs = sigs[sigs.index.isin(available_stocks)]

        n_long = (sigs > 0).sum()
        n_short = (sigs < 0).sum()

        weights = pd.Series(0.0, index=sigs.index)
        if n_long > 0:
            weights[sigs > 0] = 1.0 / n_long
        if n_short > 0 and not long_only:
            weights[sigs < 0] = -1.0 / n_short

        if i + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[i + 1]
        else:
            next_rebal = returns.index[-1] + pd.Timedelta(days=1)

        hold_dates = returns.index[(returns.index >= rebal_date) & (returns.index < next_rebal)]

        for col in weights.index:
            if col in positions.columns:
                positions.loc[hold_dates, col] = weights[col]

    daily_gross_returns = []
    daily_transaction_costs = []
    daily_cash = []
    daily_n_long = []
    daily_n_short = []
    daily_turnover = []

    for i, date in enumerate(returns.index):
        if i == 0:
            daily_gross_returns.append(0.0)
            daily_transaction_costs.append(0.0)
            daily_cash.append(0.0)
            daily_n_long.append((positions.loc[date] > 0).sum())
            daily_n_short.append((positions.loc[date] < 0).sum())
            daily_turnover.append(0.0)
            continue

        prev_date = returns.index[i - 1]
        pos = positions.loc[prev_date]
        ret = returns.loc[date]

        cash_from_delistings = 0.0
        for symbol in pos[pos != 0].index:
            if pd.isna(ret[symbol]) or symbol not in ret.index:
                cash_from_delistings += abs(pos[symbol])
                positions.loc[date:, symbol] = 0.0

        valid_returns = ret[pos.index].fillna(0.0)
        gross_ret = (pos * valid_returns).sum()

        cash = cash_position.loc[prev_date] + cash_from_delistings
        if date in rebalance_dates:
            cash = 0.0

        pos_change = (positions.loc[date] - positions.loc[prev_date]).abs().sum()
        turnover_val = pos_change / 2
        trans_cost = turnover_val * transaction_cost

        daily_gross_returns.append(gross_ret)
        daily_transaction_costs.append(trans_cost)
        daily_cash.append(cash)
        daily_n_long.append((positions.loc[date] > 0).sum())
        daily_n_short.append((positions.loc[date] < 0).sum())
        daily_turnover.append(turnover_val)

        cash_position.loc[date] = cash

    gross_returns = pd.Series(daily_gross_returns, index=returns.index)
    transaction_costs = pd.Series(daily_transaction_costs, index=returns.index)
    net_returns = gross_returns - transaction_costs

    return pd.DataFrame(
        {
            "gross_return": gross_returns,
            "transaction_cost": transaction_costs,
            "net_return": net_returns,
            "turnover": daily_turnover,
            "n_long": daily_n_long,
            "n_short": daily_n_short,
            "cash": cash_position,
        }
    )


def create_weighted_portfolio(
    prices: pd.DataFrame,
    symbols: list,
    weighting_scheme: str = "equal",
    manual_weights: Optional[dict] = None,
    share_counts: Optional[dict] = None,
    rebalance_freq: str = "ME",
) -> pd.Series:
    """
    Create portfolio returns with various weighting schemes.

    Args:
        prices: DataFrame with prices (wide format: date x symbols)
        symbols: List of symbols to include
        weighting_scheme: One of 'equal', 'manual', 'cap', 'shares', 'harmonic'
        manual_weights: Dict of {symbol: weight} for manual weighting
        share_counts: Dict of {symbol: shares} for share-based weighting
        rebalance_freq: Pandas offset (prefer ``ME``, ``QE``; legacy ``M``/``Q`` normalized).

    Returns:
        Series of portfolio returns

    Example:
        >>> returns = create_weighted_portfolio(prices, ['AAPL', 'MSFT'], 'equal')
    """
    prices_selected = prices[symbols].copy()
    returns = prices_selected.pct_change()
    freq = _normalize_rebalance_freq(rebalance_freq)
    rebalance_dates = returns.resample(freq).last().index

    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

    for i, rebal_date in enumerate(rebalance_dates):
        if rebal_date not in prices_selected.index:
            continue

        current_prices = prices_selected.loc[rebal_date]

        if weighting_scheme == "equal":
            stock_weights = pd.Series(1.0 / len(symbols), index=symbols)

        elif weighting_scheme == "manual":
            if manual_weights is None:
                raise ValueError("manual_weights required for manual weighting")
            stock_weights = pd.Series(manual_weights)
            stock_weights = stock_weights / stock_weights.sum()

        elif weighting_scheme == "cap":
            stock_weights = current_prices / current_prices.sum()

        elif weighting_scheme == "shares":
            if share_counts is None:
                raise ValueError("share_counts required for share-based weighting")
            shares = pd.Series(share_counts)
            dollar_values = shares * current_prices
            stock_weights = dollar_values / dollar_values.sum()

        elif weighting_scheme == "harmonic":
            inverse_prices = 1.0 / current_prices
            stock_weights = inverse_prices / inverse_prices.sum()

        else:
            raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")

        if i + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[i + 1]
        else:
            next_rebal = returns.index[-1] + pd.Timedelta(days=1)

        hold_dates = returns.index[(returns.index >= rebal_date) & (returns.index < next_rebal)]

        for symbol in symbols:
            if symbol in weights.columns:
                weights.loc[hold_dates, symbol] = stock_weights[symbol]

    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
    return portfolio_returns


def create_equal_weight_portfolio(
    prices: pd.DataFrame,
    symbols: Optional[list] = None,
    rebalance_freq: str = "ME",
) -> pd.Series:
    """
    Create equal-weight portfolio returns.

    Args:
        prices: DataFrame with prices (wide format: date x symbols)
        symbols: List of symbols to include (None = all symbols)
        rebalance_freq: Pandas offset (prefer ``ME``, ``QE``; legacy aliases normalized).

    Returns:
        Series of portfolio returns
    """
    if symbols is None:
        symbols = prices.columns.tolist()
    return create_weighted_portfolio(
        prices, symbols, weighting_scheme="equal", rebalance_freq=rebalance_freq
    )


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.

    Args:
        returns: Series of returns
        window: Rolling window size (default: 252 trading days = 1 year)
        periods_per_year: Number of periods per year
        risk_free_rate: Risk-free rate for Sharpe/Sortino (annualized)

    Returns:
        DataFrame with rolling metrics including Sharpe and Sortino ratios
    """
    rolling_return = returns.rolling(window).mean() * periods_per_year
    rolling_vol = returns.rolling(window).std() * np.sqrt(periods_per_year)
    rolling_sharpe = (rolling_return - risk_free_rate) / rolling_vol

    def _rolling_sortino(window_returns: pd.Series) -> float:
        if len(window_returns) < 2:
            return np.nan
        mean_return = window_returns.mean() * periods_per_year
        downside_returns = window_returns[window_returns < 0]
        if len(downside_returns) == 0:
            return np.nan
        downside_dev = downside_returns.std() * np.sqrt(periods_per_year)
        if downside_dev == 0:
            return np.nan
        return (mean_return - risk_free_rate) / downside_dev

    rolling_sortino_ratio = returns.rolling(window).apply(_rolling_sortino, raw=False)

    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.rolling(window, min_periods=1).max()
    rolling_dd = (cum_returns - rolling_max) / rolling_max

    return pd.DataFrame(
        {
            "annualized_return": rolling_return,
            "annualized_volatility": rolling_vol,
            "sharpe_ratio": rolling_sharpe,
            "sortino_ratio": rolling_sortino_ratio,
            "drawdown": rolling_dd,
        }
    )
