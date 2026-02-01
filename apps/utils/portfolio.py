"""
Portfolio simulation and backtesting utilities.

This module provides functions for simulating portfolio strategies,
calculating returns with transaction costs, and managing portfolio positions.
"""

from typing import Optional

import numpy as np
import pandas as pd


def create_signals_from_factor(
    factors_df: pd.DataFrame,
    factor_col: str,
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
    long_only: bool = False,
    min_stocks: int = 20,
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
        
    Returns:
        DataFrame with 'signal' column: 1 (long), -1 (short), 0 (neutral)
        
    Example:
        >>> signals = create_signals_from_factor(
        ...     factors_df, 'mom_12_1', top_pct=0.20, bottom_pct=0.20
        ... )
    """
    df = factors_df.copy()
    df["signal"] = 0
    
    # Process each date
    for date in df.index.get_level_values("date").unique():
        date_mask = df.index.get_level_values("date") == date
        date_data = df.loc[date_mask].copy()
        
        # Filter valid factor values (remove NaN, inf, and extreme outliers)
        valid_mask = (
            date_data[factor_col].notna()
            & np.isfinite(date_data[factor_col])
            & (date_data[factor_col].abs() < 10)  # Remove extreme outliers
        )
        valid_data = date_data[valid_mask]
        
        n_valid = len(valid_data)
        if n_valid < min_stocks:
            continue
        
        # Calculate number of stocks for long/short
        n_long = max(1, int(n_valid * top_pct))
        n_short = max(1, int(n_valid * bottom_pct))
        
        # Rank stocks by factor value
        ranks = valid_data[factor_col].rank(ascending=False)
        
        # Assign long signals (top ranked)
        long_idx = ranks[ranks <= n_long].index
        df.loc[long_idx, "signal"] = 1
        
        # Assign short signals (bottom ranked) if not long_only
        if not long_only:
            short_idx = ranks[ranks > (n_valid - n_short)].index
            df.loc[short_idx, "signal"] = -1
    
    return df


def calculate_portfolio_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_freq: str = "M",
    transaction_cost: float = 0.001,
    long_only: bool = False,
) -> pd.DataFrame:
    """
    Calculate portfolio returns from signals and prices with proper handling of:
    - Delistings (sell on last available trading date, hold as cash)
    - Rebalancing only on specified dates
    - Transaction costs
    
    Args:
        signals: DataFrame with 'signal' column (MultiIndex: date, symbol)
        prices: DataFrame with prices (wide format: date × symbols)
        rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q')
        transaction_cost: Cost per trade as decimal (0.001 = 10 bps)
        long_only: If True, ignore short signals
        
    Returns:
        DataFrame with columns:
            - gross_return: Returns before costs
            - transaction_cost: Costs incurred
            - net_return: Returns after costs
            - turnover: Portfolio turnover
            - n_long: Number of long positions
            - n_short: Number of short positions
            - cash: Cash position (from delistings)
            
    Example:
        >>> results = calculate_portfolio_returns(
        ...     signals, prices, rebalance_freq='M', transaction_cost=0.001
        ... )
    """
    # Calculate price returns
    returns = prices.pct_change()
    
    # Convert signals to wide format (date × symbols)
    signals_wide = signals["signal"].unstack(fill_value=0)
    
    # Align dates between returns and signals
    common_dates = returns.index.intersection(signals_wide.index)
    returns = returns.loc[common_dates]
    signals_wide = signals_wide.loc[common_dates]
    
    # Determine rebalance dates
    rebalance_dates = returns.resample(rebalance_freq).last().index
    
    # Initialize tracking DataFrames
    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    cash_position = pd.Series(0.0, index=returns.index)
    
    # Build positions based on rebalancing schedule
    for i, rebal_date in enumerate(rebalance_dates):
        if rebal_date not in signals_wide.index:
            continue
        
        # Get signals for this rebalance date
        sigs = signals_wide.loc[rebal_date]
        
        if long_only:
            sigs = sigs.clip(lower=0)
        
        # Filter to stocks with available prices on rebalance date
        available_stocks = prices.loc[rebal_date].dropna().index
        sigs = sigs[sigs.index.isin(available_stocks)]
        
        # Calculate weights (equal weight within long/short buckets)
        n_long = (sigs > 0).sum()
        n_short = (sigs < 0).sum()
        
        weights = pd.Series(0.0, index=sigs.index)
        if n_long > 0:
            weights[sigs > 0] = 1.0 / n_long
        if n_short > 0 and not long_only:
            weights[sigs < 0] = -1.0 / n_short
        
        # Hold positions until next rebalance
        if i + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[i + 1]
        else:
            next_rebal = returns.index[-1] + pd.Timedelta(days=1)
        
        hold_dates = returns.index[
            (returns.index >= rebal_date) & (returns.index < next_rebal)
        ]
        
        # Assign weights for holding period
        for col in weights.index:
            if col in positions.columns:
                positions.loc[hold_dates, col] = weights[col]
    
    # Calculate daily returns with delisting handling
    daily_gross_returns = []
    daily_transaction_costs = []
    daily_cash = []
    daily_n_long = []
    daily_n_short = []
    daily_turnover = []
    
    for i, date in enumerate(returns.index):
        if i == 0:
            # First day - no returns yet
            daily_gross_returns.append(0.0)
            daily_transaction_costs.append(0.0)
            daily_cash.append(0.0)
            daily_n_long.append((positions.loc[date] > 0).sum())
            daily_n_short.append((positions.loc[date] < 0).sum())
            daily_turnover.append(0.0)
            continue
        
        prev_date = returns.index[i - 1]
        
        # Get positions and returns
        pos = positions.loc[prev_date]
        ret = returns.loc[date]
        
        # Handle delistings: if stock has position but no return, it delisted
        # Sell at last available price (previous day)
        cash_from_delistings = 0.0
        for symbol in pos[pos != 0].index:
            if pd.isna(ret[symbol]) or symbol not in ret.index:
                # Stock delisted - convert position to cash
                cash_from_delistings += abs(pos[symbol])
                # Zero out position going forward
                positions.loc[date:, symbol] = 0.0
        
        # Calculate gross return from active positions
        valid_returns = ret[pos.index].fillna(0.0)
        gross_ret = (pos * valid_returns).sum()
        
        # Track cash (from delistings, waiting for rebalance)
        cash = cash_position.loc[prev_date] + cash_from_delistings
        
        # On rebalance dates, cash is reinvested (already handled in weights)
        if date in rebalance_dates:
            cash = 0.0
        
        # Calculate turnover and transaction costs
        if i > 0:
            pos_change = (positions.loc[date] - positions.loc[prev_date]).abs().sum()
            turnover_val = pos_change / 2
            trans_cost = turnover_val * transaction_cost
        else:
            turnover_val = 0.0
            trans_cost = 0.0
        
        daily_gross_returns.append(gross_ret)
        daily_transaction_costs.append(trans_cost)
        daily_cash.append(cash)
        daily_n_long.append((positions.loc[date] > 0).sum())
        daily_n_short.append((positions.loc[date] < 0).sum())
        daily_turnover.append(turnover_val)
        
        # Update cash position
        cash_position.loc[date] = cash
    
    # Convert to Series
    gross_returns = pd.Series(daily_gross_returns, index=returns.index)
    transaction_costs = pd.Series(daily_transaction_costs, index=returns.index)
    net_returns = gross_returns - transaction_costs
    
    # Combine results
    results = pd.DataFrame(
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
    
    return results


def create_weighted_portfolio(
    prices: pd.DataFrame,
    symbols: list,
    weighting_scheme: str = "equal",
    manual_weights: Optional[dict] = None,
    share_counts: Optional[dict] = None,
    rebalance_freq: str = "M",
) -> pd.Series:
    """
    Create portfolio returns with various weighting schemes.
    
    Args:
        prices: DataFrame with prices (wide format: date × symbols)
        symbols: List of symbols to include
        weighting_scheme: One of 'equal', 'manual', 'cap', 'shares', 'harmonic'
        manual_weights: Dict of {symbol: weight} for manual weighting
        share_counts: Dict of {symbol: shares} for share-based weighting
        rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q')
        
    Returns:
        Series of portfolio returns
        
    Example:
        >>> # Equal weight
        >>> returns = create_weighted_portfolio(prices, ['AAPL', 'MSFT'], 'equal')
        
        >>> # Manual weights
        >>> returns = create_weighted_portfolio(
        ...     prices, ['AAPL', 'MSFT'], 'manual',
        ...     manual_weights={'AAPL': 0.6, 'MSFT': 0.4}
        ... )
        
        >>> # Share count
        >>> returns = create_weighted_portfolio(
        ...     prices, ['AAPL', 'MSFT'], 'shares',
        ...     share_counts={'AAPL': 100, 'MSFT': 50}
        ... )
    """
    # Filter to selected symbols
    prices_selected = prices[symbols].copy()
    returns = prices_selected.pct_change()
    
    # Rebalance dates
    rebalance_dates = returns.resample(rebalance_freq).last().index
    
    # Initialize weights DataFrame
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    for i, rebal_date in enumerate(rebalance_dates):
        if rebal_date not in prices_selected.index:
            continue
        
        # Get prices at rebalance date
        current_prices = prices_selected.loc[rebal_date]
        
        # Calculate weights based on scheme
        if weighting_scheme == "equal":
            # Equal weight: 1/N for each stock
            stock_weights = pd.Series(
                1.0 / len(symbols), index=symbols
            )
            
        elif weighting_scheme == "manual":
            # Manual: use provided weights
            if manual_weights is None:
                raise ValueError("manual_weights required for manual weighting")
            stock_weights = pd.Series(manual_weights)
            # Normalize to sum to 1
            stock_weights = stock_weights / stock_weights.sum()
            
        elif weighting_scheme == "cap":
            # Cap-weighted: proportional to market cap (price as proxy)
            # In reality would use actual market cap, but price is reasonable proxy
            stock_weights = current_prices / current_prices.sum()
            
        elif weighting_scheme == "shares":
            # Share count: weight by dollar value of shares
            if share_counts is None:
                raise ValueError("share_counts required for share-based weighting")
            shares = pd.Series(share_counts)
            dollar_values = shares * current_prices
            stock_weights = dollar_values / dollar_values.sum()
            
        elif weighting_scheme == "harmonic":
            # Harmonic: inverse price (like Dow Jones)
            # Higher weight to lower-priced stocks
            inverse_prices = 1.0 / current_prices
            stock_weights = inverse_prices / inverse_prices.sum()
            
        else:
            raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")
        
        # Assign weights for holding period
        if i + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[i + 1]
        else:
            next_rebal = returns.index[-1] + pd.Timedelta(days=1)
        
        hold_dates = returns.index[
            (returns.index >= rebal_date) & (returns.index < next_rebal)
        ]
        
        for symbol in symbols:
            if symbol in weights.columns:
                weights.loc[hold_dates, symbol] = stock_weights[symbol]
    
    # Calculate portfolio returns
    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
    return portfolio_returns


def create_equal_weight_portfolio(
    prices: pd.DataFrame,
    symbols: Optional[list] = None,
    rebalance_freq: str = "M",
) -> pd.Series:
    """
    Create equal-weight portfolio returns.
    
    Args:
        prices: DataFrame with prices (wide format: date × symbols)
        symbols: List of symbols to include (None = all symbols)
        rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q')
        
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
    
    # Rolling Sortino ratio (downside deviation)
    def rolling_sortino(window_returns):
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
    
    rolling_sortino_ratio = returns.rolling(window).apply(rolling_sortino, raw=False)
    
    # Rolling max drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.rolling(window, min_periods=1).max()
    rolling_dd = (cum_returns - rolling_max) / rolling_max
    
    results = pd.DataFrame(
        {
            "annualized_return": rolling_return,
            "annualized_volatility": rolling_vol,
            "sharpe_ratio": rolling_sharpe,
            "sortino_ratio": rolling_sortino_ratio,
            "drawdown": rolling_dd,
        }
    )
    
    return results

