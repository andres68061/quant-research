#!/usr/bin/env python3
"""
Interactive Portfolio Simulator with Streamlit.

This application provides an interactive interface for simulating and analyzing
quantitative trading strategies with real-time visualizations and performance metrics.

Usage:
    streamlit run apps/portfolio_simulator.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.utils.metrics import (
    calculate_cumulative_returns,
    calculate_drawdown,
    calculate_performance_metrics,
    format_performance_table,
)
from apps.utils.portfolio import (
    calculate_portfolio_returns,
    calculate_rolling_metrics,
    create_equal_weight_portfolio,
    create_signals_from_factor,
    create_weighted_portfolio,
)

# Page configuration
st.set_page_config(
    page_title="Quant Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load prices and factors data with path resolution."""
    # Resolve data directory (handles both repo root and notebooks/)
    candidates = [ROOT / "data" / "factors", ROOT.parent / "data" / "factors"]
    
    data_dir = None
    for candidate in candidates:
        if candidate.exists():
            data_dir = candidate
            break
    
    if data_dir is None:
        st.error("❌ Data directory not found. Please run backfill_all.py first.")
        st.stop()
    
    factors_path = data_dir / "factors_price.parquet"
    prices_path = data_dir / "prices.parquet"
    
    if not factors_path.exists():
        st.error(f"❌ Factors file not found: {factors_path}")
        st.stop()
    
    if not prices_path.exists():
        st.error(f"❌ Prices file not found: {prices_path}")
        st.stop()
    
    try:
        df_factors = pd.read_parquet(factors_path)
        df_prices = pd.read_parquet(prices_path)
        return df_factors, df_prices, data_dir
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()


def plot_cumulative_returns(returns_dict, title="Cumulative Returns"):
    """Plot cumulative returns using Plotly."""
    fig = go.Figure()
    
    for name, returns in returns_dict.items():
        cum_returns = calculate_cumulative_returns(returns)
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=(cum_returns - 1) * 100,  # Convert to percentage
                mode="lines",
                name=name,
                hovertemplate="%{y:.2f}%<extra></extra>",
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
        height=600,  # Increased from 500
        showlegend=True,
        margin=dict(t=80, b=60, l=60, r=40),  # Add margins to prevent clipping
        xaxis=dict(
            rangeslider=dict(visible=True, yaxis=dict(rangemode='auto')),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="lightgray",
                activecolor="gray"
            ),
        ),
        yaxis=dict(
            fixedrange=False, 
            autorange=True,
            rangemode='normal',
        ),
    )
    
    return fig


def plot_drawdown(returns_dict, title="Drawdown Analysis"):
    """Plot drawdown using Plotly."""
    fig = go.Figure()
    
    for name, returns in returns_dict.items():
        dd = calculate_drawdown(returns)
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd * 100,  # Convert to percentage
                mode="lines",
                name=name,
                fill="tozeroy",
                hovertemplate="%{y:.2f}%<extra></extra>",
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        height=500,  # Increased from 400
        showlegend=True,
        margin=dict(t=80, b=60, l=60, r=40),  # Add margins to prevent clipping
        xaxis=dict(
            rangeslider=dict(visible=True, yaxis=dict(rangemode='auto')),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="lightgray",
                activecolor="gray"
            ),
        ),
        yaxis=dict(
            fixedrange=False, 
            autorange=True,
            rangemode='normal',
        ),
    )
    
    return fig


def plot_rolling_sharpe(returns, window=252):
    """Plot rolling Sharpe ratio."""
    rolling = calculate_rolling_metrics(returns, window=window)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling["sharpe_ratio"],
            mode="lines",
            name="Sharpe Ratio",
            line=dict(color="blue", width=2),
            hovertemplate='Sharpe: %{y:.2f}<extra></extra>',
        )
    )
    
    fig.update_layout(
        title="Rolling Sharpe Ratio (252-Day Window)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
        height=450,
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=40),
        xaxis=dict(
            rangeslider=dict(visible=True, yaxis=dict(rangemode='auto')),
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="lightgray",
                activecolor="gray"
            ),
        ),
        yaxis=dict(
            fixedrange=False, 
            autorange=True,
            rangemode='normal',
        ),
    )
    
    return fig


def plot_rolling_volatility(returns, window=252):
    """Plot rolling volatility."""
    rolling = calculate_rolling_metrics(returns, window=window)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling["annualized_volatility"] * 100,
            mode="lines",
            name="Volatility",
            line=dict(color="red", width=2),
            hovertemplate='Vol: %{y:.1f}%<extra></extra>',
        )
    )
    
    fig.update_layout(
        title="Rolling Volatility (252-Day Window)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode="x unified",
        height=450,
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=40),
        xaxis=dict(
            rangeslider=dict(visible=True, yaxis=dict(rangemode='auto')),
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="lightgray",
                activecolor="gray"
            ),
        ),
        yaxis=dict(
            fixedrange=False, 
            autorange=True,
            rangemode='normal',
        ),
    )
    
    return fig


def calculate_benchmark_returns(
    benchmark_type: str,
    df_prices: pd.DataFrame,
    component1: str = None,
    component2: str = None,
    weight1: float = 60.0,
    sp500_weighting: str = "Equal Weight",
) -> tuple:
    """
    Calculate benchmark returns based on type.
    
    Args:
        benchmark_type: Type of benchmark
        df_prices: Price data
        component1: First component for synthetic
        component2: Second component for synthetic
        weight1: Weight for component1 (percentage)
        sp500_weighting: Weighting scheme for reconstructed S&P 500
        
    Returns:
        Tuple of (benchmark_returns, benchmark_name)
    """
    # Helper to get returns for a component
    def get_component_returns(component_name):
        if component_name == "S&P 500" or component_name == "S&P 500 (^GSPC)":
            if "^GSPC" in df_prices.columns:
                return df_prices["^GSPC"].pct_change()
        elif component_name == "NASDAQ Composite" and "^IXIC" in df_prices.columns:
            return df_prices["^IXIC"].pct_change()
        elif component_name == "Dow Jones" and "^DJI" in df_prices.columns:
            return df_prices["^DJI"].pct_change()
        elif component_name == "Russell 2000" and "^RUT" in df_prices.columns:
            return df_prices["^RUT"].pct_change()
        # Default to equal weight universe
        return df_prices.pct_change().mean(axis=1)
    
    if benchmark_type == "S&P 500 (^GSPC)":
        if "^GSPC" in df_prices.columns:
            returns = df_prices["^GSPC"].pct_change()
            name = "S&P 500 (^GSPC)"
        else:
            returns = df_prices.pct_change().mean(axis=1)
            name = "Equal Weight Universe (S&P 500 not available)"
    
    elif benchmark_type == "S&P 500 Reconstructed (2020+)":
        # Use point-in-time S&P 500 constituents (eliminates survivorship bias)
        try:
            from src.data.sp500_constituents import SP500Constituents
            
            sp500 = SP500Constituents()
            sp500.load()
            
            # Pre-calculate daily returns for all stocks
            all_returns = df_prices.pct_change()
            
            if sp500_weighting == "Equal Weight":
                # Calculate equal-weight returns for S&P 500 constituents on each date
                daily_returns = []
                for date in df_prices.index:
                    # Get constituents for this date
                    constituents = sp500.get_constituents_on_date(pd.Timestamp(date))
                    
                    # Filter to constituents we have data for
                    available_constituents = [c for c in constituents if c in all_returns.columns]
                    
                    if available_constituents:
                        # Equal weight return for available constituents
                        day_return = all_returns.loc[date, available_constituents].mean()
                        daily_returns.append(day_return)
                    else:
                        daily_returns.append(np.nan)
                
                returns = pd.Series(daily_returns, index=df_prices.index)
                
                # Count unique constituents used
                all_constituents = set()
                for date in df_prices.index[::252]:  # Sample every year to avoid slow computation
                    all_constituents.update(sp500.get_constituents_on_date(pd.Timestamp(date)))
                
                name = f"S&P 500 Reconstructed (EW, ~{len(all_constituents)} tickers)"
                
            else:  # Cap-Weighted
                from src.data.market_caps import MarketCapCalculator
                
                calc = MarketCapCalculator()
                market_caps = calc.load_market_caps()
                
                if market_caps is None or market_caps.empty:
                    st.warning("Market cap data not available. Using equal weight instead.")
                    # Fallback to equal weight
                    daily_returns = []
                    for date in df_prices.index:
                        constituents = sp500.get_constituents_on_date(pd.Timestamp(date))
                        available_constituents = [c for c in constituents if c in all_returns.columns]
                        if available_constituents:
                            day_return = all_returns.loc[date, available_constituents].mean()
                            daily_returns.append(day_return)
                        else:
                            daily_returns.append(np.nan)
                    returns = pd.Series(daily_returns, index=df_prices.index)
                    name = "S&P 500 Reconstructed (EW, market caps unavailable)"
                else:
                    # Calculate cap-weighted returns for each date
                    daily_returns = []
                    
                    for date in df_prices.index:
                        # Get S&P 500 constituents for this date
                        constituents = sp500.get_constituents_on_date(pd.Timestamp(date))
                        
                        # Filter to constituents we have both price and market cap data for
                        date_ts = pd.Timestamp(date).tz_localize(None)
                        
                        # Get market caps on this date
                        try:
                            date_caps = calc.get_market_cap_on_date(date_ts, constituents)
                            available_tickers = list(date_caps.index)
                            
                            if available_tickers:
                                # Calculate weights
                                weights = date_caps / date_caps.sum()
                                
                                # Calculate weighted return
                                returns_on_date = all_returns.loc[date, available_tickers]
                                day_return = (returns_on_date * weights).sum()
                                daily_returns.append(day_return)
                            else:
                                daily_returns.append(np.nan)
                        except:
                            daily_returns.append(np.nan)
                    
                    returns = pd.Series(daily_returns, index=df_prices.index)
                    name = "S&P 500 Reconstructed (Cap-Weighted)"
            
        except Exception as e:
            st.warning(f"S&P 500 Reconstructed data not available: {str(e)}. Using ^GSPC instead.")
            if "^GSPC" in df_prices.columns:
                returns = df_prices["^GSPC"].pct_change()
                name = "S&P 500 (^GSPC)"
            else:
                returns = df_prices.pct_change().mean(axis=1)
                name = "Equal Weight Universe"
    elif benchmark_type == "Equal Weight Universe":
        returns = df_prices.pct_change().mean(axis=1)
        name = "Equal Weight Universe"
        
    elif benchmark_type == "Synthetic (Custom Mix)":
        # Get returns for each component
        returns1 = get_component_returns(component1)
        returns2 = get_component_returns(component2)
        
        # Calculate weighted average
        w1 = weight1 / 100.0
        w2 = (100 - weight1) / 100.0
        
        returns = w1 * returns1 + w2 * returns2
        name = f"{int(weight1)}% {component1} + {int(100-weight1)}% {component2}"
        
    else:
        # Default fallback
        returns = df_prices.pct_change().mean(axis=1)
        name = "Equal Weight All Stocks"
    
    return returns, name


def main():
    """Main Streamlit application."""
    st.title("📈 Quant Analytics Platform")
    
    # Multi-page navigation info
    st.info("""
    **Welcome to the Quant Analytics Platform!**
    
    👈 Use the sidebar to navigate between:
    - **Portfolio Simulator** (this page) - Backtest trading strategies
    - **Metals Analytics** - Analyze precious metals and commodities
    - **Economic Indicators** - Monitor key economic data
    """)
    
    st.markdown("---")
    st.header("Portfolio Simulator")
    
    # Load data
    with st.spinner("Loading data..."):
        df_factors, df_prices, data_dir = load_data()
    
    # STEP 1: Date Range Selection (FIRST!)
    st.sidebar.header("1️⃣ Date Range Selection")
    st.sidebar.markdown("**Select your backtest period first** - factors will be calculated from start date onwards (no look-ahead bias)")
    
    min_date = df_prices.index.min().date()
    max_date = df_prices.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Backtest Period",
        value=(max_date - pd.Timedelta(days=365 * 5), max_date),
        min_value=min_date,
        max_value=max_date,
        help="Factors will be calculated from start date onwards to prevent look-ahead bias",
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        # Filter data to start_date onwards (NO LOOK-AHEAD!)
        df_prices_filtered = df_prices[
            (df_prices.index.date >= start_date) & (df_prices.index.date <= end_date)
        ]
        
        # CRITICAL: Filter out penny stocks (prices < $5)
        # Small prices cause huge percentage returns and corrupt results
        # Standard practice in quant research
        # Note: fillna(inf) ensures NaN values don't cause false exclusions
        price_mask = (df_prices_filtered.fillna(np.inf) >= 5.0).all(axis=0)
        valid_symbols = df_prices_filtered.columns[price_mask]
        
        if len(valid_symbols) < len(df_prices_filtered.columns):
            excluded = len(df_prices_filtered.columns) - len(valid_symbols)
            st.sidebar.info(f"ℹ️ Excluded {excluded} penny stocks (price < $5) to prevent data corruption")
        
        df_prices_filtered = df_prices_filtered[valid_symbols]
        
        df_factors_filtered = df_factors[
            (df_factors.index.get_level_values("date").date >= start_date)
            & (df_factors.index.get_level_values("date").date <= end_date)
            & (df_factors.index.get_level_values("symbol").isin(valid_symbols))
        ]
        
        st.sidebar.success(f"✓ Data filtered: {start_date} to {end_date}")
        st.sidebar.info(f"📊 {len(df_prices_filtered)} trading days, {len(valid_symbols)} symbols")
    else:
        st.sidebar.warning("Please select both start and end dates")
        df_prices_filtered = df_prices
        df_factors_filtered = df_factors
    
    st.sidebar.markdown("---")
    
    # STEP 2: Strategy Configuration
    st.sidebar.header("2️⃣ Strategy Configuration")
    
    # Get available factors
    factor_columns = [col for col in df_factors_filtered.columns if col != "signal"]
    
    # Strategy settings
    strategy_type = st.sidebar.selectbox(
        "Strategy Type",
        ["Factor-Based", "Equal Weight", "Custom Selection"],
        help="Select the type of portfolio strategy to simulate",
    )
    
    if strategy_type == "Factor-Based":
        factor_col = st.sidebar.selectbox(
            "Factor",
            factor_columns,
            index=factor_columns.index("mom_12_1")
            if "mom_12_1" in factor_columns
            else 0,
            help="mom_12_1 = 12-month momentum excluding last month (industry standard for momentum strategies)",
        )
        
        st.sidebar.caption("**Factor Definitions:**")
        st.sidebar.caption("• `mom_12_1`: Return over 12 months, excluding last month")
        st.sidebar.caption("• `mom_6_1`: Return over 6 months, excluding last month")  
        st.sidebar.caption("• `vol_60d`: 60-day volatility (annualized)")
        st.sidebar.caption("• `beta_60d`: 60-day beta vs SPY")
        
        top_pct = st.sidebar.slider(
            "Top % (Long)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Go LONG the top 20% of stocks ranked by the factor (e.g., highest momentum)",
        ) / 100
        
        bottom_pct = st.sidebar.slider(
            "Bottom % (Short)",
            min_value=0,
            max_value=50,
            value=20,
            step=5,
            help="Go SHORT the bottom 20% of stocks (e.g., lowest momentum). Set to 0 for long-only.",
        ) / 100
        
        long_only = bottom_pct == 0
        
    elif strategy_type == "Custom Selection":
        # Option to filter to S&P 500 historical members
        filter_to_sp500 = st.sidebar.checkbox(
            "Show only S&P 500 Historical members",
            value=False,
            help="Filter stock list to only those that were ever in the S&P 500 (1996-2026)"
        )
        
        # Get list of available symbols
        available_symbols = sorted(df_prices.columns.tolist())
        
        # Filter to S&P 500 if requested
        if filter_to_sp500:
            try:
                from src.data.sp500_constituents import SP500Constituents
                sp500 = SP500Constituents()
                sp500.load()
                sp500_universe = sp500.get_ticker_universe()
                available_symbols = [s for s in available_symbols if s in sp500_universe]
                st.sidebar.info(f"✓ Filtered to {len(available_symbols)} S&P 500 historical members")
            except Exception as e:
                st.sidebar.warning(f"Could not load S&P 500 data: {str(e)}")
        
        selected_symbols = st.sidebar.multiselect(
            "Select Stocks",
            available_symbols,
            default=available_symbols[:10] if len(available_symbols) >= 10 else available_symbols[:5],
            help="Choose specific stocks for your portfolio",
        )
        
        # Weighting scheme
        weighting_scheme = st.sidebar.selectbox(
            "Weighting Scheme",
            ["Equal Weight", "Manual Weights", "Cap-Weighted", "Share Count", "Harmonic (Inverse Price)"],
            help="How to weight stocks in your portfolio",
        )
        
        # Conditional inputs based on weighting scheme
        manual_weights = None
        share_counts = None
        
        if weighting_scheme == "Manual Weights" and selected_symbols:
            st.sidebar.markdown("**Set Weights (%):**")
            manual_weights = {}
            total_weight = 0.0
            
            for symbol in selected_symbols:
                weight = st.sidebar.number_input(
                    f"{symbol}",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0 / len(selected_symbols),
                    step=1.0,
                    key=f"weight_{symbol}",
                )
                manual_weights[symbol] = weight / 100.0  # Convert to decimal
                total_weight += weight
            
            if abs(total_weight - 100.0) > 0.1:
                st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.1f}% (should be 100%)")
        
        elif weighting_scheme == "Share Count" and selected_symbols:
            st.sidebar.markdown("**Number of Shares:**")
            share_counts = {}
            
            for symbol in selected_symbols:
                shares = st.sidebar.number_input(
                    f"{symbol}",
                    min_value=1,
                    max_value=100000,
                    value=100,
                    step=10,
                    key=f"shares_{symbol}",
                )
                share_counts[symbol] = shares
    
    # STEP 3: Backtesting Settings
    st.sidebar.markdown("---")
    st.sidebar.header("3️⃣ Backtesting Settings")
    
    rebalance_freq = st.sidebar.selectbox(
        "Rebalancing Frequency",
        ["Daily", "Weekly", "Monthly", "Quarterly"],
        index=2,
        help="How often to rebalance the portfolio",
    )
    
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
    rebalance_freq_code = freq_map[rebalance_freq]
    
    transaction_cost = st.sidebar.slider(
        "Transaction Cost (bps)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="Transaction cost in basis points (10 bps = 0.10%)",
    ) / 10000  # Convert bps to decimal
    
    # STEP 4: Benchmark Selection
    st.sidebar.markdown("---")
    st.sidebar.header("4️⃣ Benchmark Selection")
    
    # Check available benchmarks in data
    available_indices = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ Composite",
        "^DJI": "Dow Jones",
        "^RUT": "Russell 2000",
    }
    
    indices_in_data = {k: v for k, v in available_indices.items() if k in df_prices.columns}
    
    benchmark_type = st.sidebar.selectbox(
        "Benchmark Type",
        [
            "S&P 500 (^GSPC)", 
            "S&P 500 Reconstructed (2020+)",
            "Equal Weight Universe", 
            "Synthetic (Custom Mix)"
        ],
        help="Choose the benchmark for comparison. S&P 500 Reconstructed uses point-in-time constituents (best for 2024-2026).",
    )
    
    # Show coverage warning for reconstructed S&P 500
    if benchmark_type == "S&P 500 Reconstructed (2020+)":
        st.sidebar.info("""
        **Coverage by Period:**
        - 2024-2026: ✅ 97-99% (Excellent)
        - 2020-2023: ⚠️ 93-96% (Use with caution)
        - Before 2020: ❌ Not recommended
        
        Missing symbols introduce survivorship bias.
        """)
        
        # Option for equal or cap-weighted
        sp500_weighting = st.sidebar.selectbox(
            "Weighting Scheme",
            ["Equal Weight", "Cap-Weighted"],
            help="Equal weight = 1/N per stock, Cap-weighted = weighted by market cap"
        )
    
    # Synthetic benchmark configuration
    if benchmark_type == "Synthetic (Custom Mix)":
        st.sidebar.markdown("**Configure Synthetic Benchmark:**")
        
        # Component 1
        if indices_in_data:
            component1 = st.sidebar.selectbox(
                "Component 1",
                list(indices_in_data.values()) + ["Equal Weight Universe"],
                index=0,
            )
        else:
            component1 = "Equal Weight Universe"
        
        weight1 = st.sidebar.slider(
            "Weight 1 (%)",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            help="Percentage allocation to Component 1",
        )
        
        # Component 2
        if indices_in_data:
            component2_options = [opt for opt in list(indices_in_data.values()) + ["Equal Weight Universe"] if opt != component1]
            if component2_options:
                component2 = st.sidebar.selectbox(
                    "Component 2",
                    component2_options,
                    index=0 if len(component2_options) > 0 else None,
                )
            else:
                component2 = "Equal Weight Universe"
        else:
            component2 = "Equal Weight Universe"
        
        weight2 = 100 - weight1
        st.sidebar.info(f"Component 2: {weight2}%")
    
    # Run simulation button
    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")
    
    if run_simulation:
        with st.spinner("Running simulation..."):
            try:
                # Create portfolio based on strategy type
                if strategy_type == "Factor-Based":
                    # Create signals
                    df_signals = create_signals_from_factor(
                        df_factors_filtered,
                        factor_col=factor_col,
                        top_pct=top_pct,
                        bottom_pct=bottom_pct,
                        long_only=long_only,
                    )
                    
                    # Calculate returns
                    backtest_results = calculate_portfolio_returns(
                        signals=df_signals,
                        prices=df_prices_filtered,
                        rebalance_freq=rebalance_freq_code,
                        transaction_cost=transaction_cost,
                        long_only=long_only,
                    )
                    
                    portfolio_returns = backtest_results["net_return"]
                    gross_returns = backtest_results["gross_return"]
                    
                    strategy_name = f"{factor_col.upper()} ({'Long-Only' if long_only else 'Long/Short'})"
                    
                elif strategy_type == "Equal Weight":
                    portfolio_returns = create_equal_weight_portfolio(
                        df_prices_filtered,
                        rebalance_freq=rebalance_freq_code,
                    )
                    gross_returns = portfolio_returns
                    backtest_results = None  # No detailed results for equal weight
                    strategy_name = "Equal Weight"
                    
                elif strategy_type == "Custom Selection":
                    if not selected_symbols:
                        st.warning("⚠️ Please select at least one stock.")
                        return
                    
                    # Map weighting scheme to internal name
                    scheme_map = {
                        "Equal Weight": "equal",
                        "Manual Weights": "manual",
                        "Cap-Weighted": "cap",
                        "Share Count": "shares",
                        "Harmonic (Inverse Price)": "harmonic",
                    }
                    scheme = scheme_map[weighting_scheme]
                    
                    # Create portfolio with selected weighting
                    portfolio_returns = create_weighted_portfolio(
                        df_prices_filtered,
                        symbols=selected_symbols,
                        weighting_scheme=scheme,
                        manual_weights=manual_weights,
                        share_counts=share_counts,
                        rebalance_freq=rebalance_freq_code,
                    )
                    gross_returns = portfolio_returns
                    backtest_results = None  # No detailed results for custom selection
                    strategy_name = f"Custom Portfolio ({weighting_scheme})"
                
                # Calculate benchmark based on user selection
                if benchmark_type == "Synthetic (Custom Mix)":
                    benchmark_returns, benchmark_name = calculate_benchmark_returns(
                        benchmark_type,
                        df_prices_filtered,
                        component1=component1,
                        component2=component2,
                        weight1=weight1,
                    )
                elif benchmark_type == "S&P 500 Reconstructed (2020+)":
                    benchmark_returns, benchmark_name = calculate_benchmark_returns(
                        benchmark_type,
                        df_prices_filtered,
                        sp500_weighting=sp500_weighting,
                    )
                else:
                    benchmark_returns, benchmark_name = calculate_benchmark_returns(
                        benchmark_type,
                        df_prices_filtered,
                    )
                
                benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
                
                # Calculate metrics
                portfolio_metrics = calculate_performance_metrics(
                    portfolio_returns,
                    benchmark_returns=benchmark_returns,
                    risk_free_rate=0.0,
                    periods_per_year=252,
                )
                
                benchmark_metrics = calculate_performance_metrics(
                    benchmark_returns,
                    risk_free_rate=0.0,
                    periods_per_year=252,
                )
                
                # Display results
                st.success("✅ Simulation complete!")
                
                # Key metrics in columns
                st.markdown("### 📊 Performance Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Return",
                        f"{portfolio_metrics['total_return'] * 100:.2f}%",
                        delta=f"{(portfolio_metrics['total_return'] - benchmark_metrics['total_return']) * 100:.2f}% vs {benchmark_name}",
                    )
                
                with col2:
                    st.metric(
                        "Annualized Return",
                        f"{portfolio_metrics['annualized_return'] * 100:.2f}%",
                    )
                
                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{portfolio_metrics['sharpe_ratio']:.2f}",
                    )
                
                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{portfolio_metrics['max_drawdown'] * 100:.2f}%",
                    )
                
                # Detailed metrics tables
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### {strategy_name} Metrics")
                    st.dataframe(
                        format_performance_table(portfolio_metrics),
                        use_container_width=True,
                        hide_index=True,
                    )
                
                with col2:
                    st.markdown(f"#### {benchmark_name} Metrics")
                    st.dataframe(
                        format_performance_table(benchmark_metrics),
                        use_container_width=True,
                        hide_index=True,
                    )
                
                # Visualizations
                st.markdown("---")
                st.markdown("### 📈 Performance Visualizations")
                
                # Check if we have valid data
                if len(portfolio_returns) == 0 or portfolio_returns.index.isna().all():
                    st.error("❌ No valid returns data to display")
                    st.stop()
                
                # Date range filter for charts
                st.markdown("**📅 Chart Date Range** (Y-axis auto-adjusts to selected period)")
                
                # Get valid date range (filter out NaT)
                valid_dates = portfolio_returns.index[portfolio_returns.index.notna()]
                if len(valid_dates) == 0:
                    st.error("❌ No valid dates in returns data")
                    st.stop()
                
                min_date = valid_dates.min().to_pydatetime().date()
                max_date = valid_dates.max().to_pydatetime().date()
                
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    chart_start_date = st.date_input(
                        "Start Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="chart_start_date"
                    )
                with col_date2:
                    chart_end_date = st.date_input(
                        "End Date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="chart_end_date"
                    )
                
                # Filter returns by selected date range
                chart_start = pd.Timestamp(chart_start_date).tz_localize(None)
                chart_end = pd.Timestamp(chart_end_date).tz_localize(None)
                
                # Ensure portfolio and benchmark indices are timezone-naive
                portfolio_tz_naive = portfolio_returns.copy()
                benchmark_tz_naive = benchmark_returns.copy()
                if hasattr(portfolio_tz_naive.index, 'tz') and portfolio_tz_naive.index.tz is not None:
                    portfolio_tz_naive.index = portfolio_tz_naive.index.tz_localize(None)
                if hasattr(benchmark_tz_naive.index, 'tz') and benchmark_tz_naive.index.tz is not None:
                    benchmark_tz_naive.index = benchmark_tz_naive.index.tz_localize(None)
                
                filtered_portfolio = portfolio_tz_naive.loc[chart_start:chart_end]
                filtered_benchmark = benchmark_tz_naive.loc[chart_start:chart_end]
                
                # Cumulative returns (recalculated for filtered period)
                returns_dict = {
                    strategy_name: filtered_portfolio,
                    benchmark_name: filtered_benchmark,
                }
                
                fig_cum = plot_cumulative_returns(returns_dict)
                st.plotly_chart(fig_cum, use_container_width=True)
                
                # Drawdown (recalculated for filtered period)
                fig_dd = plot_drawdown(returns_dict)
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Rolling metrics - Separate charts
                st.markdown("### 📈 Rolling Sharpe Ratio (1-Year Window)")
                fig_sharpe = plot_rolling_sharpe(portfolio_returns, window=252)
                st.plotly_chart(fig_sharpe, use_container_width=True)
                
                st.markdown("### 📉 Rolling Volatility (1-Year Window)")
                fig_vol = plot_rolling_volatility(portfolio_returns, window=252)
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # VaR Analysis Section
                st.markdown("---")
                st.markdown("### 📊 Value at Risk (VaR) Analysis")
                st.markdown("*What's the maximum loss you could expect?*")
                
                var_confidence = st.slider(
                    "VaR Confidence Level (%)",
                    min_value=90,
                    max_value=99,
                    value=95,
                    step=1,
                    help="Higher confidence = more conservative (larger VaR)"
                )
                
                # Calculate VaR metrics
                returns_array = portfolio_returns.values
                alpha = 1 - var_confidence / 100
                
                # 1. Historical VaR (use 100 - confidence to get the left tail)
                historical_var = np.percentile(returns_array, 100 - var_confidence) * -100
                historical_cvar = returns_array[returns_array <= np.percentile(returns_array, 100 - var_confidence)].mean() * -100
                
                # 2. Parametric VaR (assumes normal distribution)
                from scipy import stats as scipy_stats
                mu = returns_array.mean()
                sigma = returns_array.std()
                z_score = scipy_stats.norm.ppf(1 - var_confidence / 100)
                parametric_var = -(mu + z_score * sigma) * 100
                parametric_cvar = -(mu - sigma * scipy_stats.norm.pdf(z_score) / alpha) * 100
                
                # 3. Monte Carlo VaR
                n_simulations = 10000
                np.random.seed(42)
                mc_returns = np.random.normal(mu, sigma, n_simulations)
                mc_var = np.percentile(mc_returns, 100 - var_confidence) * -100
                mc_cvar = mc_returns[mc_returns <= np.percentile(mc_returns, 100 - var_confidence)].mean() * -100
                
                # Display VaR metrics
                st.markdown(f"**At {var_confidence}% confidence level:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 📜 Historical VaR")
                    st.metric(
                        "Daily VaR",
                        f"{historical_var:.2f}%",
                        help="Based on actual historical returns"
                    )
                    st.metric(
                        "CVaR (Expected Shortfall)",
                        f"{historical_cvar:.2f}%",
                        help="Average loss when VaR is exceeded"
                    )
                    st.caption("Uses actual past returns")
                
                with col2:
                    st.markdown("#### 📐 Parametric VaR")
                    st.metric(
                        "Daily VaR",
                        f"{parametric_var:.2f}%",
                        help="Assumes returns are normally distributed"
                    )
                    st.metric(
                        "CVaR (Expected Shortfall)",
                        f"{parametric_cvar:.2f}%",
                        help="Average loss when VaR is exceeded"
                    )
                    st.caption("Assumes normal distribution")
                
                with col3:
                    st.markdown("#### 🎲 Monte Carlo VaR")
                    st.metric(
                        "Daily VaR",
                        f"{mc_var:.2f}%",
                        help="Based on 10,000 simulated scenarios"
                    )
                    st.metric(
                        "CVaR (Expected Shortfall)",
                        f"{mc_cvar:.2f}%",
                        help="Average loss when VaR is exceeded"
                    )
                    st.caption("10,000 simulated scenarios")
                
                # VaR Comparison Chart
                st.markdown("#### 📊 VaR Comparison")
                
                fig_var = go.Figure()
                
                var_methods = ['Historical', 'Parametric', 'Monte Carlo']
                var_values = [historical_var, parametric_var, mc_var]
                cvar_values = [historical_cvar, parametric_cvar, mc_cvar]
                colors = ['#3498db', '#e74c3c', '#2ecc71']
                
                # VaR bars
                fig_var.add_trace(go.Bar(
                    name='VaR',
                    x=var_methods,
                    y=var_values,
                    marker_color=colors,
                    text=[f'{v:.2f}%' for v in var_values],
                    textposition='outside',
                ))
                
                # CVaR bars
                fig_var.add_trace(go.Bar(
                    name='CVaR (Expected Shortfall)',
                    x=var_methods,
                    y=cvar_values,
                    marker_color=[c.replace(')', ', 0.5)').replace('rgb', 'rgba') if 'rgb' in c else c for c in colors],
                    marker_pattern_shape='/',
                    text=[f'{v:.2f}%' for v in cvar_values],
                    textposition='outside',
                    opacity=0.7,
                ))
                
                fig_var.update_layout(
                    title=f"VaR Comparison at {var_confidence}% Confidence",
                    yaxis_title="Maximum Expected Loss (%)",
                    barmode='group',
                    height=500,  # Increased from 400
                    showlegend=True,
                )
                
                st.plotly_chart(fig_var, use_container_width=True)
                
                # Return Distribution with VaR
                st.markdown("#### 📈 Return Distribution with VaR Thresholds")
                
                fig_dist = go.Figure()
                
                # Histogram of returns
                fig_dist.add_trace(go.Histogram(
                    x=returns_array * 100,
                    name='Daily Returns',
                    nbinsx=50,
                    marker_color='lightblue',
                    opacity=0.7,
                ))
                
                # Add VaR lines
                fig_dist.add_vline(
                    x=-historical_var,
                    line_dash="solid",
                    line_color="blue",
                    annotation_text=f"Historical VaR: {historical_var:.2f}%",
                    annotation_position="top left"
                )
                
                fig_dist.add_vline(
                    x=-parametric_var,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Parametric VaR: {parametric_var:.2f}%",
                    annotation_position="top right"
                )
                
                fig_dist.update_layout(
                    title="Daily Return Distribution with VaR Thresholds",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=500,  # Increased from 400
                    showlegend=True,
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # VaR Interpretation
                with st.expander("ℹ️ Understanding VaR"):
                    st.markdown(f"""
                    **What does VaR mean?**
                    
                    At a **{var_confidence}% confidence level**, the VaR tells you:
                    
                    > "On {100 - var_confidence}% of days, your portfolio could lose **more than** the VaR amount."
                    
                    **Example:** If Historical VaR = {historical_var:.2f}%, then on {100 - var_confidence}% of trading days 
                    (~{int((100 - var_confidence) / 100 * 252)} days per year), you could lose more than {historical_var:.2f}%.
                    
                    **Three Methods:**
                    
                    1. **Historical VaR**: Uses actual past returns. Best when history is representative of future.
                    
                    2. **Parametric VaR**: Assumes normal distribution. Faster but **underestimates tail risk**.
                    
                    3. **Monte Carlo VaR**: Simulates many scenarios. Flexible but computationally intensive.
                    
                    **CVaR (Expected Shortfall):**
                    
                    While VaR tells you the threshold, CVaR tells you the **average loss when things go wrong**.
                    It's more useful for understanding tail risk.
                    """)
                
                # Additional details for factor-based strategies
                if strategy_type == "Factor-Based" and backtest_results is not None:
                    st.markdown("---")
                    st.markdown("### 📋 Strategy Details")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_long = backtest_results["n_long"].mean()
                        st.metric("Avg # Long Positions", f"{avg_long:.0f}")
                    
                    with col2:
                        avg_short = backtest_results["n_short"].mean()
                        st.metric("Avg # Short Positions", f"{avg_short:.0f}")
                    
                    with col3:
                        avg_turnover = backtest_results["turnover"].mean()
                        st.metric("Avg Daily Turnover", f"{avg_turnover * 100:.2f}%")
                    
                    # Transaction cost impact
                    cost_impact = (
                        gross_returns.mean() - portfolio_returns.mean()
                    ) * 252 * 100
                    st.info(
                        f"💰 Transaction costs reduced annualized returns by {cost_impact:.2f}%"
                    )
                
                # Download results
                st.markdown("---")
                st.markdown("### 💾 Download Results")
                
                # Prepare download data
                download_df = pd.DataFrame(
                    {
                        "date": portfolio_returns.index,
                        "portfolio_return": portfolio_returns.values,
                        "benchmark_return": benchmark_returns.values,
                    }
                )
                
                csv = download_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Returns (CSV)",
                    data=csv,
                    file_name=f"portfolio_returns_{strategy_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
                
            except Exception as e:
                st.error(f"❌ Error running simulation: {str(e)}")
                st.exception(e)
    
    else:
        # Show welcome message
        st.info(
            """
            👋 **Welcome to the Portfolio Simulator!**
            
            Configure your strategy in the sidebar and click **Run Simulation** to get started.
            
            **Features:**
            - Factor-based strategies (momentum, value, etc.)
            - Equal-weight portfolios
            - Custom stock selection
            - **Flexible benchmarks** (S&P 500, Equal Weight, or Synthetic)
            - Transaction cost modeling
            - Comprehensive performance metrics
            - Interactive visualizations
            
            **Data loaded:**
            - {} stocks
            - {} trading days
            - Date range: {} to {}
            """.format(
                len(df_prices.columns),
                len(df_prices),
                df_prices.index.min().strftime("%Y-%m-%d"),
                df_prices.index.max().strftime("%Y-%m-%d"),
            )
        )
        
        # Show available factors
        with st.expander("📊 Available Factors"):
            st.write(factor_columns)
        
        # Show benchmark info
        with st.expander("🎯 Benchmark Options"):
            st.markdown("""
            **Available Benchmarks:**
            
            1. **S&P 500 (^GSPC)** - Standard market benchmark
               - ✅ Official S&P 500 index from Yahoo Finance
               - ✅ Complete historical data
               - ✅ Best for general benchmarking
            
            2. **S&P 500 Reconstructed (2020+)** - Point-in-time constituents
               - ✅ Uses actual historical S&P 500 members on each date
               - ✅ Eliminates survivorship bias
               - ⚠️ **Coverage by period:**
                 - 2024-2026: 97-99% coverage (Excellent)
                 - 2020-2023: 93-96% coverage (Use with caution)
                 - Before 2020: <93% coverage (Not recommended)
               - 💡 Choose Equal Weight or Cap-Weighted
            
            3. **Equal Weight Universe** - All stocks equally weighted
               - Similar to S&P 500 Equal Weight Index (ticker: RSP)
               - Removes large-cap bias
               - Shows pure diversification effect
            
            4. **Synthetic (Custom Mix)** - Custom blends
               - Example: 60% S&P 500 + 40% Equal Weight
               - Mix different indices and strategies
            
            **Adding More Indices:**
            To add NASDAQ, Russell 2000, or other indices:
            ```bash
            python scripts/add_symbol.py ^IXIC ^RUT ^DJI
            ```
            
            Where:
            - `^IXIC` = NASDAQ Composite (tech-heavy)
            - `^RUT` = Russell 2000 (small cap)
            - `^DJI` = Dow Jones Industrial Average
            """)
        
        # Show data info
        with st.expander("ℹ️ Data Information"):
            st.write(f"**Data Directory:** {data_dir}")
            st.write(f"**Factors Shape:** {df_factors.shape}")
            st.write(f"**Prices Shape:** {df_prices.shape}")
            
            # Show available indices
            indices = [col for col in df_prices.columns if col.startswith("^")]
            if indices:
                st.write(f"**Available Indices:** {', '.join(indices)}")
            else:
                st.warning("No index data found. Add indices using `python scripts/add_symbol.py ^IXIC ^RUT`")


if __name__ == "__main__":
    main()

