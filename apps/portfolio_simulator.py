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
        height=500,
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
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
        )
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
        height=400,
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
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
        )
    )
    
    return fig


def plot_rolling_metrics(returns, window=252):
    """Plot rolling Sharpe ratio and volatility."""
    rolling = calculate_rolling_metrics(returns, window=window)
    
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Rolling Sharpe Ratio", "Rolling Volatility"),
        vertical_spacing=0.15,
    )
    
    # Sharpe ratio
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling["sharpe_ratio"],
            mode="lines",
            name="Sharpe Ratio",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    
    # Volatility
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling["annualized_volatility"] * 100,
            mode="lines",
            name="Volatility",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True),
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
        )
    )
    
    return fig


def calculate_benchmark_returns(
    benchmark_type: str,
    df_prices: pd.DataFrame,
    component1: str = None,
    component2: str = None,
    weight1: float = 60.0,
) -> tuple:
    """
    Calculate benchmark returns based on type.
    
    Args:
        benchmark_type: Type of benchmark
        df_prices: Price data
        component1: First component for synthetic
        component2: Second component for synthetic
        weight1: Weight for component1 (percentage)
        
    Returns:
        Tuple of (benchmark_returns, benchmark_name)
    """
    # Helper to get returns for a component
    def get_component_returns(component_name):
        if component_name == "S&P 500" or component_name == "S&P 500 (Cap-Weighted)":
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
    
    if benchmark_type == "S&P 500 (Cap-Weighted)":
        if "^GSPC" in df_prices.columns:
            returns = df_prices["^GSPC"].pct_change()
            name = "S&P 500 (Cap-Weighted)"
        else:
            returns = df_prices.pct_change().mean(axis=1)
            name = "Equal Weight Universe (S&P 500 not available)"
            
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
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Strategy Configuration")
    
    # Get available factors
    factor_columns = [col for col in df_factors.columns if col != "signal"]
    
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
            help="Factor to use for ranking stocks",
        )
        
        top_pct = st.sidebar.slider(
            "Top % (Long)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of top-ranked stocks to go long",
        ) / 100
        
        bottom_pct = st.sidebar.slider(
            "Bottom % (Short)",
            min_value=0,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of bottom-ranked stocks to short (0 for long-only)",
        ) / 100
        
        long_only = bottom_pct == 0
        
    elif strategy_type == "Custom Selection":
        # Get list of available symbols
        available_symbols = sorted(df_prices.columns.tolist())
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
    
    # Backtesting settings
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Backtesting Settings")
    
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
    
    # Benchmark selection
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Benchmark Selection")
    
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
        ["S&P 500 (Cap-Weighted)", "Equal Weight Universe", "Synthetic (Custom Mix)"],
        help="Choose the benchmark for comparison",
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
    
    # Date range filter
    st.sidebar.markdown("---")
    st.sidebar.header("📅 Date Range")
    
    min_date = df_prices.index.min().date()
    max_date = df_prices.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(max_date - pd.Timedelta(days=365 * 5), max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter the backtest to a specific date range",
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_prices_filtered = df_prices[
            (df_prices.index.date >= start_date) & (df_prices.index.date <= end_date)
        ]
        df_factors_filtered = df_factors[
            (df_factors.index.get_level_values("date").date >= start_date)
            & (df_factors.index.get_level_values("date").date <= end_date)
        ]
    else:
        df_prices_filtered = df_prices
        df_factors_filtered = df_factors
    
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
                
                # Cumulative returns
                returns_dict = {
                    strategy_name: portfolio_returns,
                    benchmark_name: benchmark_returns,
                }
                
                fig_cum = plot_cumulative_returns(returns_dict)
                st.plotly_chart(fig_cum, use_container_width=True)
                
                # Drawdown
                fig_dd = plot_drawdown(returns_dict)
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Rolling metrics
                st.markdown("### 📉 Rolling Metrics (1-Year Window)")
                fig_rolling = plot_rolling_metrics(portfolio_returns, window=252)
                st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Additional details for factor-based strategies
                if strategy_type == "Factor-Based":
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
            **What is "Equal Weight Universe"?**
            
            Equal Weight Universe = All stocks in your data weighted equally (1/N each).
            - **Real benchmark:** Similar to S&P 500 Equal Weight Index (ticker: RSP)
            - **Why use it:** Removes large-cap bias, shows pure diversification effect
            - **Comparison:** S&P 500 is cap-weighted (Apple ~7%), Equal Weight gives each stock 0.2%
            
            **Current Benchmarks Available:**
            - ✅ S&P 500 Cap-Weighted (^GSPC) - Standard market benchmark
            - ✅ Equal Weight Universe - All stocks equally weighted
            - ✅ Synthetic Benchmarks - Custom blends (e.g., 60% S&P + 40% Equal Weight)
            
            **Synthetic Benchmark Examples:**
            - 60% S&P 500 + 40% Equal Weight Universe
            - 70% S&P 500 + 30% Equal Weight Universe
            - 80% NASDAQ + 20% Equal Weight Universe (after adding ^IXIC)
            
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

