#!/usr/bin/env python3
"""
Metals Analytics Page

Interactive analysis of precious metals and commodity prices.
Now uses persistent parquet storage for fast loading.
"""

from src.data.commodities import COMMODITIES_CONFIG
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Page configuration
st.set_page_config(
    page_title="Metals Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Commodities & Metals Analytics")
st.markdown(
    "**Persistent Data Storage** - Daily updated from Alpha Vantage & Yahoo Finance")

st.info("""
📋 **Available Assets:**
- **Precious Metals**: Gold, Silver, Platinum, Palladium (via Yahoo Finance ETFs)
- **Energy**: Crude Oil (WTI & Brent), Natural Gas
- **Industrial Metals**: Copper, Aluminum
- **Agricultural**: Wheat, Corn, Coffee, Cotton, Sugar

💾 **Data is stored locally and updated daily** - No API rate limits!
""")

st.markdown("---")

# Load commodities data from parquet


@st.cache_data(ttl=60)  # Cache for 1 minute to allow quick updates
def load_commodities_data():
    """Load commodities data from parquet file."""
    data_file = ROOT / "data" / "commodities" / "prices.parquet"

    if not data_file.exists():
        return None, None

    try:
        df = pd.read_parquet(data_file)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # Build metadata
        metadata = {}
        for col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                config = COMMODITIES_CONFIG.get(col, {})
                metadata[col] = {
                    "name": config.get("name", col),
                    "count": len(series),
                    "start": series.index[0],
                    "end": series.index[-1],
                    "latest": series.iloc[-1],
                    "unit": config.get("unit", "USD"),
                    "category": config.get("category", "unknown"),
                }

        return df, metadata
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


# Load data
df, metadata = load_commodities_data()

if df is None:
    st.error("""
    ⚠️ **No commodities data found!**
    
    Please run the initial data fetch:
    ```bash
    python scripts/fetch_commodities.py
    ```
    
    This will download all historical commodity prices and store them locally.
    """)
    st.stop()

# Display data info
st.success(f"✅ Loaded {len(df.columns)} commodities | "
           f"{len(df):,} days | "
           f"Last update: {df.index[-1].date()}")

# Create friendly display names
symbol_to_display = {}
display_to_symbol = {}

for symbol, config in COMMODITIES_CONFIG.items():
    display_name = f"{config['name']}"
    symbol_to_display[symbol] = display_name
    display_to_symbol[display_name] = symbol

# Sidebar controls
st.sidebar.header("⚙️ Analysis Settings")

# Build available commodities list FIRST
available_commodities = [symbol_to_display.get(col, col) for col in df.columns]

# Analysis type selector (moved to top for better UX)
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    [
        "Price Trends",
        "Returns Analysis (Arithmetic)",
        "Log Returns Analysis",
        "Cumulative Wealth (NAV)",
        "Drawdown Analysis",
        "Risk Metrics Dashboard",
        "Rolling Metrics",
        "Return Distribution",
        "Ratio Analysis",
        "ML Price Prediction",
        "Correlation Matrix",
        "Normalized Comparison",
        "Seasonality Analysis",
        "Multi-Period Performance",
    ],
    help="Type of analysis to perform",
)

# Smart defaults based on analysis type
if analysis_type == "ML Price Prediction":
    # ML needs single commodity - default to Silver
    ml_default = [symbol_to_display.get("SLV", "Silver (SLV ETF)")]
    ml_default = [d for d in ml_default if d in available_commodities]
    
    if not ml_default:
        # Fallback to first available
        ml_default = [available_commodities[0]] if available_commodities else []
    
    default_for_ml = ml_default
else:
    # Other analyses can use multiple commodities
    default_for_ml = [
        symbol_to_display.get("GLD", "Gold (GLD ETF)"),
        symbol_to_display.get("SLV", "Silver (SLV ETF)"),
        symbol_to_display.get("COPPER", "Copper"),
    ]
    default_for_ml = [d for d in default_for_ml if d in available_commodities]

# Commodity selection
selected_display = st.sidebar.multiselect(
    "Select Assets",
    available_commodities,
    default=default_for_ml,
    help="Choose which assets to analyze" + (" (ML requires exactly 1 asset)" if analysis_type == "ML Price Prediction" else ""),
)

# Convert back to symbols
selected_commodities = [display_to_symbol.get(d, d) for d in selected_display]

# Data resampling
resample_freq = st.sidebar.selectbox(
    "Data Frequency",
    ["Daily", "Weekly", "Monthly"],
    index=0,
    help="Resample data to different frequencies",
)

freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}

if not selected_commodities:
    st.warning("Please select at least one commodity")
    st.stop()

# Filter and resample data
filtered_df = df[selected_commodities].copy()

if resample_freq != "Daily":
    freq = freq_map[resample_freq]
    filtered_df = filtered_df.resample(freq).last()

# Remove any remaining NaN for selected commodities
filtered_df = filtered_df.dropna(how="all")

# Display summary metrics
st.markdown("### 📊 Selected Assets Summary")
cols = st.columns(min(4, len(selected_commodities)))

for i, symbol in enumerate(selected_commodities):
    if symbol in filtered_df.columns:
        series = filtered_df[symbol].dropna()
        if len(series) > 0:
            col = cols[i % len(cols)]
            with col:
                latest = series.iloc[-1]
                pct_change = (
                    (series.iloc[-1] / series.iloc[-2]) - 1) * 100 if len(series) > 1 else 0
                config = COMMODITIES_CONFIG.get(symbol, {})

                st.metric(
                    config.get("name", symbol),
                    f"${latest:.2f}",
                    f"{pct_change:+.2f}%",
                    help=f"{config.get('unit', 'USD')} | Last: {series.index[-1].date()}",
                )

st.markdown("---")

# Date range filter for charts
st.markdown(
    "**📅 Chart Date Range** *(adjust Y-axis to zoom into specific periods)*")

# Get overall date range
overall_min = filtered_df.index.min()
overall_max = filtered_df.index.max()

col_date1, col_date2 = st.columns(2)
with col_date1:
    chart_start_date = st.date_input(
        "Start Date",
        value=overall_min,
        min_value=overall_min,
        max_value=overall_max,
        key="chart_start_date",
    )
with col_date2:
    chart_end_date = st.date_input(
        "End Date",
        value=overall_max,
        min_value=overall_min,
        max_value=overall_max,
        key="chart_end_date",
    )

# Filter by date range
chart_start = pd.Timestamp(chart_start_date)
chart_end = pd.Timestamp(chart_end_date)
date_filtered_df = filtered_df.loc[chart_start:chart_end]

st.markdown("---")

# Analysis sections
if analysis_type == "Price Trends":
    st.subheader("📈 Price Trends Over Time")

    fig = go.Figure()

    for symbol in selected_commodities:
        if symbol in date_filtered_df.columns:
            series = date_filtered_df[symbol].dropna()
            config = COMMODITIES_CONFIG.get(symbol, {})

            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=config.get("name", symbol),
                    mode="lines",
                    line=dict(width=2),
                )
            )

    fig.update_layout(
        title=f"Commodity Prices ({resample_freq})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics table
    st.markdown("### 📊 Price Statistics")

    stats_data = []
    for symbol in selected_commodities:
        if symbol in date_filtered_df.columns:
            series = date_filtered_df[symbol].dropna()
            if len(series) > 0:
                config = COMMODITIES_CONFIG.get(symbol, {})
                stats_data.append({
                    "Asset": config.get("name", symbol),
                    "Latest": f"${series.iloc[-1]:.2f}",
                    "Mean": f"${series.mean():.2f}",
                    "Min": f"${series.min():.2f}",
                    "Max": f"${series.max():.2f}",
                    "Std Dev": f"${series.std():.2f}",
                })

    if stats_data:
        st.dataframe(pd.DataFrame(stats_data),
                     use_container_width=True, hide_index=True)

elif analysis_type == "Returns Analysis (Arithmetic)":
    st.subheader("📊 Returns Analysis")

    # Calculate returns
    returns_df = date_filtered_df.pct_change().dropna()

    # Returns distribution
    fig = go.Figure()

    for symbol in selected_commodities:
        if symbol in returns_df.columns:
            ret_series = returns_df[symbol].dropna()
            config = COMMODITIES_CONFIG.get(symbol, {})

            fig.add_trace(
                go.Histogram(
                    x=ret_series * 100,
                    name=config.get("name", symbol),
                    opacity=0.7,
                    nbinsx=50,
                )
            )

    fig.update_layout(
        title="Returns Distribution (%)",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Returns statistics
    st.markdown("### 📊 Returns Statistics")

    returns_stats = []
    for symbol in selected_commodities:
        if symbol in returns_df.columns:
            ret_series = returns_df[symbol].dropna()
            if len(ret_series) > 0:
                config = COMMODITIES_CONFIG.get(symbol, {})

                # Annualization factor
                periods_per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12}
                ann_factor = periods_per_year.get(resample_freq, 252)

                returns_stats.append({
                    "Asset": config.get("name", symbol),
                    "Mean Return": f"{ret_series.mean() * 100:.3f}%",
                    "Annualized": f"{ret_series.mean() * ann_factor * 100:.2f}%",
                    "Volatility": f"{ret_series.std() * 100:.3f}%",
                    "Ann. Vol": f"{ret_series.std() * np.sqrt(ann_factor) * 100:.2f}%",
                    "Sharpe": f"{(ret_series.mean() / ret_series.std()) * np.sqrt(ann_factor):.2f}",
                    "Skewness": f"{ret_series.skew():.3f}",
                    "Kurtosis": f"{ret_series.kurtosis():.3f}",
                })

    if returns_stats:
        st.dataframe(
            pd.DataFrame(returns_stats), use_container_width=True, hide_index=True
        )

elif analysis_type == "Log Returns Analysis":
    st.subheader("📊 Log Returns Analysis")

    st.info("""
    **Why Log Returns?**
    - Foundation for signal generation and risk modeling
    - Time-additive (sum log returns = log of total return)
    - More appropriate for time-series analysis and econometric models
    - Better for multi-period returns
    """)

    # Calculate log returns
    log_returns_df = np.log(
        date_filtered_df / date_filtered_df.shift(1)).dropna()
    arith_returns_df = date_filtered_df.pct_change().dropna()

    # Check if we have enough data
    if len(log_returns_df) < 2:
        st.warning(
            "⚠️ Insufficient data for log returns analysis. Please select a longer date range or different assets.")
        st.stop()

    # Filter to only commodities with valid data in the selected date range
    valid_commodities = []
    for symbol in selected_commodities:
        if symbol in log_returns_df.columns:
            valid_count = log_returns_df[symbol].notna().sum()
            if valid_count > 1:
                valid_commodities.append(symbol)

    if not valid_commodities:
        st.warning("""
        ⚠️ **No valid data for selected commodities in this date range.**
        
        This can happen when:
        - Commodities started trading after your selected start date
        - Data is not available for the selected period
        
        **Try:**
        - Selecting a more recent date range
        - Choosing different commodities
        - Checking when each commodity's data begins
        """)
        st.stop()

    # Show info about filtered commodities
    if len(valid_commodities) < len(selected_commodities):
        excluded = set(selected_commodities) - set(valid_commodities)
        excluded_names = [COMMODITIES_CONFIG.get(
            s, {}).get("name", s) for s in excluded]
        st.info(f"""
        ℹ️ **Note:** Excluding {len(excluded)} commodity/commodities with insufficient data in this date range:
        {', '.join(excluded_names)}
        
        Analyzing {len(valid_commodities)} commodities with valid data.
        """)

    # Update selected_commodities to only valid ones
    selected_commodities_filtered = valid_commodities

    # 1. Time series comparison
    st.markdown("### 📈 Log Returns vs Arithmetic Returns")

    fig = make_subplots(
        rows=len(selected_commodities_filtered),
        cols=1,
        subplot_titles=[COMMODITIES_CONFIG.get(s, {}).get(
            "name", s) for s in selected_commodities_filtered],
        vertical_spacing=0.05,
    )

    for idx, symbol in enumerate(selected_commodities_filtered, 1):
        if symbol in log_returns_df.columns:
            log_ret = log_returns_df[symbol].dropna()
            arith_ret = arith_returns_df[symbol].dropna()

            fig.add_trace(
                go.Scatter(
                    x=log_ret.index,
                    y=log_ret * 100,
                    name=f"{symbol} (Log)",
                    mode="lines",
                    line=dict(width=1),
                    legendgroup=symbol,
                ),
                row=idx, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=arith_ret.index,
                    y=arith_ret * 100,
                    name=f"{symbol} (Arithmetic)",
                    mode="lines",
                    line=dict(width=1, dash="dot"),
                    legendgroup=symbol,
                ),
                row=idx, col=1
            )

    fig.update_xaxes(title_text="Date", row=len(
        selected_commodities_filtered), col=1)
    fig.update_yaxes(title_text="Return (%)")
    fig.update_layout(
        height=300 * len(selected_commodities_filtered),
        showlegend=True,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # 2. Scatter plot: Log vs Arithmetic
    st.markdown("### 🔍 Log vs Arithmetic Returns (Scatter)")

    fig_scatter = go.Figure()

    for symbol in selected_commodities_filtered:
        if symbol in log_returns_df.columns:
            log_ret = log_returns_df[symbol].dropna()
            arith_ret = arith_returns_df[symbol].dropna()

            # Align indices
            common_idx = log_ret.index.intersection(arith_ret.index)
            log_ret = log_ret.loc[common_idx]
            arith_ret = arith_ret.loc[common_idx]

            config = COMMODITIES_CONFIG.get(symbol, {})

            fig_scatter.add_trace(
                go.Scatter(
                    x=arith_ret * 100,
                    y=log_ret * 100,
                    name=config.get("name", symbol),
                    mode="markers",
                    marker=dict(size=3, opacity=0.5),
                )
            )

    # Add diagonal line (where log = arithmetic)
    log_values = log_returns_df.values.flatten()
    log_values = log_values[~np.isnan(log_values)]  # Remove NaN

    if len(log_values) > 0:
        max_val = max(
            abs(log_values.max()),
            abs(log_values.min())
        ) * 100
        fig_scatter.add_trace(
            go.Scatter(
                x=[-max_val, max_val],
                y=[-max_val, max_val],
                name="y=x",
                mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                showlegend=True,
            )
        )

    fig_scatter.update_layout(
        title="Log Returns vs Arithmetic Returns",
        xaxis_title="Arithmetic Return (%)",
        yaxis_title="Log Return (%)",
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Statistics comparison table
    st.markdown("### 📊 Statistics: Log vs Arithmetic Returns")

    periods_per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12}
    ann_factor = periods_per_year.get(resample_freq, 252)

    stats_comparison = []
    for symbol in selected_commodities_filtered:
        if symbol in log_returns_df.columns:
            log_ret = log_returns_df[symbol].dropna()
            arith_ret = arith_returns_df[symbol].dropna()

            config = COMMODITIES_CONFIG.get(symbol, {})

            stats_comparison.append({
                "Asset": config.get("name", symbol),
                "Type": "Log",
                "Mean (%)": f"{log_ret.mean() * 100:.4f}",
                "Ann. Mean (%)": f"{log_ret.mean() * ann_factor * 100:.2f}",
                "Std Dev (%)": f"{log_ret.std() * 100:.4f}",
                "Ann. Vol (%)": f"{log_ret.std() * np.sqrt(ann_factor) * 100:.2f}",
                "Skewness": f"{log_ret.skew():.3f}",
                "Kurtosis": f"{log_ret.kurtosis():.3f}",
            })

            stats_comparison.append({
                "Asset": config.get("name", symbol),
                "Type": "Arithmetic",
                "Mean (%)": f"{arith_ret.mean() * 100:.4f}",
                "Ann. Mean (%)": f"{arith_ret.mean() * ann_factor * 100:.2f}",
                "Std Dev (%)": f"{arith_ret.std() * 100:.4f}",
                "Ann. Vol (%)": f"{arith_ret.std() * np.sqrt(ann_factor) * 100:.2f}",
                "Skewness": f"{arith_ret.skew():.3f}",
                "Kurtosis": f"{arith_ret.kurtosis():.3f}",
            })

    if stats_comparison:
        st.dataframe(
            pd.DataFrame(stats_comparison),
            use_container_width=True,
            hide_index=True
        )

    # 4. Difference analysis
    st.markdown("### 📉 Difference: Arithmetic - Log Returns")

    st.markdown("""
    The difference between arithmetic and log returns increases with volatility.
    **Rule of thumb:** Arithmetic ≈ Log + (Volatility²/2)
    """)

    fig_diff = go.Figure()

    for symbol in selected_commodities_filtered:
        if symbol in log_returns_df.columns:
            log_ret = log_returns_df[symbol].dropna()
            arith_ret = arith_returns_df[symbol].dropna()

            common_idx = log_ret.index.intersection(arith_ret.index)
            difference = (arith_ret.loc[common_idx] -
                          log_ret.loc[common_idx]) * 100

            config = COMMODITIES_CONFIG.get(symbol, {})

            fig_diff.add_trace(
                go.Scatter(
                    x=difference.index,
                    y=difference.values,
                    name=config.get("name", symbol),
                    mode="lines",
                    line=dict(width=1),
                )
            )

    fig_diff.update_layout(
        title="Difference: Arithmetic - Log Returns",
        xaxis_title="Date",
        yaxis_title="Difference (%)",
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(fig_diff, use_container_width=True)

elif analysis_type == "Cumulative Wealth (NAV)":
    st.subheader("💰 Cumulative Wealth (NAV Path)")

    st.info("""
    **NAV (Net Asset Value)** shows the actual dollar value of an investment over time.
    This is what investors actually see in their accounts!
    """)

    # Check if we have enough data
    if len(date_filtered_df) < 2:
        st.warning(
            "⚠️ Insufficient data for NAV analysis. Please select a longer date range.")
        st.stop()

    # Initial investment amount
    initial_capital = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=100,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Starting capital for the investment"
    )

    # Calculate NAV using geometric returns
    st.markdown(f"### 📈 NAV Path (Starting: ${initial_capital:,.0f})")

    returns_df = date_filtered_df.pct_change().dropna()
    nav_df = (1 + returns_df).cumprod() * initial_capital

    fig = go.Figure()

    for symbol in selected_commodities:
        if symbol in nav_df.columns:
            nav_series = nav_df[symbol].dropna()
            config = COMMODITIES_CONFIG.get(symbol, {})

            fig.add_trace(
                go.Scatter(
                    x=nav_series.index,
                    y=nav_series.values,
                    name=config.get("name", symbol),
                    mode="lines",
                    line=dict(width=2),
                    hovertemplate="%{y:$,.0f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"Cumulative Wealth Path (Initial: ${initial_capital:,.0f})",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # 2. Performance table with dollar values
    st.markdown("### 💵 Investment Performance")

    perf_data = []
    for symbol in selected_commodities:
        if symbol in nav_df.columns:
            nav_series = nav_df[symbol].dropna()

            if len(nav_series) > 0:
                config = COMMODITIES_CONFIG.get(symbol, {})

                final_value = nav_series.iloc[-1]
                total_return_dollars = final_value - initial_capital
                total_return_pct = (final_value / initial_capital - 1) * 100

                # Calculate CAGR (geometric)
                years = len(
                    nav_series) / {"Daily": 252, "Weekly": 52, "Monthly": 12}.get(resample_freq, 252)
                cagr = ((final_value / initial_capital) **
                        (1 / years) - 1) * 100 if years > 0 else 0

                perf_data.append({
                    "Asset": config.get("name", symbol),
                    "Start Value": f"${initial_capital:,.0f}",
                    "End Value": f"${final_value:,.0f}",
                    "P&L ($)": f"${total_return_dollars:+,.0f}",
                    "Total Return (%)": f"{total_return_pct:+.2f}%",
                    "CAGR (%)": f"{cagr:.2f}%",
                    "Period (Years)": f"{years:.2f}",
                })

    if perf_data:
        st.dataframe(pd.DataFrame(perf_data),
                     use_container_width=True, hide_index=True)

    # 3. Bar chart of final values
    st.markdown("### 📊 Final Portfolio Values")

    final_values = []
    labels = []

    for symbol in selected_commodities:
        if symbol in nav_df.columns:
            nav_series = nav_df[symbol].dropna()
            if len(nav_series) > 0:
                config = COMMODITIES_CONFIG.get(symbol, {})
                final_values.append(nav_series.iloc[-1])
                labels.append(config.get("name", symbol))

    fig_bar = go.Figure(
        go.Bar(
            x=labels,
            y=final_values,
            marker_color=["green" if v >
                          initial_capital else "red" for v in final_values],
            text=[f"${v:,.0f}" for v in final_values],
            textposition="outside",
        )
    )

    # Add horizontal line at initial capital
    fig_bar.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Initial: ${initial_capital:,.0f}",
    )

    fig_bar.update_layout(
        title="Final Portfolio Values Comparison",
        xaxis_title="Asset",
        yaxis_title="Final Value ($)",
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(fig_bar, use_container_width=True)

elif analysis_type == "Drawdown Analysis":
    st.subheader("📉 Drawdown Analysis")

    st.info("""
    **Drawdown** = Peak-to-trough decline in portfolio value.
    Critical for understanding downside risk and capital preservation.
    """)

    # Check if we have enough data
    if len(date_filtered_df) < 2:
        st.warning(
            "⚠️ Insufficient data for drawdown analysis. Please select a longer date range.")
        st.stop()

    # Calculate drawdowns
    returns_df = date_filtered_df.pct_change().dropna()
    cum_returns = (1 + returns_df).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown_df = (cum_returns - running_max) / running_max

    # 1. Drawdown time series
    st.markdown("### 📉 Drawdown Over Time")

    fig = go.Figure()

    for symbol in selected_commodities:
        if symbol in drawdown_df.columns:
            dd_series = drawdown_df[symbol].dropna()
            config = COMMODITIES_CONFIG.get(symbol, {})

            fig.add_trace(
                go.Scatter(
                    x=dd_series.index,
                    y=dd_series.values * 100,
                    name=config.get("name", symbol),
                    mode="lines",
                    fill="tozeroy",
                    line=dict(width=1),
                    hovertemplate="%{y:.2f}%<extra></extra>",
                )
            )

    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # 2. Max drawdown statistics
    st.markdown("### 📊 Maximum Drawdown Statistics")

    dd_stats = []
    for symbol in selected_commodities:
        if symbol in drawdown_df.columns:
            dd_series = drawdown_df[symbol].dropna()

            if len(dd_series) > 0:
                config = COMMODITIES_CONFIG.get(symbol, {})

                max_dd = dd_series.min() * 100
                max_dd_date = dd_series.idxmin()

                # Find peak before max drawdown
                dd_values = dd_series.values
                max_dd_idx = dd_series.argmin()
                peak_idx = dd_series[:max_dd_idx].last_valid_index(
                ) if max_dd_idx > 0 else dd_series.index[0]

                # Calculate recovery
                if max_dd_idx < len(dd_series) - 1:
                    recovery_series = dd_series[max_dd_idx:]
                    # Within 1% of peak
                    recovered = (recovery_series > -0.01).any()
                    if recovered:
                        recovery_idx = recovery_series[recovery_series > -0.01].index[0]
                        recovery_days = (recovery_idx - max_dd_date).days
                        recovery_status = f"{recovery_days} days"
                    else:
                        recovery_status = "Not recovered"
                else:
                    recovery_status = "At trough"

                # Current drawdown
                current_dd = dd_series.iloc[-1] * 100

                dd_stats.append({
                    "Asset": config.get("name", symbol),
                    "Max Drawdown": f"{max_dd:.2f}%",
                    "Max DD Date": max_dd_date.strftime("%Y-%m-%d"),
                    "Recovery Time": recovery_status,
                    "Current DD": f"{current_dd:.2f}%",
                })

    if dd_stats:
        st.dataframe(pd.DataFrame(dd_stats),
                     use_container_width=True, hide_index=True)

    # 3. Drawdown duration analysis
    st.markdown("### ⏱️ Drawdown Duration Distribution")

    fig_duration = go.Figure()

    for symbol in selected_commodities:
        if symbol in drawdown_df.columns:
            dd_series = drawdown_df[symbol].dropna()
            config = COMMODITIES_CONFIG.get(symbol, {})

            # Identify drawdown periods (when DD < -1%)
            in_drawdown = dd_series < -0.01
            drawdown_periods = []

            if in_drawdown.any():
                # Find start and end of each drawdown period
                dd_starts = in_drawdown & ~in_drawdown.shift(
                    1, fill_value=False)
                dd_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)

                start_dates = dd_starts[dd_starts].index.tolist()
                end_dates = dd_ends[dd_ends].index.tolist()

                # Handle case where we're still in drawdown
                if len(start_dates) > len(end_dates):
                    end_dates.append(dd_series.index[-1])

                for start, end in zip(start_dates, end_dates):
                    duration = (end - start).days
                    drawdown_periods.append(duration)

                if drawdown_periods:
                    fig_duration.add_trace(
                        go.Histogram(
                            x=drawdown_periods,
                            name=config.get("name", symbol),
                            opacity=0.7,
                            nbinsx=20,
                        )
                    )

    fig_duration.update_layout(
        title="Drawdown Duration Distribution",
        xaxis_title="Duration (Days)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(fig_duration, use_container_width=True)

    # 4. Underwater plot (current drawdown status)
    st.markdown("### 🌊 Underwater Plot (Current Status)")

    fig_underwater = go.Figure()

    for symbol in selected_commodities:
        if symbol in drawdown_df.columns:
            dd_series = drawdown_df[symbol].dropna()
            config = COMMODITIES_CONFIG.get(symbol, {})

            # Color by severity
            colors = []
            for dd in dd_series:
                if dd > -0.05:
                    colors.append("green")
                elif dd > -0.15:
                    colors.append("orange")
                else:
                    colors.append("red")

            fig_underwater.add_trace(
                go.Scatter(
                    x=dd_series.index,
                    y=dd_series.values * 100,
                    name=config.get("name", symbol),
                    mode="lines",
                    line=dict(width=2),
                    fill="tozeroy",
                )
            )

    fig_underwater.update_layout(
        title="Underwater Plot - How Far Below Peak?",
        xaxis_title="Date",
        yaxis_title="Drawdown from Peak (%)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig_underwater, use_container_width=True)

elif analysis_type == "Risk Metrics Dashboard":
    st.subheader("📊 Comprehensive Risk Metrics")

    st.info("""
    **Beyond Sharpe Ratio:** A complete view of risk-adjusted performance.
    """)

    # Check if we have enough data
    if len(date_filtered_df) < 2:
        st.warning(
            "⚠️ Insufficient data for risk metrics analysis. Please select a longer date range.")
        st.stop()

    # Calculate all metrics
    returns_df = date_filtered_df.pct_change().dropna()

    periods_per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12}
    ann_factor = periods_per_year.get(resample_freq, 252)

    # Drawdowns for Calmar
    cum_returns = (1 + returns_df).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown_df = (cum_returns - running_max) / running_max

    risk_metrics = []

    for symbol in selected_commodities:
        if symbol in returns_df.columns:
            ret_series = returns_df[symbol].dropna()

            if len(ret_series) > 1:
                config = COMMODITIES_CONFIG.get(symbol, {})

                # Basic metrics
                mean_return = ret_series.mean()
                ann_return = mean_return * ann_factor
                volatility = ret_series.std()
                ann_vol = volatility * np.sqrt(ann_factor)

                # Sharpe Ratio
                sharpe = (mean_return / volatility) * \
                    np.sqrt(ann_factor) if volatility > 0 else 0

                # Sortino Ratio (downside deviation)
                downside_returns = ret_series[ret_series < 0]
                downside_std = downside_returns.std() if len(
                    downside_returns) > 0 else volatility
                sortino = (mean_return / downside_std) * \
                    np.sqrt(ann_factor) if downside_std > 0 else 0

                # Max Drawdown
                dd_series = drawdown_df[symbol].dropna()
                max_dd = abs(dd_series.min()) if len(dd_series) > 0 else 0

                # Calmar Ratio (CAGR / Max DD)
                years = len(ret_series) / ann_factor
                final_cum = (1 + ret_series).prod()
                cagr = (final_cum ** (1 / years) - 1) if years > 0 else 0
                calmar = (cagr / max_dd) if max_dd > 0 else 0

                # Value at Risk (95% and 99%)
                var_95 = np.percentile(ret_series, 5)
                var_99 = np.percentile(ret_series, 1)

                # Conditional VaR (CVaR / Expected Shortfall)
                cvar_95 = ret_series[ret_series <= var_95].mean() if (
                    ret_series <= var_95).any() else var_95
                cvar_99 = ret_series[ret_series <= var_99].mean() if (
                    ret_series <= var_99).any() else var_99

                # Skewness and Kurtosis
                skewness = ret_series.skew()
                kurtosis = ret_series.kurtosis()

                risk_metrics.append({
                    "Asset": config.get("name", symbol),
                    "Ann. Return": f"{ann_return * 100:.2f}%",
                    "Ann. Vol": f"{ann_vol * 100:.2f}%",
                    "Sharpe": f"{sharpe:.2f}",
                    "Sortino": f"{sortino:.2f}",
                    "Max DD": f"{max_dd * 100:.2f}%",
                    "Calmar": f"{calmar:.2f}",
                    "VaR 95%": f"{var_95 * 100:.2f}%",
                    "CVaR 95%": f"{cvar_95 * 100:.2f}%",
                    "VaR 99%": f"{var_99 * 100:.2f}%",
                    "CVaR 99%": f"{cvar_99 * 100:.2f}%",
                    "Skewness": f"{skewness:.3f}",
                    "Kurtosis": f"{kurtosis:.3f}",
                })

    if risk_metrics:
        st.dataframe(
            pd.DataFrame(risk_metrics),
            use_container_width=True,
            hide_index=True
        )

    # Visualization: Sharpe vs Sortino vs Calmar
    st.markdown("### 📈 Risk-Adjusted Returns Comparison")

    if risk_metrics:
        df_metrics = pd.DataFrame(risk_metrics)

        # Extract numeric values
        assets = df_metrics["Asset"].tolist()
        sharpe_vals = [float(x) for x in df_metrics["Sharpe"]]
        sortino_vals = [float(x) for x in df_metrics["Sortino"]]
        calmar_vals = [float(x) for x in df_metrics["Calmar"]]

        fig_compare = go.Figure()

        fig_compare.add_trace(go.Bar(
            x=assets,
            y=sharpe_vals,
            name="Sharpe Ratio",
            marker_color="lightblue",
        ))

        fig_compare.add_trace(go.Bar(
            x=assets,
            y=sortino_vals,
            name="Sortino Ratio",
            marker_color="purple",
        ))

        fig_compare.add_trace(go.Bar(
            x=assets,
            y=calmar_vals,
            name="Calmar Ratio",
            marker_color="orange",
        ))

        fig_compare.update_layout(
            title="Risk-Adjusted Returns: Sharpe vs Sortino vs Calmar",
            xaxis_title="Asset",
            yaxis_title="Ratio",
            barmode="group",
            height=500,
            template="plotly_white",
        )

        st.plotly_chart(fig_compare, use_container_width=True)

    # Interpretation guide
    with st.expander("ℹ️ Understanding Risk Metrics"):
        st.markdown("""
        ### Risk Metric Interpretation Guide
        
        **Sharpe Ratio**
        - Measures return per unit of total risk (volatility)
        - > 1.0 is good, > 2.0 is excellent
        - Treats upside and downside volatility equally
        
        **Sortino Ratio**
        - Measures return per unit of downside risk only
        - Better for asymmetric returns
        - Higher than Sharpe if returns are positively skewed
        
        **Calmar Ratio**
        - CAGR divided by Maximum Drawdown
        - Focuses on worst-case scenario
        - > 0.5 is good, > 1.0 is excellent
        
        **Value at Risk (VaR)**
        - Worst expected loss at X% confidence
        - VaR 95% = loss exceeded only 5% of the time
        - More negative = more downside risk
        
        **Conditional VaR (CVaR)**
        - Average loss when VaR is exceeded
        - Also called "Expected Shortfall"
        - Better measure of tail risk than VaR
        
        **Skewness**
        - Negative: more large losses than large gains
        - Positive: more large gains than large losses
        - Investors prefer positive skew
        
        **Kurtosis**
        - > 0: fatter tails than normal distribution
        - Higher kurtosis = more extreme events
        - Important for risk management
        """)

elif analysis_type == "Rolling Metrics":
    st.subheader("📈 Rolling Metrics Analysis")

    st.info("""
    **Time-varying risk metrics** help identify regime changes and periods of elevated risk.
    """)

    # Check if we have enough data
    if len(date_filtered_df) < 20:
        st.warning(
            "⚠️ Insufficient data for rolling metrics analysis. Please select a longer date range (minimum 20 periods).")
        st.stop()

    # Rolling window size
    default_window = {"Daily": 252, "Weekly": 52,
                      "Monthly": 12}.get(resample_freq, 252)

    rolling_window = st.sidebar.slider(
        "Rolling Window",
        min_value=max(20, default_window // 4),
        max_value=default_window * 2,
        value=default_window,
        step=default_window // 4,
        help=f"Window size for rolling calculations ({resample_freq} periods)"
    )

    returns_df = date_filtered_df.pct_change().dropna()

    periods_per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12}
    ann_factor = periods_per_year.get(resample_freq, 252)

    # Calculate rolling metrics for each asset
    rolling_data = {}

    for symbol in selected_commodities:
        if symbol in returns_df.columns:
            ret_series = returns_df[symbol].dropna()

            # Rolling mean and vol
            rolling_mean = ret_series.rolling(
                rolling_window).mean() * ann_factor
            rolling_vol = ret_series.rolling(
                rolling_window).std() * np.sqrt(ann_factor)

            # Rolling Sharpe
            rolling_sharpe = (ret_series.rolling(rolling_window).mean() /
                              ret_series.rolling(rolling_window).std()) * np.sqrt(ann_factor)

            # Rolling Sortino
            def rolling_sortino(window_data):
                if len(window_data) < 2:
                    return np.nan
                downside = window_data[window_data < 0]
                if len(downside) == 0:
                    return np.nan
                downside_std = downside.std()
                if downside_std == 0:
                    return np.nan
                return (window_data.mean() / downside_std) * np.sqrt(ann_factor)

            rolling_sortino_vals = ret_series.rolling(
                rolling_window).apply(rolling_sortino, raw=False)

            rolling_data[symbol] = {
                "mean": rolling_mean,
                "vol": rolling_vol,
                "sharpe": rolling_sharpe,
                "sortino": rolling_sortino_vals,
            }

    # 1. Rolling Sharpe Ratio
    st.markdown(f"### 📊 Rolling Sharpe Ratio ({rolling_window}-period window)")

    fig_sharpe = go.Figure()

    for symbol in selected_commodities:
        if symbol in rolling_data:
            config = COMMODITIES_CONFIG.get(symbol, {})
            sharpe_series = rolling_data[symbol]["sharpe"]

            fig_sharpe.add_trace(
                go.Scatter(
                    x=sharpe_series.index,
                    y=sharpe_series.values,
                    name=config.get("name", symbol),
                    mode="lines",
                    line=dict(width=2),
                )
            )

    fig_sharpe.add_hline(y=0, line_dash="dash",
                         line_color="gray", annotation_text="Sharpe = 0")
    fig_sharpe.add_hline(y=1, line_dash="dot",
                         line_color="green", annotation_text="Sharpe = 1")

    fig_sharpe.update_layout(
        title=f"Rolling Sharpe Ratio ({rolling_window}-{resample_freq} Window)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(fig_sharpe, use_container_width=True)

    # 2. Rolling Sortino Ratio
    st.markdown(
        f"### 💜 Rolling Sortino Ratio ({rolling_window}-period window)")

    fig_sortino = go.Figure()

    for symbol in selected_commodities:
        if symbol in rolling_data:
            config = COMMODITIES_CONFIG.get(symbol, {})
            sortino_series = rolling_data[symbol]["sortino"]

            fig_sortino.add_trace(
                go.Scatter(
                    x=sortino_series.index,
                    y=sortino_series.values,
                    name=config.get("name", symbol),
                    mode="lines",
                    line=dict(width=2),
                )
            )

    fig_sortino.add_hline(y=0, line_dash="dash",
                          line_color="gray", annotation_text="Sortino = 0")
    fig_sortino.add_hline(y=1, line_dash="dot",
                          line_color="purple", annotation_text="Sortino = 1")

    fig_sortino.update_layout(
        title=f"Rolling Sortino Ratio ({rolling_window}-{resample_freq} Window)",
        xaxis_title="Date",
        yaxis_title="Sortino Ratio",
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(fig_sortino, use_container_width=True)

    # 3. Rolling Volatility
    st.markdown(f"### 📉 Rolling Volatility ({rolling_window}-period window)")

    fig_vol = go.Figure()

    for symbol in selected_commodities:
        if symbol in rolling_data:
            config = COMMODITIES_CONFIG.get(symbol, {})
            vol_series = rolling_data[symbol]["vol"]

            fig_vol.add_trace(
                go.Scatter(
                    x=vol_series.index,
                    y=vol_series.values * 100,
                    name=config.get("name", symbol),
                    mode="lines",
                    line=dict(width=2),
                )
            )

    fig_vol.update_layout(
        title=f"Rolling Annualized Volatility ({rolling_window}-{resample_freq} Window)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(fig_vol, use_container_width=True)

    # Regime Classification
    st.markdown("---")
    st.markdown("### 🎯 Automatic Regime Classification")

    # Classify regimes for each commodity
    regime_data = {}
    for symbol in selected_commodities:
        if symbol in rolling_data:
            # Convert to percentage
            vol_series = rolling_data[symbol]["vol"] * 100

            # Calculate percentiles for regime thresholds
            vol_25 = vol_series.quantile(0.33)
            vol_75 = vol_series.quantile(0.67)

            # Classify regimes
            regimes = pd.Series(index=vol_series.index, dtype=str)
            regimes[vol_series <= vol_25] = "Low Volatility"
            regimes[(vol_series > vol_25) & (
                vol_series <= vol_75)] = "Medium Volatility"
            regimes[vol_series > vol_75] = "High Volatility"

            regime_data[symbol] = {
                'volatility': vol_series,
                'regimes': regimes,
                'threshold_low': vol_25,
                'threshold_high': vol_75,
            }

    # Show current regime for each commodity
    st.markdown("#### 📊 Current Market Regime")

    cols = st.columns(len(selected_commodities))
    for idx, symbol in enumerate(selected_commodities):
        if symbol in regime_data:
            config = COMMODITIES_CONFIG.get(symbol, {})
            current_regime = regime_data[symbol]['regimes'].iloc[-1]
            current_vol = regime_data[symbol]['volatility'].iloc[-1]

            with cols[idx]:
                # Color-code regime
                if current_regime == "Low Volatility":
                    st.success(f"""
                    **{config.get('name', symbol)}**
                    
                    🟢 **{current_regime}**
                    
                    Volatility: {current_vol:.1f}%
                    """)
                elif current_regime == "Medium Volatility":
                    st.info(f"""
                    **{config.get('name', symbol)}**
                    
                    🟡 **{current_regime}**
                    
                    Volatility: {current_vol:.1f}%
                    """)
                else:  # High Volatility
                    st.warning(f"""
                    **{config.get('name', symbol)}**
                    
                    🔴 **{current_regime}**
                    
                    Volatility: {current_vol:.1f}%
                    """)

    # Regime duration statistics
    st.markdown("---")
    st.markdown("#### ⏱️ Regime Duration Statistics")

    for symbol in selected_commodities:
        if symbol in regime_data:
            config = COMMODITIES_CONFIG.get(symbol, {})
            regimes = regime_data[symbol]['regimes']

            # Calculate regime durations
            regime_changes = regimes != regimes.shift(1)
            regime_blocks = regime_changes.cumsum()

            durations = []
            regime_types = []

            for block_id in regime_blocks.unique():
                if pd.isna(block_id):
                    continue
                block = regimes[regime_blocks == block_id]
                if len(block) > 0:
                    durations.append(len(block))
                    regime_types.append(block.iloc[0])

            # Calculate statistics
            regime_stats = pd.DataFrame({
                'Duration': durations,
                'Regime': regime_types
            })

            avg_durations = regime_stats.groupby(
                'Regime')['Duration'].agg(['mean', 'median', 'max'])
            regime_counts = regime_stats['Regime'].value_counts()

            with st.expander(f"📊 {config.get('name', symbol)} - Regime Statistics"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Regime Frequency:**")
                    for regime in ["Low Volatility", "Medium Volatility", "High Volatility"]:
                        count = regime_counts.get(regime, 0)
                        pct = (count / len(regime_counts)) * \
                            100 if len(regime_counts) > 0 else 0
                        st.metric(regime, f"{count} periods ({pct:.1f}%)")

                with col2:
                    st.markdown(f"**Average Duration ({freq_label}):**")
                    for regime in ["Low Volatility", "Medium Volatility", "High Volatility"]:
                        if regime in avg_durations.index:
                            mean_dur = avg_durations.loc[regime, 'mean']
                            st.metric(regime, f"{mean_dur:.1f} {freq_label}")

    # Regime-based insights and trading signals
    st.markdown("---")
    st.markdown("### 💡 Regime-Based Trading Insights")

    for symbol in selected_commodities:
        if symbol in regime_data:
            config = COMMODITIES_CONFIG.get(symbol, {})
            current_regime = regime_data[symbol]['regimes'].iloc[-1]
            current_vol = regime_data[symbol]['volatility'].iloc[-1]
            threshold_low = regime_data[symbol]['threshold_low']
            threshold_high = regime_data[symbol]['threshold_high']

            # Get recent regime history (last 10 periods)
            recent_regimes = regime_data[symbol]['regimes'].tail(10)
            # 70% same regime
            regime_stable = (recent_regimes == current_regime).sum() >= 7

            with st.expander(f"🎯 {config.get('name', symbol)} - Trading Strategy"):
                st.markdown(
                    f"#### Current Regime: **{current_regime}** (Volatility: {current_vol:.1f}%)")

                if current_regime == "Low Volatility":
                    st.success(f"""
                    **🟢 Low Volatility Regime Detected**
                    
                    **Market Characteristics:**
                    - Calm, stable market conditions
                    - Volatility below {threshold_low:.1f}%
                    - Trends tend to persist
                    - Noise is minimal
                    
                    **Recommended ML Strategy:**
                    1. **Data Frequency:** Daily or Weekly (high precision)
                    2. **Model:** XGBoost performs well (captures trends)
                    3. **Training Period:** 63-252 days (recent data sufficient)
                    4. **Features:** Emphasize momentum indicators
                    5. **Leverage:** Can use higher leverage
                    
                    **Trading Approach:**
                    - Momentum strategies excel
                    - Trend-following works well
                    - Breakout strategies effective
                    - Lower stop-loss thresholds acceptable
                    
                    **Risk Management:**
                    - Standard position sizing
                    - Wider stop losses (less noise)
                    - Focus on entry timing
                    """)

                elif current_regime == "Medium Volatility":
                    st.info(f"""
                    **🟡 Medium Volatility Regime Detected**
                    
                    **Market Characteristics:**
                    - Moderate volatility ({threshold_low:.1f}% - {threshold_high:.1f}%)
                    - Mixed market conditions
                    - Some noise but trends visible
                    - Normal trading environment
                    
                    **Recommended ML Strategy:**
                    1. **Data Frequency:** Weekly (balanced approach)
                    2. **Model:** Compare Both (XGBoost + LSTM)
                    3. **Training Period:** 104-208 weeks (2-4 years)
                    4. **Features:** Balanced momentum + mean reversion
                    5. **Leverage:** Moderate leverage
                    
                    **Trading Approach:**
                    - Hybrid strategies work best
                    - Combine momentum + mean reversion
                    - Adapt to sub-regime changes
                    - Technical analysis reliable
                    
                    **Risk Management:**
                    - Standard position sizing
                    - Moderate stop losses
                    - Monitor regime transitions
                    """)

                else:  # High Volatility
                    st.warning(f"""
                    **🔴 High Volatility Regime Detected**
                    
                    **Market Characteristics:**
                    - High volatility above {threshold_high:.1f}%
                    - Crisis or turbulent conditions
                    - Trends are choppy
                    - High noise-to-signal ratio
                    
                    **Recommended ML Strategy:**
                    1. **Data Frequency:** Weekly or Monthly (filter noise)
                    2. **Model:** LSTM (captures regime persistence)
                    3. **Training Period:** 208+ weeks or 60+ months (long history)
                    4. **Features:** Emphasize volatility, downside deviation
                    5. **Leverage:** Reduce significantly
                    
                    **Trading Approach:**
                    - Mean reversion strategies
                    - Volatility trading
                    - Options strategies (vega plays)
                    - Shorter holding periods
                    
                    **Risk Management:**
                    - ⚠️ CRITICAL: Reduce position sizes
                    - Tighter stop losses (manage drawdown)
                    - Increase cash allocation
                    - Consider hedging strategies
                    - Wait for regime confirmation before big bets
                    """)

                # Regime transition warning
                if not regime_stable:
                    st.warning(f"""
                    ⚠️ **Regime Transition Detected!**
                    
                    Recent {len(recent_regimes)} periods show regime instability.
                    
                    **Action Items:**
                    - Reduce position sizes
                    - Wait for regime to stabilize
                    - Use tighter risk controls
                    - Consider using ensemble predictions
                    - Monitor volatility closely
                    """)
                else:
                    st.success(f"""
                    ✅ **Stable Regime** (70%+ consistency in last 10 periods)
                    
                    Regime has been stable - strategies appropriate for current regime
                    are more likely to work effectively.
                    """)

    # Storytelling section
    st.markdown("---")
    st.markdown("### 📖 Regime Detection Story: Real-World Example")

    st.markdown("""
    **March 2020 - COVID-19 Market Crash:**
    
    **Before Crisis (January-February 2020):**
    - 🟢 **Low Volatility Regime**: Gold volatility ~12%
    - **ML Strategy**: Daily XGBoost, 63-day training, momentum features
    - **Model Performance**: 58% accuracy (good edge)
    - **Trading**: Momentum strategies, higher leverage
    - **Result**: Profitable trend-following
    
    **Crisis Begins (March 2020):**
    - 🔴 **Regime Transition Detected**: Volatility spikes to 35%+
    - **Automatic Classification**: HIGH VOLATILITY REGIME
    - **Signal Generated**: "⚠️ REDUCE LEVERAGE - High Vol Regime"
    
    **During Crisis (March-April 2020):**
    - 🔴 **High Volatility Regime**: Gold volatility ~40%
    - **ML Strategy Changed**: Weekly LSTM, 208-week training, volatility features
    - **Model Re-trained**: Focus on downside protection
    - **Model Performance**: 52% accuracy (lower but still edge)
    - **Trading**: Mean reversion, reduced position sizes, tighter stops
    - **Result**: Capital preserved, avoided major drawdowns
    
    **Recovery (May-June 2020):**
    - 🟡 **Medium Volatility Regime**: Volatility drops to 20%
    - **Signal**: "Regime stabilizing - gradually increase positions"
    - **ML Strategy**: Hybrid (Compare Both models)
    - **Trading**: Cautious re-entry, balanced strategies
    
    **New Normal (July 2020+):**
    - 🟢 **Low Volatility Regime**: Volatility back to 15%
    - **Signal**: "✅ Stable regime - resume normal strategies"
    - **ML Strategy**: Back to daily XGBoost, momentum focus
    - **Result**: Full recovery and new profits
    
    ---
    
    **The Key Insight:**
    
    Without regime detection → Same strategy throughout → -30% drawdown in March
    
    With regime detection → Adaptive strategy → -5% drawdown, quick recovery
    
    **This is why regime-aware trading matters!**
    """)

    # 4. Rolling correlation (if multiple assets selected)
    if len(selected_commodities) >= 2:
        st.markdown(
            f"### 🔗 Rolling Correlation ({rolling_window}-period window)")

        st.markdown("Select two assets to compare:")

        col1, col2 = st.columns(2)

        with col1:
            asset1 = st.selectbox(
                "Asset 1",
                selected_commodities,
                index=0,
                key="rolling_corr_asset1"
            )

        with col2:
            other_assets = [s for s in selected_commodities if s != asset1]
            asset2 = st.selectbox(
                "Asset 2",
                other_assets,
                index=0 if other_assets else None,
                key="rolling_corr_asset2"
            )

        if asset1 and asset2 and asset1 in returns_df.columns and asset2 in returns_df.columns:
            ret1 = returns_df[asset1].dropna()
            ret2 = returns_df[asset2].dropna()

            # Align indices
            common_idx = ret1.index.intersection(ret2.index)
            ret1 = ret1.loc[common_idx]
            ret2 = ret2.loc[common_idx]

            rolling_corr = ret1.rolling(rolling_window).corr(ret2)

            config1 = COMMODITIES_CONFIG.get(asset1, {})
            config2 = COMMODITIES_CONFIG.get(asset2, {})

            fig_corr = go.Figure()

            fig_corr.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode="lines",
                    line=dict(width=2, color="blue"),
                    fill="tozeroy",
                    name="Correlation",
                )
            )

            fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_corr.add_hline(y=0.5, line_dash="dot",
                               line_color="green", annotation_text="Corr = 0.5")
            fig_corr.add_hline(y=-0.5, line_dash="dot",
                               line_color="red", annotation_text="Corr = -0.5")

            fig_corr.update_layout(
                title=f"Rolling Correlation: {config1.get('name', asset1)} vs {config2.get('name', asset2)}",
                xaxis_title="Date",
                yaxis_title="Correlation",
                height=500,
                template="plotly_white",
            )

            st.plotly_chart(fig_corr, use_container_width=True)

elif analysis_type == "Return Distribution":
    st.subheader("📊 Return Distribution Analysis")

    st.info("""
    **Understanding return distributions** helps identify tail risk, skewness, and deviation from normality.
    """)

    # Check if we have enough data
    if len(date_filtered_df) < 30:
        st.warning(
            "⚠️ Insufficient data for distribution analysis. Please select a longer date range (minimum 30 periods).")
        st.stop()

    returns_df = date_filtered_df.pct_change().dropna()
    log_returns_df = np.log(
        date_filtered_df / date_filtered_df.shift(1)).dropna()

    # 1. Histogram comparison: Log vs Arithmetic
    st.markdown("### 📈 Log Returns vs Arithmetic Returns Distribution")

    for symbol in selected_commodities:
        if symbol in returns_df.columns:
            config = COMMODITIES_CONFIG.get(symbol, {})
            st.markdown(f"#### {config.get('name', symbol)}")

            arith_ret = returns_df[symbol].dropna() * 100
            log_ret = log_returns_df[symbol].dropna() * 100

            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=arith_ret,
                    name="Arithmetic Returns",
                    opacity=0.6,
                    nbinsx=50,
                    marker_color="lightblue",
                )
            )

            fig.add_trace(
                go.Histogram(
                    x=log_ret,
                    name="Log Returns",
                    opacity=0.6,
                    nbinsx=50,
                    marker_color="orange",
                )
            )

            fig.update_layout(
                title=f"Return Distribution: {config.get('name', symbol)}",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                barmode="overlay",
                height=400,
                template="plotly_white",
            )

            st.plotly_chart(fig, use_container_width=True)

    # 2. Q-Q Plot (Quantile-Quantile) - Check for normality
    st.markdown("### 🎯 Q-Q Plot: Are Returns Normal?")

    st.markdown("""
    **Q-Q Plot** compares actual returns to theoretical normal distribution.
    - Points on the line = normal distribution
    - Points above line = more extreme positive returns than expected
    - Points below line = more extreme negative returns than expected
    """)

    for symbol in selected_commodities:
        if symbol in returns_df.columns:
            config = COMMODITIES_CONFIG.get(symbol, {})
            ret_series = returns_df[symbol].dropna()

            # Calculate theoretical quantiles
            sorted_returns = np.sort(ret_series)
            theoretical_quantiles = stats.norm.ppf(
                np.linspace(0.01, 0.99, len(sorted_returns)))

            fig_qq = go.Figure()

            # Scatter plot
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_returns,
                    mode="markers",
                    marker=dict(size=4, color="blue", opacity=0.5),
                    name="Actual",
                )
            )

            # 45-degree line (perfect normality)
            min_val = min(theoretical_quantiles.min(), sorted_returns.min())
            max_val = max(theoretical_quantiles.max(), sorted_returns.max())

            fig_qq.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="red", dash="dash", width=2),
                    name="Normal",
                )
            )

            fig_qq.update_layout(
                title=f"Q-Q Plot: {config.get('name', symbol)}",
                xaxis_title="Theoretical Quantiles (Normal)",
                yaxis_title="Actual Quantiles",
                height=500,
                template="plotly_white",
            )

            st.plotly_chart(fig_qq, use_container_width=True)

    # 3. Distribution statistics table
    st.markdown("### 📊 Distribution Statistics")

    dist_stats = []

    for symbol in selected_commodities:
        if symbol in returns_df.columns:
            ret_series = returns_df[symbol].dropna() * 100

            config = COMMODITIES_CONFIG.get(symbol, {})

            # Normality test (Jarque-Bera)
            jb_stat, jb_pvalue = stats.jarque_bera(ret_series)
            is_normal = "Yes" if jb_pvalue > 0.05 else "No"

            dist_stats.append({
                "Asset": config.get("name", symbol),
                "Mean": f"{ret_series.mean():.3f}%",
                "Median": f"{ret_series.median():.3f}%",
                "Std Dev": f"{ret_series.std():.3f}%",
                "Skewness": f"{ret_series.skew():.3f}",
                "Kurtosis": f"{ret_series.kurtosis():.3f}",
                "Min": f"{ret_series.min():.2f}%",
                "Max": f"{ret_series.max():.2f}%",
                "Normal?": is_normal,
                "JB p-value": f"{jb_pvalue:.4f}",
            })

    if dist_stats:
        st.dataframe(
            pd.DataFrame(dist_stats),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("ℹ️ Interpreting Distribution Statistics"):
        st.markdown("""
        **Skewness:**
        - = 0: Symmetric distribution
        - < 0: Left skew (more large losses)
        - > 0: Right skew (more large gains)
        
        **Kurtosis:**
        - = 0: Normal distribution tails
        - > 0: Fatter tails (more extreme events)
        - < 0: Thinner tails (fewer extreme events)
        
        **Jarque-Bera Test:**
        - Tests if returns follow normal distribution
        - p-value > 0.05: Cannot reject normality
        - p-value < 0.05: Returns are NOT normal
        
        **Why This Matters:**
        - Most risk models assume normal returns
        - Fat tails mean more crashes than models predict
        - Negative skew means asymmetric downside risk
        """)

elif analysis_type == "Multi-Period Performance":
    st.subheader("📅 Multi-Period Performance")

    st.info("""
    **Performance across different time horizons** helps identify consistency and recent trends.
    """)

    # Check if we have enough data
    if len(date_filtered_df) < 2:
        st.warning(
            "⚠️ Insufficient data for multi-period analysis. Please select a longer date range.")
        st.stop()

    # Define periods
    today = date_filtered_df.index[-1]

    periods = {
        "1 Month": today - pd.DateOffset(months=1),
        "3 Months": today - pd.DateOffset(months=3),
        "6 Months": today - pd.DateOffset(months=6),
        "YTD": pd.Timestamp(today.year, 1, 1),
        "1 Year": today - pd.DateOffset(years=1),
        "3 Years": today - pd.DateOffset(years=3),
        "5 Years": today - pd.DateOffset(years=5),
        "Since Inception": date_filtered_df.index[0],
    }

    # Calculate returns for each period
    perf_data = []

    for symbol in selected_commodities:
        if symbol in date_filtered_df.columns:
            config = COMMODITIES_CONFIG.get(symbol, {})
            price_series = date_filtered_df[symbol].dropna()

            period_returns = {}

            for period_name, start_date in periods.items():
                # Filter to period
                period_data = price_series[price_series.index >= start_date]

                if len(period_data) >= 2:
                    # Calculate return
                    start_price = period_data.iloc[0]
                    end_price = period_data.iloc[-1]
                    period_return = (end_price / start_price - 1) * 100

                    period_returns[period_name] = f"{period_return:+.2f}%"
                else:
                    period_returns[period_name] = "N/A"

            perf_data.append({
                "Asset": config.get("name", symbol),
                **period_returns
            })

    if perf_data:
        df_perf = pd.DataFrame(perf_data)

        # Apply color styling
        def color_returns(val):
            if val == "N/A" or val == "Asset":
                return ""
            try:
                num_val = float(val.strip("+%"))
                if num_val > 0:
                    return "background-color: #d4edda; color: #155724"  # Green
                elif num_val < 0:
                    return "background-color: #f8d7da; color: #721c24"  # Red
                else:
                    return ""
            except:
                return ""

        st.dataframe(
            df_perf.style.applymap(color_returns),
            use_container_width=True,
            hide_index=True
        )

    # Visualization: Performance comparison
    st.markdown("### 📊 Performance Comparison Across Periods")

    if perf_data:
        # Create bar chart for each period
        fig = make_subplots(
            rows=len(periods),
            cols=1,
            subplot_titles=list(periods.keys()),
            vertical_spacing=0.05,
        )

        for idx, (period_name, _) in enumerate(periods.items(), 1):
            returns_for_period = []
            labels = []

            for row in perf_data:
                if row[period_name] != "N/A":
                    try:
                        ret_val = float(row[period_name].strip("+%"))
                        returns_for_period.append(ret_val)
                        labels.append(row["Asset"])
                    except:
                        pass

            if returns_for_period:
                colors = ["green" if r >
                          0 else "red" for r in returns_for_period]

                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=returns_for_period,
                        marker_color=colors,
                        showlegend=False,
                        text=[f"{r:+.1f}%" for r in returns_for_period],
                        textposition="outside",
                    ),
                    row=idx, col=1
                )

        fig.update_xaxes(title_text="Asset", row=len(periods), col=1)
        fig.update_yaxes(title_text="Return (%)")
        fig.update_layout(
            height=300 * len(periods),
            showlegend=False,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Annualized returns comparison
    st.markdown("### 📈 Annualized Returns (CAGR)")

    cagr_data = []

    for symbol in selected_commodities:
        if symbol in date_filtered_df.columns:
            config = COMMODITIES_CONFIG.get(symbol, {})
            price_series = date_filtered_df[symbol].dropna()

            cagr_by_period = {}

            for period_name, start_date in periods.items():
                period_data = price_series[price_series.index >= start_date]

                if len(period_data) >= 2:
                    start_price = period_data.iloc[0]
                    end_price = period_data.iloc[-1]

                    # Calculate years
                    days = (period_data.index[-1] - period_data.index[0]).days
                    years = days / 365.25

                    if years > 0:
                        cagr = ((end_price / start_price)
                                ** (1 / years) - 1) * 100
                        cagr_by_period[period_name] = f"{cagr:+.2f}%"
                    else:
                        cagr_by_period[period_name] = "N/A"
                else:
                    cagr_by_period[period_name] = "N/A"

            cagr_data.append({
                "Asset": config.get("name", symbol),
                **cagr_by_period
            })

    if cagr_data:
        df_cagr = pd.DataFrame(cagr_data)

        st.dataframe(
            df_cagr.style.applymap(color_returns),
            use_container_width=True,
            hide_index=True
        )

elif analysis_type == "Ratio Analysis":
    st.subheader("📊 Commodity Ratio Analysis")

    st.info("""
    **Price Ratios** are powerful trading indicators:
    - **Gold/Silver Ratio**: Classic precious metals indicator (normal range: 50-80)
    - **Energy Ratios**: Crack spreads, relative value
    - **Cross-Commodity**: Identify relative strength/weakness
    """)

    # Check if we have at least 2 commodities
    if len(selected_commodities) < 2:
        st.warning(
            "⚠️ Ratio analysis requires at least 2 commodities. Please select more assets.")
        st.stop()

    # Check data availability
    if len(date_filtered_df) < 2:
        st.warning(
            "⚠️ Insufficient data for ratio analysis. Please select a longer date range.")
        st.stop()

    # Filter to commodities with valid data
    valid_commodities = []
    for symbol in selected_commodities:
        if symbol in date_filtered_df.columns:
            valid_count = date_filtered_df[symbol].notna().sum()
            if valid_count > 1:
                valid_commodities.append(symbol)

    if len(valid_commodities) < 2:
        st.warning(
            "⚠️ Need at least 2 commodities with valid data. Please adjust your selection or date range.")
        st.stop()

    # Different behavior based on number of commodities
    num_commodities = len(valid_commodities)

    if num_commodities == 2:
        # Simple case: Show the single ratio
        st.markdown("### 📈 Price Ratio Over Time")

        numerator = valid_commodities[0]
        denominator = valid_commodities[1]

        config_num = COMMODITIES_CONFIG.get(numerator, {})
        config_denom = COMMODITIES_CONFIG.get(denominator, {})

        name_num = config_num.get("name", numerator)
        name_denom = config_denom.get("name", denominator)

        # Calculate ratio
        ratio = date_filtered_df[numerator] / date_filtered_df[denominator]
        ratio = ratio.dropna()

        if len(ratio) < 2:
            st.warning(
                f"⚠️ Insufficient overlapping data for {name_num}/{name_denom} ratio.")
            st.stop()

        # Plot ratio
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=ratio.index,
                y=ratio.values,
                mode="lines",
                line=dict(width=2, color="blue"),
                name=f"{name_num}/{name_denom}",
                hovertemplate="%{y:.2f}<extra></extra>",
            )
        )

        # Add mean line
        mean_ratio = ratio.mean()
        fig.add_hline(
            y=mean_ratio,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Mean: {mean_ratio:.2f}",
        )

        # Add std bands
        std_ratio = ratio.std()
        fig.add_hline(
            y=mean_ratio + std_ratio,
            line_dash="dot",
            line_color="green",
            annotation_text=f"+1 SD: {mean_ratio + std_ratio:.2f}",
        )
        fig.add_hline(
            y=mean_ratio - std_ratio,
            line_dash="dot",
            line_color="red",
            annotation_text=f"-1 SD: {mean_ratio - std_ratio:.2f}",
        )

        fig.update_layout(
            title=f"{name_num} / {name_denom} Ratio",
            xaxis_title="Date",
            yaxis_title="Ratio",
            hovermode="x unified",
            height=600,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        st.markdown("### 📊 Ratio Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Ratio", f"{ratio.iloc[-1]:.2f}")

        with col2:
            st.metric("Mean Ratio", f"{mean_ratio:.2f}")

        with col3:
            st.metric("Min Ratio", f"{ratio.min():.2f}")

        with col4:
            st.metric("Max Ratio", f"{ratio.max():.2f}")

        # Additional analysis
        st.markdown("### 📉 Ratio Distribution")

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Histogram(
                x=ratio.values,
                nbinsx=50,
                marker_color="lightblue",
                name="Ratio",
            )
        )

        # Add mean and std lines
        fig_hist.add_vline(x=mean_ratio, line_dash="dash",
                           line_color="red", annotation_text="Mean")
        fig_hist.add_vline(x=mean_ratio + std_ratio,
                           line_dash="dot", line_color="green")
        fig_hist.add_vline(x=mean_ratio - std_ratio,
                           line_dash="dot", line_color="green")

        fig_hist.update_layout(
            title=f"Distribution of {name_num}/{name_denom} Ratio",
            xaxis_title="Ratio",
            yaxis_title="Frequency",
            height=400,
            template="plotly_white",
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        # Trading signals (if within typical range)
        st.markdown("### 🎯 Current Status")

        current_ratio = ratio.iloc[-1]
        z_score = (current_ratio - mean_ratio) / std_ratio

        if z_score > 1:
            st.success(f"""
            📈 **Ratio is HIGH** (z-score: {z_score:.2f})
            
            Current: {current_ratio:.2f} vs Mean: {mean_ratio:.2f}
            
            **Interpretation:** {name_num} is expensive relative to {name_denom}
            - Consider: Long {name_denom}, Short {name_num} (mean reversion trade)
            - Or: {name_num} is outperforming (momentum trade)
            """)
        elif z_score < -1:
            st.info(f"""
            📉 **Ratio is LOW** (z-score: {z_score:.2f})
            
            Current: {current_ratio:.2f} vs Mean: {mean_ratio:.2f}
            
            **Interpretation:** {name_num} is cheap relative to {name_denom}
            - Consider: Long {name_num}, Short {name_denom} (mean reversion trade)
            - Or: {name_denom} is outperforming (momentum trade)
            """)
        else:
            st.warning(f"""
            ⚖️ **Ratio is NEUTRAL** (z-score: {z_score:.2f})
            
            Current: {current_ratio:.2f} vs Mean: {mean_ratio:.2f}
            
            **Interpretation:** Ratio is near its historical average
            - No strong relative value signal
            - Wait for more extreme readings
            """)

    elif num_commodities == 3:
        # Show all 3 possible ratios
        st.markdown("### 📈 All Possible Ratios (3 Commodities)")

        st.markdown(f"""
        With 3 commodities, there are 3 possible ratios:
        - {COMMODITIES_CONFIG.get(valid_commodities[0], {}).get('name', valid_commodities[0])} / {COMMODITIES_CONFIG.get(valid_commodities[1], {}).get('name', valid_commodities[1])}
        - {COMMODITIES_CONFIG.get(valid_commodities[0], {}).get('name', valid_commodities[0])} / {COMMODITIES_CONFIG.get(valid_commodities[2], {}).get('name', valid_commodities[2])}
        - {COMMODITIES_CONFIG.get(valid_commodities[1], {}).get('name', valid_commodities[1])} / {COMMODITIES_CONFIG.get(valid_commodities[2], {}).get('name', valid_commodities[2])}
        """)

        # Calculate all 3 ratios
        pairs = [
            (valid_commodities[0], valid_commodities[1]),
            (valid_commodities[0], valid_commodities[2]),
            (valid_commodities[1], valid_commodities[2]),
        ]

        fig = go.Figure()

        for num_sym, denom_sym in pairs:
            config_num = COMMODITIES_CONFIG.get(num_sym, {})
            config_denom = COMMODITIES_CONFIG.get(denom_sym, {})

            name_num = config_num.get("name", num_sym)
            name_denom = config_denom.get("name", denom_sym)

            ratio = date_filtered_df[num_sym] / date_filtered_df[denom_sym]
            ratio = ratio.dropna()

            if len(ratio) >= 2:
                # Normalize to start at 100 for comparison
                ratio_normalized = (ratio / ratio.iloc[0]) * 100

                fig.add_trace(
                    go.Scatter(
                        x=ratio_normalized.index,
                        y=ratio_normalized.values,
                        mode="lines",
                        name=f"{name_num}/{name_denom}",
                        line=dict(width=2),
                    )
                )

        fig.update_layout(
            title="Normalized Ratios (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Ratio",
            hovermode="x unified",
            height=600,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics table
        st.markdown("### 📊 Ratio Statistics")

        ratio_stats = []
        for num_sym, denom_sym in pairs:
            config_num = COMMODITIES_CONFIG.get(num_sym, {})
            config_denom = COMMODITIES_CONFIG.get(denom_sym, {})

            name_num = config_num.get("name", num_sym)
            name_denom = config_denom.get("name", denom_sym)

            ratio = date_filtered_df[num_sym] / date_filtered_df[denom_sym]
            ratio = ratio.dropna()

            if len(ratio) >= 2:
                ratio_stats.append({
                    "Ratio": f"{name_num}/{name_denom}",
                    "Current": f"{ratio.iloc[-1]:.2f}",
                    "Mean": f"{ratio.mean():.2f}",
                    "Std Dev": f"{ratio.std():.2f}",
                    "Min": f"{ratio.min():.2f}",
                    "Max": f"{ratio.max():.2f}",
                    "Z-Score": f"{(ratio.iloc[-1] - ratio.mean()) / ratio.std():.2f}",
                })

        if ratio_stats:
            st.dataframe(pd.DataFrame(ratio_stats),
                         use_container_width=True, hide_index=True)

    else:
        # 4+ commodities: Let user select which pair to analyze
        st.markdown("### 🔍 Select Ratio to Analyze")

        st.markdown(f"""
        You have selected **{num_commodities} commodities**.
        
        Choose which pair you'd like to analyze:
        """)

        # Create friendly names for selection
        commodity_names = {}
        for symbol in valid_commodities:
            config = COMMODITIES_CONFIG.get(symbol, {})
            commodity_names[config.get("name", symbol)] = symbol

        col1, col2 = st.columns(2)

        with col1:
            numerator_name = st.selectbox(
                "Numerator (Top)",
                list(commodity_names.keys()),
                key="ratio_numerator"
            )
            numerator = commodity_names[numerator_name]

        with col2:
            # Filter denominator options to exclude numerator
            denom_options = [
                name for name in commodity_names.keys() if name != numerator_name]
            denominator_name = st.selectbox(
                "Denominator (Bottom)",
                denom_options,
                key="ratio_denominator"
            )
            denominator = commodity_names[denominator_name]

        # Calculate selected ratio
        ratio = date_filtered_df[numerator] / date_filtered_df[denominator]
        ratio = ratio.dropna()

        if len(ratio) < 2:
            st.warning(
                f"⚠️ Insufficient overlapping data for {numerator_name}/{denominator_name} ratio.")
            st.stop()

        st.markdown(f"### 📈 {numerator_name} / {denominator_name} Ratio")

        # Plot ratio
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=ratio.index,
                y=ratio.values,
                mode="lines",
                line=dict(width=2, color="blue"),
                name=f"{numerator_name}/{denominator_name}",
                hovertemplate="%{y:.2f}<extra></extra>",
            )
        )

        # Add mean and std bands
        mean_ratio = ratio.mean()
        std_ratio = ratio.std()

        fig.add_hline(
            y=mean_ratio,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Mean: {mean_ratio:.2f}",
        )
        fig.add_hline(
            y=mean_ratio + std_ratio,
            line_dash="dot",
            line_color="green",
            annotation_text=f"+1 SD",
        )
        fig.add_hline(
            y=mean_ratio - std_ratio,
            line_dash="dot",
            line_color="red",
            annotation_text=f"-1 SD",
        )

        fig.update_layout(
            title=f"{numerator_name} / {denominator_name} Ratio",
            xaxis_title="Date",
            yaxis_title="Ratio",
            hovermode="x unified",
            height=600,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        st.markdown("### 📊 Ratio Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current", f"{ratio.iloc[-1]:.2f}")

        with col2:
            st.metric("Mean", f"{mean_ratio:.2f}")

        with col3:
            st.metric("Std Dev", f"{std_ratio:.2f}")

        with col4:
            z_score = (ratio.iloc[-1] - mean_ratio) / std_ratio
            st.metric("Z-Score", f"{z_score:.2f}")

        # Distribution
        st.markdown("### 📉 Ratio Distribution")

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Histogram(
                x=ratio.values,
                nbinsx=50,
                marker_color="lightblue",
            )
        )

        fig_hist.add_vline(x=mean_ratio, line_dash="dash",
                           line_color="red", annotation_text="Mean")

        fig_hist.update_layout(
            title=f"Distribution of {numerator_name}/{denominator_name}",
            xaxis_title="Ratio",
            yaxis_title="Frequency",
            height=400,
            template="plotly_white",
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        # Show all available ratios (summary table)
        st.markdown("### 📋 All Available Ratios")

        st.markdown(
            f"Quick reference for all {num_commodities * (num_commodities - 1) // 2} possible ratios:")

        all_ratios = []
        for i, num_sym in enumerate(valid_commodities):
            for denom_sym in valid_commodities[i+1:]:
                config_num = COMMODITIES_CONFIG.get(num_sym, {})
                config_denom = COMMODITIES_CONFIG.get(denom_sym, {})

                name_num = config_num.get("name", num_sym)
                name_denom = config_denom.get("name", denom_sym)

                r = date_filtered_df[num_sym] / date_filtered_df[denom_sym]
                r = r.dropna()

                if len(r) >= 2:
                    all_ratios.append({
                        "Ratio": f"{name_num}/{name_denom}",
                        "Current": f"{r.iloc[-1]:.2f}",
                        "Mean": f"{r.mean():.2f}",
                        "Z-Score": f"{(r.iloc[-1] - r.mean()) / r.std():.2f}",
                    })

        if all_ratios:
            st.dataframe(pd.DataFrame(all_ratios),
                         use_container_width=True, hide_index=True)

    # Educational section
    with st.expander("ℹ️ Understanding Commodity Ratios"):
        st.markdown("""
        ### What Are Commodity Ratios?
        
        **Definition:** Price of one commodity divided by price of another.
        
        ### Famous Ratios:
        
        **1. Gold/Silver Ratio**
        - Historical average: ~50-80
        - High ratio (>80): Silver is cheap relative to gold
        - Low ratio (<50): Gold is cheap relative to silver
        - Used by precious metals traders for centuries
        
        **2. Oil/Gold Ratio**
        - Measures energy vs monetary metal
        - Useful for inflation analysis
        
        **3. Copper/Gold Ratio**
        - "Dr. Copper" as economic indicator
        - High ratio: Economic growth (copper demand up)
        - Low ratio: Economic slowdown
        
        ### How to Use:
        
        **Mean Reversion Strategy:**
        - When ratio is HIGH (>+1 SD): Expect reversion down
          - Trade: Long denominator, Short numerator
        - When ratio is LOW (<-1 SD): Expect reversion up
          - Trade: Long numerator, Short denominator
        
        **Momentum Strategy:**
        - Rising ratio: Numerator outperforming
        - Falling ratio: Denominator outperforming
        
        **Z-Score Interpretation:**
        - |Z| < 1: Normal range
        - |Z| > 1: Somewhat extreme
        - |Z| > 2: Very extreme (strong signal)
        
        ### Limitations:
        
        - Ratios can trend for long periods
        - Mean reversion is not guaranteed
        - Consider fundamentals, not just statistics
        - Different commodities have different drivers
        """)

elif analysis_type == "ML Price Prediction":
    st.subheader("🤖 ML Price Direction Prediction")

    st.info("""
    **Machine Learning** to predict commodity price direction (up/down tomorrow).
    
    **Models:**
    - 🌳 XGBoost (tree-based, no scaling needed)
    - 🧠 LSTM (neural network, with StandardScaler)
    
    **Validation:** Walk-forward with expanding window (3-month start, 1-week test)
    """)

    # Import ML modules
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from data.ml_features import create_ml_features_with_transparency
        from models.commodity_direction import compare_models, run_walk_forward_validation
    except ImportError as e:
        st.error(f"❌ ML modules not available: {e}")
        st.stop()

    # ML requires single commodity selection
    if len(selected_commodities) != 1:
        st.warning("""
        ⚠️ **ML Price Prediction requires exactly ONE commodity.**
        
        Please select a single commodity from the sidebar to continue.
        """)
        st.stop()

    symbol = selected_commodities[0]
    config = COMMODITIES_CONFIG.get(symbol, {})
    commodity_name = config.get("name", symbol)

    st.markdown(f"### 🎯 Predicting: {commodity_name}")
    
    # ============================================================================
    # DETECT REGIME FIRST (Before configuration)
    # ============================================================================
    st.markdown("### 🎯 Step 1: Detect Current Market Regime")
    
    with st.spinner("Analyzing current market conditions..."):
        # Calculate rolling volatility to detect regime
        price_series = date_filtered_df[symbol].dropna()
        returns = np.log(price_series / price_series.shift(1)).dropna()
        
        # Calculate volatility for regime detection (use 63-day window)
        rolling_vol = returns.rolling(63).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1] * 100
        
        # Import regime functions
        from data.ml_features import classify_volatility_regime, get_regime_trading_recommendation
        
        current_regime = classify_volatility_regime(current_vol / 100, rolling_vol)
        regime_rec = get_regime_trading_recommendation(current_regime)
    
    # Display detected regime prominently
    regime_color_map = {
        "Low Volatility": "success",
        "Medium Volatility": "info",
        "High Volatility": "warning"
    }
    
    regime_method = getattr(st, regime_color_map[current_regime])
    regime_method(f"""
    **{regime_rec['emoji']} Current Regime: {current_regime}**
    
    Current volatility: **{current_vol:.1f}%** annualized
    
    **🎯 Regime-Based Recommendations:**
    - 📊 Data frequency: **{regime_rec['frequency']}**
    - 🤖 Best model: **{regime_rec['model']}**
    - 📅 Training period: **{regime_rec['training']}**
    - 🎯 Focus features: {regime_rec['features']}
    
    **These recommendations are now applied as defaults in the sidebar! ➡️**
    """)
    
    # Show current date range info
    st.markdown("---")
    st.markdown("### 📅 Step 2: Review Data Range")
    
    st.info(f"""
    📅 **Current Date Range:** {date_filtered_df.index[0].strftime('%Y-%m-%d')} to {date_filtered_df.index[-1].strftime('%Y-%m-%d')}
    
    **Total periods available:** {len(date_filtered_df)} days ({len(date_filtered_df) / 365.25:.1f} years)
    
    💡 **Tip:** Adjust "Training Data Selection" in sidebar if you want more/less history.
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Step 3: Configure & Run (Sidebar →)")
    
    st.info("""
    **Ready to run with regime-optimized defaults!**
    
    Check the sidebar → All parameters are pre-configured based on the detected regime.
    
    You can:
    - ✅ Click "Run ML Prediction" immediately (recommended)
    - 📝 Or customize parameters first if needed
    """)

    # Check data availability
    if len(date_filtered_df) < 100:
        st.warning(f"""
        ⚠️ **Insufficient data for ML training.**
        
        Available: {len(date_filtered_df)} days
        Minimum needed: 100 days (preferably 252+ for reliable results)
        
        Please select a longer date range.
        """)
        st.stop()

    # ============================================================================
    # SIDEBAR CONFIGURATION (Regime info at top)
    # ============================================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 ML Configuration")
    
    st.sidebar.success(f"""
    ✅ **Regime Detected: {current_regime}**
    
    Defaults below are optimized for current market conditions.
    """)
    
    # ============================================================================
    # REGIME-AWARE DEFAULT INDEXES (Before UI elements)
    # ============================================================================
    # Set default indexes based on detected regime (not values yet)
    if current_regime == "Low Volatility":
        default_freq_index = 0  # Daily (more precision in stable markets)
        default_lookback_index = 1  # 5 years (stable regime, can use more history)
        default_model_index = 0  # XGBoost (faster, sufficient for low vol)
    elif current_regime == "Medium Volatility":
        default_freq_index = 1  # Weekly (balance precision/noise)
        default_lookback_index = 1  # 5 years (balanced)
        default_model_index = 0  # XGBoost
    else:  # High Volatility
        default_freq_index = 2  # Monthly (reduce noise)
        default_lookback_index = 2  # 3 years (recent regime-specific)
        default_model_index = 1  # LSTM (better for volatility)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 Training Data Selection")
    
    with st.sidebar.expander("ℹ️ How Much History? (Click for Guidance)", expanded=False):
        st.markdown("""
        **Trade-off: More Data vs. Regime Relevance**
        
        **Full History (20 years):**
        - ✅ More training examples
        - ✅ Captures multiple regimes
        - ❌ Includes outdated patterns (2008 crisis)
        - ❌ Slower training (950+ splits)
        - **Use when:** Testing long-term strategy
        
        **Last 5 Years (Recommended Default):**
        - ✅ Recent market structure
        - ✅ Captures 1-2 regime transitions
        - ✅ Reasonable training time (~250 splits)
        - **Use when:** Balancing data quantity + relevance
        
        **Last 3 Years:**
        - ✅ Focused on current regime
        - ✅ Fast training (~150 splits)
        - ❌ Less data for learning
        - **Use when:** Current regime detected is stable
        
        **Last 2 Years:**
        - ✅ Very regime-specific
        - ✅ Very fast (~100 splits)
        - ❌ Minimal training data
        - **Use when:** Quick validation only
        
        **Regime Detection Insight:**
        If current regime is "Low Volatility," training on 2008 
        crash data may hurt performance. Use shorter lookback.
        """)
    
    data_lookback = st.sidebar.selectbox(
        "Historical Data to Use",
        [
            "Full History (All Regimes)",
            "Last 5 Years (Recent Context)",
            "Last 3 Years (Current Regime Focus)",
            "Last 2 Years (Very Recent Only)",
        ],
        index=default_lookback_index,  # Regime-aware default
        help="How much historical data to use for training. Shorter = faster + more regime-specific."
    )
    
    # Map selection to years
    lookback_map = {
        "Full History (All Regimes)": None,  # Use all available
        "Last 5 Years (Recent Context)": 5,
        "Last 3 Years (Current Regime Focus)": 3,
        "Last 2 Years (Very Recent Only)": 2,
    }
    
    lookback_years = lookback_map[data_lookback]
    
    # Filter data by lookback
    if lookback_years:
        cutoff_date = date_filtered_df.index[-1] - pd.DateOffset(years=lookback_years)
        date_filtered_df = date_filtered_df.loc[cutoff_date:]
        st.sidebar.success(f"""
        ✅ **Using last {lookback_years} years**
        - From: {date_filtered_df.index[0].date()}
        - To: {date_filtered_df.index[-1].date()}
        - Total: {len(date_filtered_df)} days
        """)
    else:
        st.sidebar.info(f"""
        📅 **Using full history**
        - From: {date_filtered_df.index[0].date()}
        - Total: {len(date_filtered_df)} days
        """)
    
    # Data frequency selector (AFTER lookback selection)
    data_freq = st.sidebar.selectbox(
        "Data Frequency",
        ["Daily", "Weekly", "Monthly"],
        index=default_freq_index,  # Regime-aware default
        help="Frequency of data points for ML training"
    )

    # Resample data based on frequency
    original_len = len(date_filtered_df)
    if data_freq == "Weekly":
        date_filtered_df = date_filtered_df.resample(
            'W-FRI').last()  # Weekly, Friday close
        st.sidebar.success(f"""
        ✅ **Resampled to Weekly**
        - {original_len} days → {len(date_filtered_df)} weeks
        - Friday closing prices
        - ~5x faster training!
        """)
    elif data_freq == "Monthly":
        date_filtered_df = date_filtered_df.resample(
            'M').last()  # Monthly, end of month
        st.sidebar.success(f"""
        ✅ **Resampled to Monthly**
        - {original_len} days → {len(date_filtered_df)} months
        - End-of-month prices
        - ~21x faster training!
        """)
    elif data_freq == "Daily":
        st.sidebar.info(f"📅 **Daily frequency:** {original_len} days")

    # Calculate dynamic limits based on available data (AFTER resampling)
    available_periods = len(date_filtered_df)
    max_train_periods = int(available_periods * 0.8)  # 80% for training
    # Max 10% for single test period
    max_test_periods = int(available_periods * 0.1)
    max_sequence = min(252, int(available_periods * 0.2))  # Max 20% or 1 year

    freq_label = {"Daily": "days", "Weekly": "weeks",
                  "Monthly": "months"}[data_freq]
    
    # Smart defaults for fast training (30-60 seconds)
    # Ensure defaults respect minimum of 30
    if data_freq == "Daily":
        default_train = max(30, min(252, max_train_periods))  # 1 year or 30 min
        default_test = max(1, min(5, max_test_periods))      # 1 week
        default_seq = max(20, min(60, max_sequence))          # 2-3 months
        est_time = "~30-60 seconds"
    elif data_freq == "Weekly":
        default_train = max(30, min(104, max_train_periods))  # 2 years or 30 min
        default_test = max(1, min(4, max_test_periods))      # 1 month
        default_seq = max(20, min(26, max_sequence))          # 6 months
        est_time = "~15-30 seconds"
    else:  # Monthly
        default_train = max(30, min(60, max_train_periods))   # 5 years or 30 min
        default_test = max(1, min(3, max_test_periods))      # 3 months
        default_seq = max(20, min(12, max_sequence))          # 1 year
        est_time = "~10-20 seconds"
    
    # Final safety check: if max_train_periods < 30, user needs more data
    if max_train_periods < 30:
        st.sidebar.error(f"""
        ❌ **Insufficient data after resampling**
        
        Available: {available_periods} {freq_label}
        Need at least: 30 {freq_label} for training
        
        **Solutions:**
        1. Select longer date range at top
        2. Use higher data frequency (Daily has more points)
        3. Choose "Last 5 Years" or "Full History"
        """)
        st.stop()

    st.sidebar.info(f"""
    **Available Data:** {available_periods} {freq_label}
    
    **Dynamic Limits (80% rule):**
    - Max train_size: {max_train_periods} {freq_label}
    - Max test_size: {max_test_periods} {freq_label}
    - Max seq_len: {max_sequence} {freq_label}
    
    **⚡ Quick Start Defaults:** Pre-configured for {est_time} training time
    """)
    
    model_choice = st.sidebar.selectbox(
        "Model",
        ["XGBoost Only", "LSTM Only", "Compare Both"],
        index=default_model_index,  # Regime-aware default
        help="XGBoost Only is fastest (~15s), Compare Both takes 2x longer"
    )
    
    # ============================================================================
    # CORE PARAMETERS (Required for Walk-Forward Validation)
    # ============================================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Core Parameters (Required)")
    
    st.sidebar.caption("""
    **The 3 essential settings for walk-forward validation:**
    """)
    
    train_size = st.sidebar.number_input(
        "1️⃣ Training Window (train_size)",
        min_value=30,
        max_value=max_train_periods,
        value=default_train,
        step=21 if data_freq == "Daily" else 4,
        help=f"How many past {freq_label} to train on (expanding window)"
    )
    
    test_size = st.sidebar.number_input(
        "2️⃣ Test Window (test_size)",
        min_value=1,
        max_value=max_test_periods,
        value=default_test,
        step=1,
        help=f"How many {freq_label} to test on each iteration"
    )
    
    seq_len = None
    if model_choice in ["Compare Both", "LSTM Only"]:
        seq_len = st.sidebar.number_input(
            "3️⃣ Sequence Length (seq_len)",
            min_value=20,
            max_value=max_sequence,
            value=default_seq,
            step=10 if data_freq == "Daily" else 4,
            help=f"LSTM lookback: how many {freq_label} in each input sample"
        )
    
    st.sidebar.markdown("""
    **What each parameter means:**
    - `train_size`: Past data for training (e.g., 252 days = 1 year)
    - `test_size`: Future data for testing (e.g., 5 days = 1 week)
    - `seq_len`: LSTM's memory window (e.g., 60 days = what it "sees" at once)
    """)
    
    # Max splits control (critical for performance)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Performance Control")
    
    max_splits = st.sidebar.number_input(
        "Max Walk-Forward Splits",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="Limit number of training iterations. Lower = faster. 50 splits ≈ 2-3 minutes for LSTM."
    )
    
    est_splits = max((available_periods - train_size) // test_size, 1)
    actual_splits = min(est_splits, max_splits)
    
    if est_splits > max_splits:
        st.sidebar.warning(f"""
        ⚠️ **Split limit active:**
        - Possible splits: {est_splits}
        - Will use: {actual_splits} (max_splits limit)
        
        **Estimated time:**
        - XGBoost: ~{actual_splits * 0.5:.0f} seconds
        - LSTM: ~{actual_splits * 2:.0f} seconds
        """)
    else:
        st.sidebar.info(f"""
        ✅ **Estimated splits:** {actual_splits}
        
        **Estimated time:**
        - XGBoost: ~{actual_splits * 0.5:.0f} seconds
        - LSTM: ~{actual_splits * 2:.0f} seconds
        """)
    
    # ============================================================================
    # OPTIONAL PARAMETERS (Advanced Walk-Forward Design)
    # ============================================================================
    with st.sidebar.expander("⚙️ Advanced Walk-Forward (Optional)", expanded=False):
        st.markdown("**Optional settings - defaults work well for most use cases!**")
        st.markdown("---")
        
        step_size = st.number_input(
            "4️⃣ Step Size (step_size)",
            min_value=1,
            max_value=max_test_periods,
            value=test_size,  # Default: same as test_size
            step=1,
            help=f"How far to move window forward (default = test_size for no overlap)"
        )
        
        st.caption("""
        **What is step_size?**
        - `step_size = test_size` → No overlap (recommended)
        - `step_size < test_size` → Overlapping tests (more splits)
        - `step_size > test_size` → Gaps between tests (faster)
        
        **Current:** Using default (step_size = test_size)
        """)
        
        st.markdown("---")
        
        val_size = st.number_input(
            "5️⃣ Validation Window (val_size)",
            min_value=0,
            max_value=int(train_size * 0.3),
            value=0,
            step=10 if data_freq == "Daily" else 2,
            help=f"Slice of training for hyperparameter tuning (0 = no validation)"
        )
        
        st.caption("""
        **What is val_size?**
        - Used for hyperparameter tuning INSIDE each training split
        - `val_size = 0` → Use defaults, no tuning (current)
        - `val_size = 20% train_size` → Standard for tuning
        
        **When to use:**
        - If you want to tune hyperparameters at each split
        - If you have lots of data (can spare 20% for validation)
        
        **When to skip:**
        - Using default hyperparameters (current approach)
        - Limited data (use all for training)
        """)
    
    # Summary of walk-forward configuration
    st.sidebar.markdown("---")
    st.sidebar.success(f"""
    **✅ Walk-Forward Configuration:**
    
    **Core Settings:**
    - `train_size`: {train_size} {freq_label}
    - `test_size`: {test_size} {freq_label}
    - `seq_len`: {seq_len if seq_len else 'N/A (XGBoost only)'} {freq_label}
    
    **Advanced (Optional):**
    - `step_size`: {test_size} {freq_label} (default)
    - `val_size`: 0 (no validation split)
    
    **Estimated:** ~{max((available_periods - train_size) // test_size, 1)} walk-forward splits
    """)

    # Model-specific parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Model Parameters")

    # XGBoost parameters
    with st.sidebar.expander("🌳 XGBoost Settings", expanded=(model_choice in ["XGBoost Only", "Compare Both"])):
        xgb_n_estimators = st.number_input(
            "N Estimators",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of boosting rounds (default 100)"
        )

        xgb_max_depth = st.number_input(
            "Max Depth",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Maximum tree depth (3 = shallow to prevent overfitting)"
        )

        xgb_learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Learning rate (default 0.1)"
        )

    # LSTM parameters
    lstm_hidden_units = None
    lstm_dropout = None
    lstm_epochs = None
    
    # ============================================================================
    # MODEL HYPERPARAMETERS (Optional - Use Defaults or Customize)
    # ============================================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Model Hyperparameters (Optional)")
    
    with st.sidebar.expander("🌳 XGBoost Hyperparameters", expanded=False):
        st.caption("**Optional:** Customize or use defaults")
        
        xgb_n_estimators = st.number_input(
            "N Estimators",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of boosting rounds"
        )
        
        xgb_max_depth = st.number_input(
            "Max Depth",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Tree depth (3 = shallow, prevents overfitting)"
        )
        
        xgb_learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Shrinkage rate (lower = more conservative)"
        )
    
    if model_choice in ["Compare Both", "LSTM Only"]:
        with st.sidebar.expander("🧠 LSTM Hyperparameters", expanded=False):
            st.caption("**Optional:** Customize or use defaults")
            
            lstm_hidden_units = st.number_input(
                "Hidden Units",
                min_value=16,
                max_value=256,
                value=64,
                step=16,
                help="LSTM layer size (higher = more capacity)"
            )
            
            lstm_dropout = st.number_input(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Regularization (0.3 = 30% dropout)"
            )
            
            lstm_epochs = st.number_input(
                "Max Epochs",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Training epochs (early stopping active)"
            )
            
            # LSTM data requirement check (use official variable names)
            min_lstm_train = seq_len + 100 if seq_len else 160
            if train_size < min_lstm_train:
                st.warning(f"""
                ⚠️ **LSTM needs more data!**
                
                - seq_len: {seq_len} {freq_label}
                - Recommended train_size: ≥{min_lstm_train} {freq_label}
                - Current train_size: {train_size} {freq_label}
                
                Increase train_size for better LSTM performance.
                """)
            else:
                st.success(f"✅ train_size sufficient for LSTM")
    
    # Model comparison guidance
    if model_choice == "Compare Both":
        st.sidebar.info("""
        **Comparing Both Models:**
        
        Each model uses its own hyperparameters.
        XGBoost and LSTM trained independently.
        """)

    # Run button
    run_ml = st.sidebar.button("🚀 Run ML Prediction", type="primary")

    if run_ml:
        # Regime already detected above, skip duplicate detection
        st.markdown("---")
        st.markdown("### 🤖 Training Models")

        with st.spinner(f"Creating features for {commodity_name}..."):
            # Create features
            features_df, metadata = create_ml_features_with_transparency(
                price_series,
                symbol=symbol
            )

            st.success(
                f"✅ Features created: {metadata['final_rows']} rows, {metadata['total_features']} features")
        
        # ============================================================================
        # DATA SUFFICIENCY CHECK
        # ============================================================================
        # Check if we have enough data for walk-forward validation
        available_rows = len(features_df)
        
        # Calculate minimum required rows
        if model_choice in ["Compare Both", "LSTM Only"]:
            # LSTM needs: train_size + seq_len + test_size (seq_len for first window)
            min_required = train_size + seq_len + test_size
            model_req_msg = f"LSTM requires train_size ({train_size}) + seq_len ({seq_len}) + test_size ({test_size})"
        else:
            # XGBoost needs: train_size + test_size
            min_required = train_size + test_size
            model_req_msg = f"XGBoost requires train_size ({train_size}) + test_size ({test_size})"
        
        if available_rows < min_required:
            st.error(f"""
            ❌ **Insufficient data for walk-forward validation**
            
            **Data Available:**
            - After feature creation: **{available_rows} {freq_label}**
            
            **Data Required:**
            - {model_req_msg}
            - **Minimum needed: {min_required} {freq_label}**
            
            **Gap: Need {min_required - available_rows} more {freq_label}**
            
            **Solutions:**
            1. **Reduce train_size** (currently {train_size}) → try {max(30, train_size - 20)}
            2. **Reduce seq_len** (currently {seq_len if seq_len else 'N/A'}) → try {max(20, seq_len - 10) if seq_len else 'N/A'}
            3. **Use Daily frequency** instead of {data_freq} (more data points)
            4. **Extend date range** to include more history
            
            **Quick Fix:** Try train_size={max(30, available_rows - test_size - (seq_len if seq_len else 0) - 10)}
            """)
            st.stop()
        
        # Show data sufficiency status
        margin = available_rows - min_required
        margin_pct = (margin / min_required) * 100
        
        if margin_pct < 20:
            st.warning(f"""
            ⚠️ **Tight data margin:** Only {margin} extra {freq_label} ({margin_pct:.1f}% buffer)
            
            This allows only ~{margin // test_size} walk-forward splits.
            
            **Recommended:** Reduce parameters or increase date range for more robust validation.
            """)
        else:
            st.success(f"""
            ✅ **Sufficient data:** {available_rows} {freq_label} available, {min_required} {freq_label} required
            
            **Margin:** {margin} extra {freq_label} ({margin_pct:.0f}% buffer)
            
            **Estimated walk-forward splits:** ~{max((available_rows - train_size) // test_size, 1)}
            """)

        # Show transparency section
        with st.expander("📋 Data Preparation Transparency", expanded=True):
            st.markdown("### What We Did")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ✅ Completed:")
                for item in metadata['transparency']['data_prep_completed']:
                    st.markdown(f"- ✓ {item}")

            with col2:
                st.markdown("#### ❌ NOT Done:")
                for item in metadata['transparency']['data_prep_NOT_done']:
                    st.markdown(f"- ✗ {item}")

            st.markdown("#### ⚠️ To Be Decided:")
            for item in metadata['transparency']['to_be_decided']:
                st.markdown(f"- ? {item}")

            # Outlier analysis
            st.markdown("---")
            st.markdown("### 🔍 Outlier Analysis")

            outliers = metadata['outliers']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Returns", outliers['total_returns'])
            with col2:
                st.metric(
                    "Outliers (>3σ)", f"{outliers['outlier_count']} ({outliers['outlier_pct']:.2f}%)")
            with col3:
                st.metric("Min Return", f"{outliers['min_return']:.2f}%")
            with col4:
                st.metric("Max Return", f"{outliers['max_return']:.2f}%")

            st.info(f"""
            **Interpretation:** {outliers['interpretation']}
            
            **Action Taken:** {outliers['action_taken']}
            """)

            # Class distribution
            st.markdown("---")
            st.markdown("### ⚖️ Class Distribution (Up vs Down Days)")

            dist = metadata['class_distribution']

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Down Days (0)", f"{dist.get('class_0_count', 0)} ({dist.get('class_0_pct', 0):.1f}%)")
            with col2:
                st.metric(
                    "Up Days (1)", f"{dist.get('class_1_count', 0)} ({dist.get('class_1_pct', 0):.1f}%)")

            if dist['is_imbalanced']:
                st.warning(f"⚠️ {dist['recommendation']}")
            else:
                st.success(f"✅ {dist['recommendation']}")

        # Run model(s)
        st.markdown("---")

        if model_choice == "Compare Both":
            # Show progress placeholder
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            status_placeholder.info(f"🚀 **Starting training:** {actual_splits} splits total")
            
            with st.spinner("Training XGBoost and LSTM models..."):
                # Prepare model params
                xgb_params = {
                    'n_estimators': xgb_n_estimators,
                    'max_depth': xgb_max_depth,
                    'learning_rate': xgb_learning_rate,
                }

                lstm_params = {
                    'sequence_length': seq_len,
                    'hidden_units': lstm_hidden_units,
                    'dropout_rate': lstm_dropout,
                    'epochs': lstm_epochs,
                } if seq_len else {}

                results = compare_models(
                    features_df,
                    initial_train_days=train_size,
                    test_days=test_size,
                    max_splits=max_splits,
                    xgb_params=xgb_params,
                    lstm_params=lstm_params,
                    verbose=False,
                )
            
            status_placeholder.success(f"✅ **Training complete!** Processed {actual_splits} splits")

            # Display comparison
            st.markdown("### 📊 Model Comparison Results")

            xgb_metrics = results['xgboost']['overall_metrics']
            lstm_metrics = results['lstm']['overall_metrics']

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🌳 XGBoost")
                st.metric("Accuracy", f"{xgb_metrics['accuracy']:.2%}")
                st.metric("Precision", f"{xgb_metrics['precision']:.2%}")
                st.metric("Recall", f"{xgb_metrics['recall']:.2%}")
                st.metric("F1 Score", f"{xgb_metrics['f1_score']:.2%}")
                if xgb_metrics.get('roc_auc'):
                    st.metric("ROC AUC", f"{xgb_metrics['roc_auc']:.3f}")

            with col2:
                st.markdown("#### 🧠 LSTM")
                st.metric("Accuracy", f"{lstm_metrics['accuracy']:.2%}")
                st.metric("Precision", f"{lstm_metrics['precision']:.2%}")
                st.metric("Recall", f"{lstm_metrics['recall']:.2%}")
                st.metric("F1 Score", f"{lstm_metrics['f1_score']:.2%}")
                if lstm_metrics.get('roc_auc'):
                    st.metric("ROC AUC", f"{lstm_metrics['roc_auc']:.3f}")

            with col3:
                st.markdown("#### 🏆 Winner")
                winner_name = results['winner'].upper()
                margin = results['margin']

                if results['winner'] == 'tie':
                    st.info("🤝 **TIE**\n\nBoth models perform equally")
                else:
                    emoji = "🌳" if results['winner'] == 'xgboost' else "🧠"
                    st.success(
                        f"{emoji} **{winner_name}**\n\n+{margin:.2f}% advantage")

                # Baseline comparison
                st.markdown("---")
                st.caption("**Baseline (Random):** 50%")

                xgb_lift = (xgb_metrics['accuracy'] - 0.5) * 100
                lstm_lift = (lstm_metrics['accuracy'] - 0.5) * 100

                st.caption(f"**XGBoost Lift:** +{xgb_lift:.1f}%")
                st.caption(f"**LSTM Lift:** +{lstm_lift:.1f}%")

            # Confusion matrices
            st.markdown("---")
            st.markdown("### 📊 Confusion Matrices")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🌳 XGBoost")

                cm_xgb = np.array([
                    [xgb_metrics['true_negatives'], xgb_metrics['false_positives']],
                    [xgb_metrics['false_negatives'], xgb_metrics['true_positives']]
                ])

                fig_cm_xgb = go.Figure(data=go.Heatmap(
                    z=cm_xgb,
                    x=['Predicted Down', 'Predicted Up'],
                    y=['Actual Down', 'Actual Up'],
                    text=cm_xgb,
                    texttemplate="%{text}",
                    colorscale="Blues",
                ))

                fig_cm_xgb.update_layout(
                    title="XGBoost Confusion Matrix",
                    height=400,
                )

                st.plotly_chart(fig_cm_xgb, use_container_width=True)

            with col2:
                st.markdown("#### 🧠 LSTM")

                cm_lstm = np.array([
                    [lstm_metrics['true_negatives'],
                        lstm_metrics['false_positives']],
                    [lstm_metrics['false_negatives'],
                        lstm_metrics['true_positives']]
                ])

                fig_cm_lstm = go.Figure(data=go.Heatmap(
                    z=cm_lstm,
                    x=['Predicted Down', 'Predicted Up'],
                    y=['Actual Down', 'Actual Up'],
                    text=cm_lstm,
                    texttemplate="%{text}",
                    colorscale="Purples",
                ))

                fig_cm_lstm.update_layout(
                    title="LSTM Confusion Matrix",
                    height=400,
                )

                st.plotly_chart(fig_cm_lstm, use_container_width=True)

            # Prediction accuracy over time
            st.markdown("---")
            st.markdown("### 📈 Accuracy Over Walk-Forward Splits")

            xgb_split_df = pd.DataFrame(results['xgboost']['split_metrics'])
            lstm_split_df = pd.DataFrame(results['lstm']['split_metrics'])

            fig_acc = go.Figure()

            fig_acc.add_trace(go.Scatter(
                x=xgb_split_df['split'],
                y=xgb_split_df['accuracy'] * 100,
                mode='lines+markers',
                name='XGBoost',
                line=dict(color='blue', width=2),
                marker=dict(size=8),
            ))

            fig_acc.add_trace(go.Scatter(
                x=lstm_split_df['split'],
                y=lstm_split_df['accuracy'] * 100,
                mode='lines+markers',
                name='LSTM',
                line=dict(color='purple', width=2),
                marker=dict(size=8),
            ))

            fig_acc.add_hline(
                y=50,
                line_dash="dash",
                line_color="gray",
                annotation_text="Random (50%)"
            )

            fig_acc.update_layout(
                title="Prediction Accuracy by Split (Expanding Window)",
                xaxis_title="Split Number",
                yaxis_title="Accuracy (%)",
                height=500,
                template="plotly_white",
            )

            st.plotly_chart(fig_acc, use_container_width=True)

            # Feature importance (XGBoost only)
            if results['xgboost']['feature_importance'] is not None:
                st.markdown("---")
                st.markdown("### 📊 XGBoost Feature Importance")

                feat_imp = results['xgboost']['feature_importance']

                fig_imp = go.Figure(go.Bar(
                    x=feat_imp.values[:15],  # Top 15
                    y=feat_imp.index[:15],
                    orientation='h',
                    marker_color='lightblue',
                ))

                fig_imp.update_layout(
                    title="Top 15 Most Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=500,
                    template="plotly_white",
                )

                st.plotly_chart(fig_imp, use_container_width=True)

            # Model comparison table
            st.markdown("---")
            st.markdown("### 📋 Detailed Metrics Comparison")

            comparison_data = []
            for model_name, model_results in [('XGBoost', results['xgboost']), ('LSTM', results['lstm'])]:
                metrics = model_results['overall_metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.2%}",
                    'Precision': f"{metrics['precision']:.2%}",
                    'Recall': f"{metrics['recall']:.2%}",
                    'F1 Score': f"{metrics['f1_score']:.2%}",
                    'ROC AUC': f"{metrics.get('roc_auc', 0):.3f}" if metrics.get('roc_auc') else 'N/A',
                    'True Positives': metrics['true_positives'],
                    'True Negatives': metrics['true_negatives'],
                    'False Positives': metrics['false_positives'],
                    'False Negatives': metrics['false_negatives'],
                })

            st.dataframe(pd.DataFrame(comparison_data),
                         use_container_width=True, hide_index=True)

        elif model_choice in ["XGBoost Only", "LSTM Only"]:
            model_type = "xgboost" if model_choice == "XGBoost Only" else "lstm"
            model_emoji = "🌳" if model_type == "xgboost" else "🧠"
            
            # Show progress placeholder
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            status_placeholder.info(f"🚀 **Starting training:** {actual_splits} splits with {model_choice}")

            # Run model
            with st.spinner(f"Training {model_choice} model..."):
                # Prepare model params
                model_params = {}
                if model_type == "xgboost":
                    model_params = {
                        'n_estimators': xgb_n_estimators,
                        'max_depth': xgb_max_depth,
                        'learning_rate': xgb_learning_rate,
                    }
                elif model_type == "lstm" and seq_len:
                    model_params = {
                        'sequence_length': seq_len,
                        'hidden_units': lstm_hidden_units,
                        'dropout_rate': lstm_dropout,
                        'epochs': lstm_epochs,
                    }

                results = run_walk_forward_validation(
                    features_df,
                    model_type=model_type,
                    initial_train_days=train_size,
                    test_days=test_size,
                    max_splits=max_splits,
                    model_params=model_params,
                    verbose=False,
                )
            
            status_placeholder.success(f"✅ **Training complete!** Processed {actual_splits} splits")

            if 'error' in results:
                st.error(f"❌ {results['error']}")
                st.stop()

            # Show transparency section (collapsed by default, after training)
            with st.expander("📋 Data Preparation Transparency"):
                st.markdown("### What We Did")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### ✅ Completed:")
                    for item in metadata['transparency']['data_prep_completed']:
                        st.markdown(f"- ✓ {item}")

                with col2:
                    st.markdown("#### ❌ NOT Done:")
                    for item in metadata['transparency']['data_prep_NOT_done']:
                        st.markdown(f"- ✗ {item}")

                st.markdown("#### ⚠️ To Be Decided:")
                for item in metadata['transparency']['to_be_decided']:
                    st.markdown(f"- ? {item}")

                # Outlier analysis
                st.markdown("---")
                st.markdown("### 🔍 Outlier Analysis")

                outliers = metadata['outliers']

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Returns", outliers['total_returns'])
                with col2:
                    st.metric(
                        "Outliers (>3σ)", f"{outliers['outlier_count']} ({outliers['outlier_pct']:.2f}%)")
                with col3:
                    st.metric("Min Return", f"{outliers['min_return']:.2f}%")
                with col4:
                    st.metric("Max Return", f"{outliers['max_return']:.2f}%")

                st.info(f"""
                **Interpretation:** {outliers['interpretation']}
                
                **Action Taken:** {outliers['action_taken']}
                """)

                # Class distribution
                st.markdown("---")
                st.markdown("### ⚖️ Class Distribution (Up vs Down Days)")

                dist = metadata['class_distribution']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Down Days (0)", f"{dist.get('class_0_count', 0)} ({dist.get('class_0_pct', 0):.1f}%)")
                with col2:
                    st.metric(
                        "Up Days (1)", f"{dist.get('class_1_count', 0)} ({dist.get('class_1_pct', 0):.1f}%)")

                if dist['is_imbalanced']:
                    st.warning(f"⚠️ {dist['recommendation']}")
                else:
                    st.success(f"✅ {dist['recommendation']}")

            # Display results
            st.markdown(f"### {model_emoji} {model_choice} Results")

            metrics = results['overall_metrics']

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.2%}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.2%}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1_score']:.2%}")

            # Baseline comparison
            baseline = 0.5
            lift = (metrics['accuracy'] - baseline) * 100

            if lift > 2:
                st.success(
                    f"✅ Model beats random baseline by **{lift:.1f}%** (Good!)")
            elif lift > 0:
                st.info(
                    f"ℹ️ Model beats random baseline by **{lift:.1f}%** (Modest improvement)")
            else:
                st.warning(
                    f"⚠️ Model does NOT beat random baseline (accuracy < 50%)")

            # Confusion matrix
            st.markdown("---")
            st.markdown("### 📊 Confusion Matrix")

            cm = np.array([
                [metrics['true_negatives'], metrics['false_positives']],
                [metrics['false_negatives'], metrics['true_positives']]
            ])

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Down', 'Predicted Up'],
                y=['Actual Down', 'Actual Up'],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues" if model_type == "xgboost" else "Purples",
                colorbar=dict(title="Count"),
            ))

            fig_cm.update_layout(
                title=f"{model_choice} Confusion Matrix",
                height=500,
                template="plotly_white",
            )

            st.plotly_chart(fig_cm, use_container_width=True)

            # Accuracy by split
            st.markdown("---")
            st.markdown("### 📈 Accuracy by Walk-Forward Split")

            split_df = pd.DataFrame(results['split_metrics'])

            fig_acc = go.Figure()

            fig_acc.add_trace(go.Scatter(
                x=split_df['split'],
                y=split_df['accuracy'] * 100,
                mode='lines+markers',
                name=model_choice,
                line=dict(color='blue' if model_type ==
                          'xgboost' else 'purple', width=2),
                marker=dict(size=8),
            ))

            fig_acc.add_hline(y=50, line_dash="dash",
                              line_color="gray", annotation_text="Random")

            fig_acc.update_layout(
                title=f"{model_choice} - Accuracy Over Time",
                xaxis_title="Split Number",
                yaxis_title="Accuracy (%)",
                height=500,
                template="plotly_white",
            )

            st.plotly_chart(fig_acc, use_container_width=True)

            # Feature importance (XGBoost only)
            if model_type == "xgboost" and results['feature_importance'] is not None:
                st.markdown("---")
                st.markdown("### 📊 Feature Importance")

                feat_imp = results['feature_importance']

                fig_imp = go.Figure(go.Bar(
                    x=feat_imp.values[:15],
                    y=feat_imp.index[:15],
                    orientation='h',
                    marker_color='lightblue',
                    text=[f"{v:.3f}" for v in feat_imp.values[:15]],
                    textposition='outside',
                ))

                fig_imp.update_layout(
                    title="Top 15 Most Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=600,
                    template="plotly_white",
                )

                st.plotly_chart(fig_imp, use_container_width=True)

        # Interpretation guide
        with st.expander("ℹ️ Understanding ML Prediction Results"):
            st.markdown("""
            ### How to Interpret These Results
            
            **Accuracy:**
            - 50% = Random guess (baseline)
            - 52-55% = Weak signal (marginally profitable)
            - 55-60% = Decent signal (potentially profitable)
            - 60%+ = Strong signal (excellent if consistent)
            
            **Precision:**
            - When model predicts UP, how often is it correct?
            - High precision = fewer false alarms
            
            **Recall:**
            - Of all actual UP days, how many did model catch?
            - High recall = doesn't miss opportunities
            
            **F1 Score:**
            - Harmonic mean of precision and recall
            - Balanced metric
            
            **ROC AUC:**
            - 0.5 = Random
            - 0.6-0.7 = Weak
            - 0.7-0.8 = Good
            - 0.8+ = Excellent
            
            ### Walk-Forward Validation
            
            **Why expanding window?**
            - Uses all available historical data
            - More realistic (in production, you'd use all past data)
            - Training set grows over time
            
            **Why 1-week test periods?**
            - Practical for trading decisions
            - Enough to evaluate short-term accuracy
            - Multiple splits for robustness
            
            ### XGBoost vs LSTM
            
            **XGBoost tends to win when:**
            - Limited data (<1000 samples)
            - Features are well-engineered
            - Relationships are non-linear but not deeply sequential
            
            **LSTM tends to win when:**
            - Long sequences (1000+ samples)
            - Strong sequential dependencies
            - Complex temporal patterns
            
            For commodities, XGBoost usually performs as well or better than LSTM.
            
            ### Feature Importance (XGBoost)
            
            Tells you which features matter most:
            - High importance = model relies on this heavily
            - Low importance = could potentially remove
            
            Common patterns:
            - Recent returns (1d, 5d) usually important
            - Volatility regime indicators
            - Mean reversion signals (distance from MA)
            """)

            st.markdown("---")
            st.markdown("### 🔍 Data Preparation Transparency")

            st.markdown("""
            #### ✅ What We DID:
            
            1. **Forward filled missing data**
               - Commodities trade continuously
               - Gaps are typically weekends/holidays
            
            2. **Used LOG returns (not arithmetic)**
               - Time-additive: `log(P_t / P_{t-1})`
               - Better for ML and time-series
               - More symmetric distribution
            
            3. **Mixed expanding and rolling windows**
               - Expanding: Long-term baseline (e.g., `downside_dev_expanding`)
               - Rolling: Recent regime (e.g., `vol_21d`, `downside_dev_21d`)
               - Best of both worlds
            
            4. **All features LAGGED**
               - No look-ahead bias
               - Every feature uses only past data
               - Example: `log_return_1d = yesterday's return`
            
            5. **Dropped rows with NaN in features**
               - Ensures complete feature matrix
               - Typically first ~200 rows (for 200-day MA)
            
            6. **StandardScaler for LSTM only**
               - Neural networks need scaling
               - Fit on training data only (no leakage)
               - XGBoost doesn't need scaling (tree-based)
            
            7. **Auto-balanced class weights**
               - Detects if >65% one class
               - Applies `class_weight='balanced'` automatically
            """)

            st.markdown("""
            #### ❌ What We DID NOT Do:
            
            1. **NO outlier removal/capping**
               - All data kept (including extreme returns)
               - We REPORT outliers but don't remove them
               - You decide: Keep / Cap / Winsorize
            
            2. **NO PCA or dimensionality reduction**
               - All features are interpretable
               - ~15 features is manageable
               - Can add if needed
            
            3. **NO Box-Cox transforms**
               - Log returns already approximately normal
               - Additional transforms add complexity
            
            4. **NO synthetic data (SMOTE)**
               - Class imbalance handled by weights
               - SMOTE can introduce artifacts in time series
            
            5. **NO hyperparameter tuning**
               - Using sensible defaults
               - XGBoost: `max_depth=3`, `n_estimators=100`
               - LSTM: `hidden_units=64`, `dropout=0.3`
            
            6. **NO ensemble methods**
               - Not combining multiple models (yet)
               - Could stack XGBoost + LSTM predictions
            """)

            st.markdown("""
            #### ⚠️ To Be DECIDED (By You):
            
            1. **Outlier Treatment**
               - Check outlier report above
               - Options: Keep all / Cap at 3σ / Winsorize 1%/99%
               - Current: Keeping all (transparent)
            
            2. **Hyperparameter Tuning**
               - Current: Using defaults
               - Consider if accuracy < 52%
               - Risk: Overfitting to specific period
            
            3. **Additional Features**
               - Ratio features (Gold/Silver, Copper/Gold)
               - Cross-asset features (SPY, VIX, DXY)
               - More technical indicators (MACD, Bollinger Bands)
            
            4. **Model Selection**
               - Current: XGBoost + LSTM
               - Could try: LightGBM, CatBoost, Transformers
               - Could ensemble: Combine predictions
            
            5. **Class Imbalance Strategy**
               - Current: Auto class_weight='balanced'
               - Alternative: Adjust threshold (0.4 or 0.6 instead of 0.5)
               - Alternative: Oversample minority class
            """)

            st.markdown("---")
            st.markdown("### ⚠️ Important Disclaimers")

            st.warning("""
            **This is Educational/Research Code:**
            
            - ❌ Not production-ready for live trading
            - ❌ No transaction costs included
            - ❌ No slippage modeling
            - ❌ No position sizing or risk management
            
            **Before Using for Trading:**
            
            1. Paper trade first (at least 3 months)
            2. Add transaction costs (0.1-0.5% per trade)
            3. Implement stop losses and position sizing
            4. Monitor performance out-of-sample
            5. Understand that backtest ≠ live performance
            
            **Even 55% accuracy can be profitable with:**
            - Proper risk management
            - Position sizing (Kelly criterion)
            - Stop losses
            - Transaction cost awareness
            """)

            st.markdown("---")
            st.markdown("### 📚 Learn More")

            st.info("""
            **Documentation:**
            - `docs/ML_PRICE_PREDICTION.md` - Complete technical details
            - `docs/ML_TRANSPARENCY_REPORT.md` - All data prep decisions
            - `docs/ML_QUICK_START.md` - Installation and usage guide
            
            **Code:**
            - `src/data/ml_features.py` - Feature engineering
            - `src/models/commodity_direction.py` - Models and validation
            """)

    else:
        st.info("""
        👋 **Configure and Run ML Prediction**
        
        **Steps:**
        1. Select exactly ONE commodity from sidebar
        2. Choose date range (recommend 2+ years)
        3. Configure ML settings in sidebar
        4. Click **Run ML Prediction**
        
        **What you'll get:**
        - ✅ XGBoost vs LSTM comparison
        - ✅ Accuracy, precision, recall, F1
        - ✅ Confusion matrices
        - ✅ Feature importance (XGBoost)
        - ✅ Walk-forward validation results
        - ✅ Full transparency on data prep
        """)

        # Frequency selection guide
        with st.expander("💡 Data Frequency Guide - Click to Expand", expanded=False):
            st.markdown("""
            ### 📊 Choose Your Data Frequency
            
            Different frequencies serve different trading strategies:
            
            #### 📅 Daily (High Resolution)
            **Best for:** Day trading, short-term patterns, news impact
            
            **Example - Gold (20 years):**
            - 5,040 days available
            - Max training: ~4,000 days (~16 years)
            - Training time: ~3 hours for full walk-forward
            - Use case: Predict tomorrow's direction
            
            **Recommended settings:**
            - XGBoost: 63-252 days training, 5 day test
            - LSTM: 252+ days training, 60 day sequence
            
            ---
            
            #### 📈 Weekly (~5x Faster)
            **Best for:** Swing trading, medium-term patterns, weekly momentum
            
            **Example - Silver (20 years):**
            - 4,967 days → ~993 weeks
            - Max training: ~794 weeks (~15 years)
            - Training time: ~40 minutes (5x faster!)
            - Use case: Predict next week's direction
            
            **Recommended settings:**
            - XGBoost: 52-104 weeks training (1-2 years), 1-4 week test
            - LSTM: 104-208 weeks training (2-4 years), 52 week sequence (1 year lookback)
            
            **Why Friday close?** Market standard for weekly bars
            
            ---
            
            #### 📆 Monthly (~21x Faster)
            **Best for:** Long-term trends, macro analysis, position trading
            
            **Example - Copper (20 years):**
            - 5,040 days → ~240 months
            - Max training: ~192 months (~16 years!)
            - Training time: ~10 minutes (21x faster!)
            - Use case: Predict next month's direction
            
            **Recommended settings:**
            - XGBoost: 24-60 months training (2-5 years), 1-3 month test
            - LSTM: 60-120 months training (5-10 years), 12-24 month sequence
            
            **Perfect for:** Correlations with macro indicators (GDP, rates, inflation)
            
            ---
            
            ### 🎯 Smart Strategy Combinations
            
            #### Multi-Timeframe Analysis
            1. **Daily XGBoost**: Short-term momentum (63 days)
            2. **Weekly LSTM**: Medium-term trend (104 weeks)
            3. **Monthly regime**: Long-term macro signal (60 months)
            4. **Ensemble**: Combine predictions across timeframes!
            
            #### Trading Style Matching
            
            | Trading Style | Frequency | Training | Sequence | Test Period |
            |---------------|-----------|----------|----------|-------------|
            | Day Trading | Daily | 63-252 days | 60 days | 5 days |
            | Swing Trading | Weekly | 104-208 weeks | 52 weeks | 1-4 weeks |
            | Position Trading | Monthly | 60-120 months | 12-24 months | 1-3 months |
            | Long-term Investing | Monthly | 120+ months | 24-36 months | 3-6 months |
            
            #### Regime Detection Strategy
            **We have full regime detection in "Rolling Metrics" analysis!**
            
            Use **Rolling Volatility** to identify market regimes:
            - 📊 **Rolling Sharpe Ratio**: Performance over time (trending vs mean-reverting)
            - 📈 **Rolling Volatility**: Regime identification (calm vs turbulent)
            - 📉 **Rolling Sortino**: Downside risk regimes (asymmetric risk periods)
            - 🔗 **Rolling Correlation**: Relationship changes (diversification breakdown)
            
            **Regime-Based Trading Strategy:**
            1. **Low Volatility Regime** (Calm markets)
               - Use Daily/Weekly data for precision
               - Momentum strategies work well
               - Higher leverage acceptable
               
            2. **High Volatility Regime** (Crisis/Turbulence)
               - Use Weekly/Monthly to filter noise
               - Mean reversion strategies
               - Reduce leverage, focus on risk management
               
            3. **Regime Transitions** (Volatility spikes)
               - Be cautious during transitions
               - Wait for regime confirmation
               - Consider ensemble predictions
            
            **How to use:**
            - Go to "Rolling Metrics" analysis
            - Check rolling volatility chart
            - Identify current regime
            - Adjust ML timeframe accordingly!
            
            ---
            
            ### ⚡ Training Time Comparison
            
            **For 20 years of Gold data:**
            
            | Frequency | Data Points | XGBoost Time | LSTM Time | Total (Compare) |
            |-----------|-------------|--------------|-----------|-----------------|
            | Daily | 5,040 | ~90 min | ~180 min | ~270 min (4.5 hrs) |
            | Weekly | ~1,008 | ~18 min | ~36 min | ~54 min |
            | Monthly | ~240 | ~4 min | ~8 min | ~12 min |
            
            **💡 Tip:** Start with **Weekly** for best balance of detail and speed!
            
            ---
            
            ### 🔬 When to Use Each Model
            
            #### XGBoost (Tree-Based)
            **Strengths:**
            - Works with limited data
            - Fast training
            - Feature importance
            - Non-linear patterns
            
            **Best with:**
            - Daily/Weekly data
            - 100-1000 samples
            - Tabular features
            
            #### LSTM (Neural Network)
            **Strengths:**
            - Learns sequences
            - Long-term dependencies
            - Regime persistence
            
            **Best with:**
            - Weekly/Monthly data
            - 500+ samples
            - Temporal patterns
            
            **Needs more data:** If training < sequence + 100, expect warning!
            
            ---
            
            ### 📝 Example Configurations
            
            #### Beginner: Quick Test
            ```
            Frequency: Weekly
            Training: 104 weeks (2 years)
            Test: 4 weeks (1 month)
            Model: XGBoost Only
            Time: ~5 minutes
            ```
            
            #### Intermediate: Balanced
            ```
            Frequency: Weekly
            Training: 208 weeks (4 years)
            Test: 1 week
            Model: Compare Both
            LSTM Sequence: 52 weeks
            Time: ~30 minutes
            ```
            
            #### Advanced: Maximum History
            ```
            Frequency: Monthly
            Training: 180 months (15 years)
            Test: 1 month
            Model: Compare Both
            LSTM Sequence: 24 months (2 years)
            Time: ~10 minutes
            ```
            
            #### Expert: Multi-Timeframe
            - Run Daily XGBoost (short-term)
            - Run Weekly LSTM (medium-term)
            - Run Monthly LSTM (long-term)
            - Compare & ensemble results!
            """)

        st.markdown("---")
        st.caption(
            "**Note:** First run may take 2-3 minutes for daily data. Weekly/Monthly are much faster!")


elif analysis_type == "Correlation Matrix":
    st.subheader("🔗 Correlation Matrix")

    # Calculate correlation matrix
    corr_matrix = date_filtered_df[selected_commodities].corr()

    # Replace symbols with display names
    display_names = [COMMODITIES_CONFIG.get(s, {}).get(
        "name", s) for s in corr_matrix.index]
    corr_matrix.index = display_names
    corr_matrix.columns = display_names

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Commodity Price Correlations",
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Normalized Comparison":
    st.subheader("📈 Normalized Price Comparison (Base = 100)")

    # Normalize prices to start at 100
    normalized_df = (date_filtered_df / date_filtered_df.iloc[0]) * 100

    fig = go.Figure()

    for symbol in selected_commodities:
        if symbol in normalized_df.columns:
            series = normalized_df[symbol].dropna()
            config = COMMODITIES_CONFIG.get(symbol, {})

            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=config.get("name", symbol),
                    mode="lines",
                    line=dict(width=2),
                )
            )

    fig.update_layout(
        title=f"Normalized Commodity Performance (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Indexed Price",
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Performance table
    st.markdown("### 📊 Cumulative Performance")

    perf_data = []
    for symbol in selected_commodities:
        if symbol in normalized_df.columns:
            series = normalized_df[symbol].dropna()
            if len(series) > 0:
                config = COMMODITIES_CONFIG.get(symbol, {})
                total_return = series.iloc[-1] - 100
                perf_data.append({
                    "Asset": config.get("name", symbol),
                    "Start": f"100.00",
                    "End": f"{series.iloc[-1]:.2f}",
                    "Total Return": f"{total_return:+.2f}%",
                })

    if perf_data:
        st.dataframe(pd.DataFrame(perf_data),
                     use_container_width=True, hide_index=True)

elif analysis_type == "Seasonality Analysis":
    st.subheader("🌙 Seasonality Analysis")

    st.markdown("""
    Analyze monthly patterns in commodity returns to identify seasonal trends.
    """)

    # Use the full date range for seasonality (not filtered)
    seasonality_df = filtered_df.copy()

    for symbol in selected_commodities:
        if symbol in seasonality_df.columns:
            series = seasonality_df[symbol].dropna()

            if len(series) < 24:  # Need at least 2 years
                st.warning(
                    f"Not enough data for {symbol} seasonality analysis")
                continue

            config = COMMODITIES_CONFIG.get(symbol, {})
            st.markdown(f"### {config.get('name', symbol)}")

            # Calculate monthly returns
            returns = series.pct_change()
            monthly_data = pd.DataFrame({
                "return": returns,
                "month": returns.index.month,
                "year": returns.index.year,
            })

            # 1. Average returns by month
            col1, col2 = st.columns(2)

            with col1:
                avg_by_month = monthly_data.groupby(
                    "month")["return"].mean() * 100
                month_names = [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                ]

                fig_bar = go.Figure(
                    go.Bar(
                        x=month_names,
                        y=avg_by_month.values,
                        marker_color=["green" if v >
                                      0 else "red" for v in avg_by_month.values],
                    )
                )
                fig_bar.update_layout(
                    title="Average Return by Month",
                    xaxis_title="Month",
                    yaxis_title="Avg Return (%)",
                    height=400,
                    template="plotly_white",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                # 2. Returns heatmap (year x month)
                pivot = monthly_data.pivot_table(
                    values="return", index="year", columns="month", aggfunc="mean"
                ) * 100

                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=month_names,
                        y=pivot.index,
                        colorscale="RdYlGn",
                        zmid=0,
                        text=pivot.values,
                        texttemplate="%{text:.1f}",
                        textfont={"size": 8},
                        colorbar=dict(title="Return (%)"),
                    )
                )
                fig_heat.update_layout(
                    title="Returns Heatmap (Year × Month)",
                    xaxis_title="Month",
                    yaxis_title="Year",
                    height=400,
                    template="plotly_white",
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            # 3. Box plot of returns by month
            fig_box = go.Figure()
            for month in range(1, 13):
                month_returns = monthly_data[monthly_data["month"]
                                             == month]["return"] * 100
                fig_box.add_trace(
                    go.Box(
                        y=month_returns,
                        name=month_names[month - 1],
                        boxmean="sd",
                    )
                )

            fig_box.update_layout(
                title="Return Distribution by Month",
                yaxis_title="Return (%)",
                height=400,
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # 4. Statistics table
            stats_by_month = monthly_data.groupby("month")["return"].agg(
                ["mean", "std", "min", "max", "count"]
            ) * 100
            stats_by_month["mean"] = stats_by_month["mean"].round(2)
            stats_by_month["std"] = stats_by_month["std"].round(2)
            stats_by_month["min"] = stats_by_month["min"].round(2)
            stats_by_month["max"] = stats_by_month["max"].round(2)
            stats_by_month.index = month_names
            stats_by_month.columns = [
                "Mean (%)", "Std (%)", "Min (%)", "Max (%)", "Count"]

            st.dataframe(stats_by_month, use_container_width=True)

            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 <b>Tip:</b> Run <code>python scripts/update_commodities.py</code> to fetch the latest prices</p>
</div>
""", unsafe_allow_html=True)
