#!/usr/bin/env python3
"""
Metals Analytics Page

Interactive analysis of precious metals and commodity prices.
Now uses persistent parquet storage for fast loading.
"""

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

from src.data.commodities import COMMODITIES_CONFIG

# Page configuration
st.set_page_config(
    page_title="Metals Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Commodities & Metals Analytics")
st.markdown("**Persistent Data Storage** - Daily updated from Alpha Vantage & Yahoo Finance")

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
@st.cache_data
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

# Commodity selection
available_commodities = [symbol_to_display.get(col, col) for col in df.columns]
default_selection = [
    symbol_to_display.get("GLD", "Gold (GLD ETF)"),
    symbol_to_display.get("SLV", "Silver (SLV ETF)"),
    symbol_to_display.get("COPPER", "Copper"),
]
# Filter defaults to only those that exist
default_selection = [d for d in default_selection if d in available_commodities]

selected_display = st.sidebar.multiselect(
    "Select Assets",
    available_commodities,
    default=default_selection,
    help="Choose which assets to analyze",
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

# Analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    [
        "Price Trends",
        "Returns Analysis",
        "Correlation Matrix",
        "Normalized Comparison",
        "Seasonality Analysis",
    ],
    help="Type of analysis to perform",
)

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
                pct_change = ((series.iloc[-1] / series.iloc[-2]) - 1) * 100 if len(series) > 1 else 0
                config = COMMODITIES_CONFIG.get(symbol, {})
                
                st.metric(
                    config.get("name", symbol),
                    f"${latest:.2f}",
                    f"{pct_change:+.2f}%",
                    help=f"{config.get('unit', 'USD')} | Last: {series.index[-1].date()}",
                )

st.markdown("---")

# Date range filter for charts
st.markdown("**📅 Chart Date Range** *(adjust Y-axis to zoom into specific periods)*")

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
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

elif analysis_type == "Returns Analysis":
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

elif analysis_type == "Correlation Matrix":
    st.subheader("🔗 Correlation Matrix")
    
    # Calculate correlation matrix
    corr_matrix = date_filtered_df[selected_commodities].corr()
    
    # Replace symbols with display names
    display_names = [COMMODITIES_CONFIG.get(s, {}).get("name", s) for s in corr_matrix.index]
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
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

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
                st.warning(f"Not enough data for {symbol} seasonality analysis")
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
                avg_by_month = monthly_data.groupby("month")["return"].mean() * 100
                month_names = [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                ]
                
                fig_bar = go.Figure(
                    go.Bar(
                        x=month_names,
                        y=avg_by_month.values,
                        marker_color=["green" if v > 0 else "red" for v in avg_by_month.values],
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
                month_returns = monthly_data[monthly_data["month"] == month]["return"] * 100
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
            stats_by_month.columns = ["Mean (%)", "Std (%)", "Min (%)", "Max (%)", "Count"]
            
            st.dataframe(stats_by_month, use_container_width=True)
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 <b>Tip:</b> Run <code>python scripts/update_commodities.py</code> to fetch the latest prices</p>
</div>
""", unsafe_allow_html=True)
