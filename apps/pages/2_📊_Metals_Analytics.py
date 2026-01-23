#!/usr/bin/env python3
"""
Metals Analytics Page

Interactive analysis of precious metals and commodity prices from FRED.
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import ALPHAVANTAGE_API_KEY

# Page configuration
st.set_page_config(
    page_title="Metals Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Commodities & Metals Analytics")
st.markdown("**Multi-Source Data** - Alpha Vantage + Yahoo Finance ETFs")

st.info("""
📋 **Available Assets:**
- **Precious Metals**: Gold, Silver, Platinum, Palladium (via Yahoo Finance ETFs)
- **Energy**: Crude Oil (WTI & Brent), Natural Gas
- **Industrial Metals**: Copper, Aluminum
- **Agricultural**: Wheat, Corn, Coffee, Cotton, Sugar
""")

st.markdown("---")

# Check API key (only warning, not blocking - Yahoo Finance works without it)
if not ALPHAVANTAGE_API_KEY:
    st.warning("""
    ⚠️ **Alpha Vantage API Key not found!**
    
    You can still use **Precious Metals ETFs** (Gold, Silver, Platinum, Palladium) which use Yahoo Finance.
    
    To enable energy and agricultural commodities, add your Alpha Vantage API key to `.env`:
    ```
    ALPHAVANTAGE_API_KEY=your_key_here
    ```
    
    Get a free key at: https://www.alphavantage.co/support/#api-key
    """)

# Commodities metadata - supports both Alpha Vantage and Yahoo Finance
METALS_DATA = {
    # Precious Metals (Yahoo Finance ETFs)
    "Gold (GLD)": {
        "symbol": "GLD",
        "name": "SPDR Gold Trust ETF",
        "color": "#FFD700",
        "unit": "USD",
        "source": "yahoo",
    },
    "Silver (SLV)": {
        "symbol": "SLV",
        "name": "iShares Silver Trust ETF",
        "color": "#C0C0C0",
        "unit": "USD",
        "source": "yahoo",
    },
    "Platinum (PPLT)": {
        "symbol": "PPLT",
        "name": "Aberdeen Physical Platinum ETF",
        "color": "#E5E4E2",
        "unit": "USD",
        "source": "yahoo",
    },
    "Palladium (PALL)": {
        "symbol": "PALL",
        "name": "Aberdeen Physical Palladium ETF",
        "color": "#CED0DD",
        "unit": "USD",
        "source": "yahoo",
    },
    # Energy (Alpha Vantage)
    "Crude Oil (WTI)": {
        "symbol": "WTI",
        "name": "WTI Crude Oil",
        "color": "#000000",
        "unit": "USD/barrel",
        "source": "alphavantage",
    },
    "Crude Oil (Brent)": {
        "symbol": "BRENT",
        "name": "Brent Crude Oil",
        "color": "#2C2C2C",
        "unit": "USD/barrel",
        "source": "alphavantage",
    },
    "Natural Gas": {
        "symbol": "NATURAL_GAS",
        "name": "Natural Gas",
        "color": "#87CEEB",
        "unit": "USD/MMBtu",
        "source": "alphavantage",
    },
    # Industrial Metals (Alpha Vantage)
    "Copper": {
        "symbol": "COPPER",
        "name": "Copper Spot Price",
        "color": "#B87333",
        "unit": "USD/lb",
        "source": "alphavantage",
    },
    "Aluminum": {
        "symbol": "ALUMINUM",
        "name": "Aluminum Spot Price",
        "color": "#848789",
        "unit": "USD/lb",
        "source": "alphavantage",
    },
    # Agricultural (Alpha Vantage)
    "Wheat": {
        "symbol": "WHEAT",
        "name": "Wheat Price",
        "color": "#DAA520",
        "unit": "USD/bushel",
        "source": "alphavantage",
    },
    "Corn": {
        "symbol": "CORN",
        "name": "Corn Price",
        "color": "#F4C430",
        "unit": "USD/bushel",
        "source": "alphavantage",
    },
    "Coffee": {
        "symbol": "COFFEE",
        "name": "Coffee Price",
        "color": "#6F4E37",
        "unit": "USD/lb",
        "source": "alphavantage",
    },
    "Cotton": {
        "symbol": "COTTON",
        "name": "Cotton Price",
        "color": "#FFFFF0",
        "unit": "USD/lb",
        "source": "alphavantage",
    },
    "Sugar": {
        "symbol": "SUGAR",
        "name": "Sugar Price",
        "color": "#FFFFFF",
        "unit": "USD/lb",
        "source": "alphavantage",
    },
}

# Sidebar controls
st.sidebar.header("⚙️ Analysis Settings")

# Commodity selection
selected_commodities = st.sidebar.multiselect(
    "Select Assets",
    list(METALS_DATA.keys()),
    default=["Gold (GLD)", "Silver (SLV)", "Copper"],
    help="Choose which assets to analyze",
)

# Data interval
data_interval = st.sidebar.selectbox(
    "Data Interval",
    ["Monthly", "Weekly", "Daily"],
    index=0,
    help="Data granularity - monthly recommended for long histories",
)

interval_map = {"Daily": "daily", "Weekly": "weekly", "Monthly": "monthly"}

# Analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Price Trends", "Returns Analysis", "Correlation Matrix", "Normalized Comparison"],
    help="Type of analysis to perform",
)

# Fetch button
if st.sidebar.button("🔄 Fetch Data", type="primary"):
    st.session_state.fetch_data = True

# Fetch and cache data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_commodity_data(commodities_list, interval="monthly"):
    """Fetch commodities data from multiple sources (Alpha Vantage + Yahoo Finance)."""
    data = {}
    metadata = {}
    
    for commodity in commodities_list:
        try:
            symbol = METALS_DATA[commodity]["symbol"]
            source = METALS_DATA[commodity]["source"]
            
            if source == "yahoo":
                # Fetch from Yahoo Finance (for precious metals ETFs)
                ticker = yf.Ticker(symbol)
                
                # Determine period based on interval
                if interval == "daily":
                    period = "10y"
                    hist_interval = "1d"
                elif interval == "weekly":
                    period = "10y"
                    hist_interval = "1wk"
                else:  # monthly
                    period = "max"
                    hist_interval = "1mo"
                
                hist = ticker.history(period=period, interval=hist_interval)
                
                if not hist.empty:
                    # Use closing prices
                    series = hist["Close"]
                    series.index.name = "date"
                    
                    data[commodity] = series
                    metadata[commodity] = {
                        "count": len(series),
                        "start": series.index[0],
                        "end": series.index[-1],
                        "latest": series.iloc[-1],
                        "unit": METALS_DATA[commodity]["unit"],
                        "source": "Yahoo Finance",
                    }
                else:
                    st.sidebar.warning(f"⚠️ {commodity}: No data from Yahoo Finance")
                    
            elif source == "alphavantage":
                # Fetch from Alpha Vantage (for commodities)
                if not ALPHAVANTAGE_API_KEY:
                    st.sidebar.warning(f"⚠️ {commodity}: Alpha Vantage API key required")
                    continue
                
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": symbol,
                    "interval": interval,
                    "apikey": ALPHAVANTAGE_API_KEY,
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    json_data = response.json()
                    
                    # Check for errors
                    if "Error Message" in json_data:
                        st.sidebar.warning(f"⚠️ {commodity}: {json_data['Error Message']}")
                        continue
                    
                    if "Note" in json_data:
                        st.sidebar.warning(f"⚠️ Rate limit: {json_data['Note']}")
                        continue
                    
                    # Extract data
                    if "data" in json_data:
                        df_data = pd.DataFrame(json_data["data"])
                        
                        if not df_data.empty:
                            df_data["date"] = pd.to_datetime(df_data["date"])
                            df_data = df_data.set_index("date")
                            df_data["value"] = pd.to_numeric(df_data["value"], errors="coerce")
                            series = df_data["value"].sort_index()
                            
                            data[commodity] = series
                            metadata[commodity] = {
                                "count": len(series),
                                "start": series.index[0],
                                "end": series.index[-1],
                                "latest": series.iloc[-1],
                                "unit": METALS_DATA[commodity]["unit"],
                                "source": "Alpha Vantage",
                            }
                else:
                    st.sidebar.warning(f"⚠️ {commodity}: HTTP {response.status_code}")
                
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not fetch {commodity}: {str(e)}")
    
    return data, metadata

# Main content
if not selected_commodities:
    st.info("👈 Please select at least one commodity from the sidebar.")
    st.stop()

if "fetch_data" not in st.session_state:
    st.info("👈 Click **Fetch Data** in the sidebar to load commodity prices.")
    st.stop()

# Fetch data
with st.spinner(f"Fetching data for {len(selected_commodities)} commodities..."):
    commodities_data, metadata = fetch_commodity_data(
        selected_commodities, interval=interval_map[data_interval]
    )

if not commodities_data:
    st.error("❌ No data available. Check your Alpha Vantage API key and try again.")
    st.stop()

# Display metadata
st.success(f"✅ Loaded data for {len(commodities_data)} commodities")

cols = st.columns(min(4, len(commodities_data)))
for i, (commodity, meta) in enumerate(metadata.items()):
    col = cols[i % len(cols)]
    with col:
        st.metric(
            commodity,
            f"${meta['latest']:.2f}",
            help=f"Latest: {meta['end'].strftime('%Y-%m-%d')} ({meta['unit']})",
        )

st.markdown("---")

# Analysis sections
if analysis_type == "Price Trends":
    st.subheader("📈 Price Trends Over Time")
    
    fig = go.Figure()
    
    for commodity in selected_commodities:
        if commodity in commodities_data:
            series = commodities_data[commodity]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=commodity,
                    mode="lines",
                    line=dict(color=METALS_DATA[commodity]["color"], width=2),
                    hovertemplate=f"{commodity}: $%{{y:.2f}}<extra></extra>",
                )
            )
    
    fig.update_layout(
        title="Commodity Prices (USD)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=600,
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="lightgray",
                activecolor="gray"
            ),
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    st.subheader("📊 Price Statistics")
    
    stats_data = []
    for commodity in selected_commodities:
        if commodity in commodities_data:
            series = commodities_data[commodity]
            unit = metadata[commodity]["unit"]
            stats_data.append({
                "Commodity": commodity,
                "Current": f"${series.iloc[-1]:.2f}",
                "Min": f"${series.min():.2f}",
                "Max": f"${series.max():.2f}",
                "Mean": f"${series.mean():.2f}",
                "Std Dev": f"${series.std():.2f}",
                "Unit": unit,
            })
    
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

elif analysis_type == "Returns Analysis":
    st.subheader("📊 Returns Analysis")
    
    # Calculate returns
    returns_data = {}
    for commodity in selected_commodities:
        if commodity in commodities_data:
            returns_data[commodity] = commodities_data[commodity].pct_change() * 100
    
    # Plot cumulative returns
    fig = go.Figure()
    
    for commodity, returns in returns_data.items():
        cum_returns = (1 + returns / 100).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=cum_returns.values * 100,
                name=commodity,
                mode="lines",
                line=dict(color=METALS_DATA[commodity]["color"], width=2),
                hovertemplate=f"{commodity}: %{{y:.2f}}%<extra></extra>",
            )
        )
    
    fig.update_layout(
        title="Cumulative Returns (%)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
        height=500,
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Returns statistics
    st.subheader("📈 Return Statistics")
    
    stats_data = []
    for commodity, returns in returns_data.items():
        clean_returns = returns.dropna()
        stats_data.append({
            "Commodity": commodity,
            "Total Return": f"{((1 + clean_returns / 100).prod() - 1) * 100:.2f}%",
            "Ann. Return": f"{clean_returns.mean() * 252:.2f}%",
            "Ann. Volatility": f"{clean_returns.std() * (252 ** 0.5):.2f}%",
            "Sharpe": f"{(clean_returns.mean() / clean_returns.std()) * (252 ** 0.5):.2f}",
            "Max Drawdown": f"{(clean_returns.cumsum().cummax() - clean_returns.cumsum()).max():.2f}%",
        })
    
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

elif analysis_type == "Correlation Matrix":
    st.subheader("🔗 Correlation Matrix")
    
    # Create price DataFrame
    price_df = pd.DataFrame(commodities_data)
    
    # Calculate returns
    returns_df = price_df.pct_change().dropna()
    
    # Correlation matrix
    corr_matrix = returns_df.corr()
    
    # Plot heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmid=0,
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
    ))
    
    fig.update_layout(
        title="Commodity Return Correlations",
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Interpretation:**
    - Values close to +1: Strong positive correlation (move together)
    - Values close to -1: Strong negative correlation (move opposite)
    - Values close to 0: Little to no correlation
    """)

elif analysis_type == "Normalized Comparison":
    st.subheader("📊 Normalized Price Comparison")
    
    st.info("All prices normalized to 100 at the start date for easy comparison.")
    
    fig = go.Figure()
    
    for commodity in selected_commodities:
        if commodity in commodities_data:
            series = commodities_data[commodity]
            # Normalize to 100 at start
            normalized = (series / series.iloc[0]) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=normalized.index,
                    y=normalized.values,
                    name=commodity,
                    mode="lines",
                    line=dict(color=METALS_DATA[commodity]["color"], width=2),
                    hovertemplate=f"{commodity}: %{{y:.2f}}<extra></extra>",
                )
            )
    
    fig.update_layout(
        title="Normalized Prices (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        hovermode="x unified",
        height=600,
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="lightgray",
                activecolor="gray"
            ),
        )
    )
    
    # Add horizontal line at 100
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary
    st.subheader("🏆 Performance Summary")
    
    perf_data = []
    for commodity in selected_commodities:
        if commodity in commodities_data:
            series = commodities_data[commodity]
            change = ((series.iloc[-1] / series.iloc[0]) - 1) * 100
            perf_data.append({
                "Commodity": commodity,
                "Start Price": f"${series.iloc[0]:.2f}",
                "End Price": f"${series.iloc[-1]:.2f}",
                "Total Change": f"{change:+.2f}%",
                "Best Performer": "🏆" if change == max([((commodities_data[c].iloc[-1] / commodities_data[c].iloc[0]) - 1) * 100 for c in selected_commodities if c in commodities_data]) else "",
            })
    
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

# Download section
st.markdown("---")
st.subheader("💾 Download Data")

if st.button("📥 Download as CSV"):
    # Combine all data
    download_df = pd.DataFrame(commodities_data)
    csv = download_df.to_csv()
    
    st.download_button(
        label="Download Commodity Data",
        data=csv,
        file_name=f"commodity_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# Info footer
with st.expander("ℹ️ Data Sources & Available Assets"):
    st.markdown("""
    **Multi-Source Data Integration**
    
    This page combines data from:
    - **Yahoo Finance**: Precious metals ETFs (no API key needed)
    - **Alpha Vantage**: Energy, industrial metals, agricultural commodities
    
    ---
    
    ### Precious Metals (Yahoo Finance ETFs)
    - **Gold (GLD)**: SPDR Gold Trust - tracks gold bullion prices
    - **Silver (SLV)**: iShares Silver Trust - tracks silver bullion prices
    - **Platinum (PPLT)**: Aberdeen Physical Platinum - tracks platinum prices
    - **Palladium (PALL)**: Aberdeen Physical Palladium - tracks palladium prices
    
    ### Energy (Alpha Vantage)
    - **Crude Oil (WTI)**: West Texas Intermediate crude oil
    - **Crude Oil (Brent)**: Brent crude oil benchmark
    - **Natural Gas**: Natural gas spot prices
    
    ### Industrial Metals (Alpha Vantage)
    - **Copper**: Global copper spot prices
    - **Aluminum**: Aluminum spot prices
    
    ### Agricultural (Alpha Vantage)
    - **Wheat, Corn, Coffee, Cotton, Sugar**: Global commodity prices
    
    ---
    
    **API Configuration:**
    - **Alpha Vantage**: Free tier = 25 requests/day | Get key: https://www.alphavantage.co/support/#api-key
    - **Yahoo Finance**: No API key required | Unlimited requests
    
    **Data Caching:** 1 hour (to optimize API usage)
    """)

