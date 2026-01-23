#!/usr/bin/env python3
"""
Economic Indicators Page

Interactive analysis of key economic indicators from FRED.
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fredapi import Fred
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import FRED_API_KEY

# Page configuration
st.set_page_config(
    page_title="Economic Indicators",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Economic Indicators Dashboard")
st.markdown("---")

# Check API key
if not FRED_API_KEY:
    st.error("""
    ❌ **FRED API Key not found!**
    
    Please add your FRED API key to the `.env` file:
    ```
    FRED_API_KEY=your_key_here
    ```
    
    Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """)
    st.stop()

# Initialize FRED client
@st.cache_resource
def get_fred_client():
    """Initialize and cache FRED client."""
    return Fred(api_key=FRED_API_KEY)

fred = get_fred_client()

# Economic indicators metadata
INDICATORS_DATA = {
    "Interest Rates": {
        "DFF": {
            "name": "Federal Funds Rate",
            "unit": "%",
            "color": "#1f77b4",
        },
        "DGS10": {
            "name": "10-Year Treasury Rate",
            "unit": "%",
            "color": "#ff7f0e",
        },
        "DGS2": {
            "name": "2-Year Treasury Rate",
            "unit": "%",
            "color": "#2ca02c",
        },
    },
    "Inflation": {
        "CPIAUCSL": {
            "name": "Consumer Price Index (CPI)",
            "unit": "Index",
            "color": "#d62728",
        },
        "PCEPI": {
            "name": "Personal Consumption Expenditures",
            "unit": "Index",
            "color": "#9467bd",
        },
    },
    "GDP & Growth": {
        "GDP": {
            "name": "Gross Domestic Product",
            "unit": "Billions $",
            "color": "#8c564b",
        },
        "GDPC1": {
            "name": "Real GDP",
            "unit": "Billions 2012 $",
            "color": "#e377c2",
        },
    },
    "Employment": {
        "UNRATE": {
            "name": "Unemployment Rate",
            "unit": "%",
            "color": "#7f7f7f",
        },
        "PAYEMS": {
            "name": "Nonfarm Payrolls",
            "unit": "Thousands",
            "color": "#bcbd22",
        },
    },
}

# Sidebar controls
st.sidebar.header("⚙️ Dashboard Settings")

# Category selection
selected_category = st.sidebar.selectbox(
    "Select Category",
    list(INDICATORS_DATA.keys()),
    help="Choose indicator category to analyze",
)

# Get indicators for selected category
available_indicators = INDICATORS_DATA[selected_category]

selected_indicators = st.sidebar.multiselect(
    "Select Indicators",
    list(available_indicators.keys()),
    default=list(available_indicators.keys())[:2],
    help="Choose which indicators to display",
)

# Date range
max_years = st.sidebar.slider(
    "Historical Period (Years)",
    min_value=1,
    max_value=50,
    value=10,
    help="How many years of historical data to fetch",
)

start_date = (datetime.now() - timedelta(days=365 * max_years)).strftime("%Y-%m-%d")

# Display options
show_yoy_change = st.sidebar.checkbox(
    "Show Year-over-Year % Change",
    value=False,
    help="Calculate and display YoY percentage changes",
)

show_recession_bars = st.sidebar.checkbox(
    "Show Recession Periods",
    value=True,
    help="Highlight recession periods (NBER dates)",
)

# Fetch button
if st.sidebar.button("🔄 Fetch Data", type="primary"):
    st.session_state.fetch_econ_data = True

# Fetch recession data
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_recession_dates(start_date_str):
    """Fetch NBER recession indicator."""
    try:
        recession = fred.get_series("USREC", observation_start=start_date_str)
        # Find recession periods
        recession_periods = []
        in_recession = False
        start = None
        
        for date, value in recession.items():
            if value == 1 and not in_recession:
                start = date
                in_recession = True
            elif value == 0 and in_recession:
                recession_periods.append((start, date))
                in_recession = False
        
        if in_recession:
            recession_periods.append((start, recession.index[-1]))
        
        return recession_periods
    except:
        return []

# Fetch and cache data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_indicator_data(indicators_list, category, start_date_str):
    """Fetch economic indicators from FRED."""
    data = {}
    metadata = {}
    
    indicators_info = INDICATORS_DATA[category]
    
    for indicator_id in indicators_list:
        try:
            series = fred.get_series(indicator_id, observation_start=start_date_str)
            
            if not series.empty:
                data[indicator_id] = series
                metadata[indicator_id] = {
                    "name": indicators_info[indicator_id]["name"],
                    "unit": indicators_info[indicator_id]["unit"],
                    "count": len(series),
                    "start": series.index[0],
                    "end": series.index[-1],
                    "latest": series.iloc[-1],
                }
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not fetch {indicator_id}: {str(e)}")
    
    return data, metadata

# Main content
if not selected_indicators:
    st.info("👈 Please select at least one indicator from the sidebar.")
    st.stop()

if "fetch_econ_data" not in st.session_state:
    st.info("👈 Click **Fetch Data** in the sidebar to load economic indicators.")
    st.stop()

# Fetch data
with st.spinner(f"Fetching {len(selected_indicators)} indicators..."):
    indicator_data, metadata = fetch_indicator_data(
        selected_indicators, selected_category, start_date
    )
    
    if show_recession_bars:
        recession_periods = fetch_recession_dates(start_date)

if not indicator_data:
    st.error("❌ No data available. Check your FRED API key and try again.")
    st.stop()

# Display metadata
st.success(f"✅ Loaded {len(indicator_data)} indicators")

cols = st.columns(len(indicator_data))
for i, (indicator_id, meta) in enumerate(metadata.items()):
    with cols[i]:
        latest_value = meta['latest']
        unit = meta['unit']
        
        # Format value based on unit
        if unit == "%":
            display_value = f"{latest_value:.2f}%"
        elif "Billions" in unit:
            display_value = f"${latest_value:.0f}B"
        elif "Thousands" in unit:
            display_value = f"{latest_value:.0f}K"
        else:
            display_value = f"{latest_value:.2f}"
        
        st.metric(
            meta['name'],
            display_value,
            help=f"Latest: {meta['end'].strftime('%Y-%m-%d')}",
        )

st.markdown("---")

# Main chart
st.subheader(f"📊 {selected_category} - Historical Trends")

fig = make_subplots(
    rows=len(selected_indicators),
    cols=1,
    subplot_titles=[metadata[ind]["name"] for ind in selected_indicators],
    vertical_spacing=0.08,
)

for i, indicator_id in enumerate(selected_indicators, 1):
    series = indicator_data[indicator_id]
    info = INDICATORS_DATA[selected_category][indicator_id]
    
    if show_yoy_change and indicator_id not in ["UNRATE", "DFF", "DGS10", "DGS2"]:
        # Calculate YoY % change for non-rate indicators
        yoy_change = series.pct_change(12) * 100  # Assume monthly data
        plot_series = yoy_change
        y_title = "YoY % Change"
    else:
        plot_series = series
        y_title = info["unit"]
    
    fig.add_trace(
        go.Scatter(
            x=plot_series.index,
            y=plot_series.values,
            name=info["name"],
            mode="lines",
            line=dict(color=info["color"], width=2),
            hovertemplate=f"{info['name']}: %{{y:.2f}}<extra></extra>",
        ),
        row=i,
        col=1,
    )
    
    # Add recession shading
    if show_recession_bars:
        for start, end in recession_periods:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=i,
                col=1,
            )
    
    fig.update_yaxes(title_text=y_title, row=i, col=1)

fig.update_layout(
    height=300 * len(selected_indicators),
    showlegend=False,
    hovermode="x unified",
    xaxis=dict(
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(count=10, label="10y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="lightgray",
            activecolor="gray"
        ),
    )
)

fig.update_xaxes(title_text="Date", row=len(selected_indicators), col=1)

st.plotly_chart(fig, use_container_width=True)

# Statistics table
st.subheader("📈 Summary Statistics")

stats_data = []
for indicator_id in selected_indicators:
    series = indicator_data[indicator_id]
    meta = metadata[indicator_id]
    
    stats_data.append({
        "Indicator": meta["name"],
        "Latest": f"{series.iloc[-1]:.2f}",
        "Min": f"{series.min():.2f}",
        "Max": f"{series.max():.2f}",
        "Mean": f"{series.mean():.2f}",
        "Std Dev": f"{series.std():.2f}",
        "Unit": meta["unit"],
    })

st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

# Correlation analysis (if multiple indicators)
if len(selected_indicators) > 1:
    st.markdown("---")
    st.subheader("🔗 Correlation Analysis")
    
    # Create DataFrame
    corr_df = pd.DataFrame(indicator_data)
    corr_df.columns = [metadata[ind]["name"] for ind in selected_indicators]
    
    # Calculate correlation
    corr_matrix = corr_df.corr()
    
    # Plot heatmap
    fig_corr = go.Figure(data=go.Heatmap(
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
    
    fig_corr.update_layout(
        title="Indicator Correlations",
        height=400,
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# Download section
st.markdown("---")
st.subheader("💾 Download Data")

if st.button("📥 Download as CSV"):
    # Combine all data
    download_df = pd.DataFrame(indicator_data)
    download_df.columns = [metadata[ind]["name"] for ind in selected_indicators]
    csv = download_df.to_csv()
    
    st.download_button(
        label="Download Economic Data",
        data=csv,
        file_name=f"economic_indicators_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# Info footer
with st.expander("ℹ️ Data Source & Definitions"):
    st.markdown(f"""
    **Data Source:** Federal Reserve Economic Data (FRED)
    
    **{selected_category} Indicators:**
    """)
    
    for indicator_id, info in INDICATORS_DATA[selected_category].items():
        st.markdown(f"- **{info['name']}** ({indicator_id}): {info['unit']}")
    
    st.markdown("""
    
    **Update Frequencies:**
    - Interest Rates: Daily
    - CPI/Inflation: Monthly
    - GDP: Quarterly
    - Employment: Monthly
    
    **Recession Periods:** Based on NBER (National Bureau of Economic Research) dates
    
    **Data cached for:** 1 hour
    """)

