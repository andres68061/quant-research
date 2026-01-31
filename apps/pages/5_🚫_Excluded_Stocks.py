#!/usr/bin/env python3
"""
Excluded Stocks Viewer - Interactive page to explore penny stocks and data quality issues.

This page shows stocks that are automatically excluded from portfolio simulations
due to price filters, and allows detailed inspection of their price history.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Page configuration
st.set_page_config(
    page_title="Excluded Stocks",
    page_icon="🚫",
    layout="wide",
)

st.title("🚫 Excluded Stocks Viewer")
st.markdown("""
This page shows stocks that are automatically excluded from portfolio simulations
to prevent data quality issues and unrealistic returns.

**Why stocks are excluded:**
- Price < $5 on any day (penny stocks)
- Missing data for extended periods
- Extreme volatility (data corruption)
""")

st.markdown("---")


@st.cache_data
def load_prices():
    """Load prices data."""
    candidates = [ROOT / "data" / "factors", ROOT.parent / "data" / "factors"]
    
    data_dir = None
    for candidate in candidates:
        if candidate.exists():
            data_dir = candidate
            break
    
    if data_dir is None:
        st.error("❌ Data directory not found.")
        st.stop()
    
    prices_path = data_dir / "prices.parquet"
    
    if not prices_path.exists():
        st.error(f"❌ Prices file not found: {prices_path}")
        st.stop()
    
    return pd.read_parquet(prices_path)


# Load data
df_prices = load_prices()

st.sidebar.header("🔍 Filter Settings")

# Date range selection
min_date = df_prices.index.min().date()
max_date = df_prices.index.max().date()

date_range = st.sidebar.date_input(
    "Analysis Period",
    value=(max_date - pd.Timedelta(days=365 * 5), max_date),
    min_value=min_date,
    max_value=max_date,
    help="Select period to analyze for exclusions"
)

if len(date_range) != 2:
    st.warning("⚠️ Please select both start and end dates")
    st.stop()

start_date, end_date = date_range

# Filter prices to date range
df_prices_filtered = df_prices[
    (df_prices.index.date >= start_date) & (df_prices.index.date <= end_date)
]

# Price threshold
price_threshold = st.sidebar.slider(
    "Price Threshold ($)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.5,
    help="Stocks below this price on any day are excluded"
)

st.sidebar.markdown("---")

# Calculate exclusions
price_mask = (df_prices_filtered >= price_threshold).all(axis=0)
valid_symbols = df_prices_filtered.columns[price_mask]
excluded_symbols = df_prices_filtered.columns[~price_mask]

# Summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Stocks",
        len(df_prices_filtered.columns),
        help="All stocks in dataset"
    )

with col2:
    st.metric(
        "✅ Valid Stocks",
        len(valid_symbols),
        delta=f"{len(valid_symbols)/len(df_prices_filtered.columns)*100:.1f}%",
        help=f"Stocks always >= ${price_threshold}"
    )

with col3:
    st.metric(
        "🚫 Excluded Stocks",
        len(excluded_symbols),
        delta=f"{len(excluded_symbols)/len(df_prices_filtered.columns)*100:.1f}%",
        delta_color="inverse",
        help=f"Stocks < ${price_threshold} on at least one day"
    )

st.markdown("---")

# Exclusion Analysis
st.header("📊 Exclusion Analysis")

if len(excluded_symbols) == 0:
    st.success(f"✅ No stocks excluded at ${price_threshold} threshold!")
    st.stop()

# Create exclusion statistics
exclusion_stats = []
for symbol in excluded_symbols:
    symbol_prices = df_prices_filtered[symbol].dropna()
    if len(symbol_prices) > 0:
        min_price = symbol_prices.min()
        max_price = symbol_prices.max()
        current_price = symbol_prices.iloc[-1]
        days_below = (symbol_prices < price_threshold).sum()
        pct_below = days_below / len(symbol_prices) * 100
        
        exclusion_stats.append({
            'Symbol': symbol,
            'Min Price': min_price,
            'Max Price': max_price,
            'Current Price': current_price,
            'Days Below Threshold': days_below,
            '% Days Below': pct_below
        })

df_exclusions = pd.DataFrame(exclusion_stats)
df_exclusions = df_exclusions.sort_values('Min Price')

# Display table
st.markdown(f"### Excluded Stocks (${price_threshold} threshold)")
st.dataframe(
    df_exclusions.style.format({
        'Min Price': '${:.2f}',
        'Max Price': '${:.2f}',
        'Current Price': '${:.2f}',
        'Days Below Threshold': '{:.0f}',
        '% Days Below': '{:.1f}%'
    }),
    use_container_width=True,
    height=400
)

st.markdown("---")

# Stock Detail Viewer
st.header("📈 Price Chart Viewer")
st.markdown("Select a stock to view its price history and understand why it was excluded.")

# Stock selector
selected_symbol = st.selectbox(
    "Select Stock to Analyze",
    options=sorted(excluded_symbols),
    help="Choose a stock to view detailed price chart"
)

if selected_symbol:
    st.markdown(f"### {selected_symbol} - Price History")
    
    # Get price data
    symbol_prices = df_prices_filtered[selected_symbol].dropna()
    
    if len(symbol_prices) == 0:
        st.warning(f"⚠️ No price data available for {selected_symbol} in selected period")
    else:
        # Calculate statistics
        min_price = symbol_prices.min()
        max_price = symbol_prices.max()
        current_price = symbol_prices.iloc[-1]
        days_below = (symbol_prices < price_threshold).sum()
        pct_below = days_below / len(symbol_prices) * 100
        
        # Calculate returns
        returns = symbol_prices.pct_change().dropna()
        extreme_returns = (returns.abs() > 0.5).sum()
        max_daily_gain = returns.max()
        max_daily_loss = returns.min()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min Price", f"${min_price:.2f}")
        with col2:
            st.metric("Max Price", f"${max_price:.2f}")
        with col3:
            st.metric("Current Price", f"${current_price:.2f}")
        with col4:
            st.metric(
                f"Days < ${price_threshold}",
                f"{days_below}",
                delta=f"{pct_below:.1f}%",
                delta_color="inverse"
            )
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Max Daily Gain", f"{max_daily_gain*100:.1f}%")
        with col6:
            st.metric("Max Daily Loss", f"{max_daily_loss*100:.1f}%")
        with col7:
            st.metric("Extreme Returns", extreme_returns, help="Days with >50% return")
        with col8:
            volatility = returns.std() * np.sqrt(252)
            st.metric("Annualized Vol", f"{volatility*100:.1f}%")
        
        # Create price chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=symbol_prices.index,
                y=symbol_prices.values,
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2),
                hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
            )
        )
        
        # Add threshold line
        fig.add_hline(
            y=price_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"${price_threshold} Threshold",
            annotation_position="right"
        )
        
        # Highlight periods below threshold
        below_threshold = symbol_prices < price_threshold
        if below_threshold.any():
            fig.add_trace(
                go.Scatter(
                    x=symbol_prices[below_threshold].index,
                    y=symbol_prices[below_threshold].values,
                    mode='markers',
                    name=f'Below ${price_threshold}',
                    marker=dict(color='red', size=8, symbol='x'),
                    hovertemplate='%{x}<br>Price: $%{y:.2f}<br>(Below threshold)<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=f"{selected_symbol} Price History ({start_date} to {end_date})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            showlegend=True,
            yaxis=dict(type='log' if min_price < 1 else 'linear')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns distribution
        st.markdown("### 📊 Daily Returns Distribution")
        
        fig_returns = go.Figure()
        
        fig_returns.add_trace(
            go.Histogram(
                x=returns.values * 100,
                nbinsx=50,
                name='Daily Returns',
                marker_color='lightblue',
                hovertemplate='Return: %{x:.1f}%<br>Count: %{y}<extra></extra>'
            )
        )
        
        fig_returns.update_layout(
            title=f"{selected_symbol} Daily Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Explanation
        st.markdown("### 🔍 Why This Stock is Excluded")
        
        reasons = []
        
        if min_price < 1:
            reasons.append(f"- **Penny stock**: Minimum price of ${min_price:.2f} (< $1)")
        elif min_price < price_threshold:
            reasons.append(f"- **Below threshold**: Minimum price of ${min_price:.2f} (< ${price_threshold})")
        
        if pct_below > 50:
            reasons.append(f"- **Frequently below threshold**: {pct_below:.1f}% of days below ${price_threshold}")
        
        if extreme_returns > 0:
            reasons.append(f"- **Extreme volatility**: {extreme_returns} days with >50% daily return")
        
        if volatility > 1.0:
            reasons.append(f"- **Very high volatility**: {volatility*100:.1f}% annualized (>100%)")
        
        if reasons:
            st.markdown("**This stock is excluded because:**")
            for reason in reasons:
                st.markdown(reason)
        
        st.info("""
        **Impact on Portfolio:**
        - Small prices create huge percentage returns
        - Example: $0.50 → $1.00 = 100% return (not realistic for large positions)
        - Low liquidity means high transaction costs
        - Data quality often poor for penny stocks
        
        **Excluding these stocks ensures more realistic backtest results.**
        """)

st.markdown("---")

# Additional Statistics
with st.expander("📊 Additional Statistics"):
    st.markdown("### Price Distribution of Excluded Stocks")
    
    if len(df_exclusions) > 0:
        fig_dist = go.Figure()
        
        fig_dist.add_trace(
            go.Histogram(
                x=df_exclusions['Min Price'].values,
                nbinsx=30,
                name='Min Price',
                marker_color='lightcoral',
                opacity=0.7
            )
        )
        
        fig_dist.add_vline(
            x=price_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"${price_threshold} Threshold"
        )
        
        fig_dist.update_layout(
            title="Distribution of Minimum Prices (Excluded Stocks)",
            xaxis_title="Minimum Price ($)",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Price Statistics:**")
            st.write(df_exclusions[['Min Price', 'Max Price', 'Current Price']].describe())
        
        with col2:
            st.markdown("**Exclusion Statistics:**")
            st.write(df_exclusions[['Days Below Threshold', '% Days Below']].describe())

st.markdown("---")
st.caption("💡 **Tip:** Adjust the price threshold in the sidebar to see how it affects exclusions")
