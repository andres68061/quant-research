#!/usr/bin/env python3
"""
Excluded Stocks Viewer - Interactive page to explore penny stocks and data quality issues.

This page shows stocks that are automatically excluded from portfolio simulations
due to price filters, and allows detailed inspection of their price history.
"""

import sys
import warnings
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Path for manual validations
VALIDATIONS_FILE = ROOT / "data" / "manual_validations.json"


def load_manual_validations():
    """Load manually validated symbols from JSON file."""
    if VALIDATIONS_FILE.exists():
        with open(VALIDATIONS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_manual_validation(symbol, reason, threshold):
    """Save a manual validation to JSON file."""
    validations = load_manual_validations()
    validations[symbol] = {
        'validated_at': datetime.now().isoformat(),
        'reason': reason,
        'threshold_at_validation': threshold
    }
    VALIDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATIONS_FILE, 'w') as f:
        json.dump(validations, f, indent=2)
    return validations


def remove_manual_validation(symbol):
    """Remove a manual validation from JSON file."""
    validations = load_manual_validations()
    if symbol in validations:
        del validations[symbol]
        with open(VALIDATIONS_FILE, 'w') as f:
            json.dump(validations, f, indent=2)
    return validations

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

# Manual validations filter
manual_validations = load_manual_validations()
show_validated = st.sidebar.checkbox(
    "Show Only Manually Validated",
    value=False,
    help=f"Filter to show only the {len(manual_validations)} manually validated stocks"
)

if len(manual_validations) > 0:
    with st.sidebar.expander(f"📋 Validated Stocks ({len(manual_validations)})"):
        for symbol, info in manual_validations.items():
            validated_date = datetime.fromisoformat(info['validated_at']).strftime('%Y-%m-%d')
            st.caption(f"**{symbol}** - {validated_date}")

st.sidebar.markdown("---")

# Calculate exclusions
# Note: fillna(inf) ensures NaN values don't cause false exclusions
# We only want to exclude stocks that actually trade below threshold
price_mask = (df_prices_filtered.fillna(np.inf) >= price_threshold).all(axis=0)
valid_symbols = df_prices_filtered.columns[price_mask]
excluded_symbols = df_prices_filtered.columns[~price_mask]

# Apply manual validation filter if requested
if show_validated:
    excluded_symbols = [s for s in excluded_symbols if s in manual_validations]
    if len(excluded_symbols) == 0:
        st.info("ℹ️ No manually validated stocks to display. Validate stocks below to add them to the list.")
        st.stop()
    
excluded_symbols = list(excluded_symbols)  # Convert to list for indexing

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
        extreme_returns_positive = (returns > 0.5).sum()
        extreme_returns_negative = (returns < -0.5).sum()
        extreme_returns_total = (returns.abs() > 0.5).sum()
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
            st.metric("Extreme Returns (+)", extreme_returns_positive, help="Days with >50% gain")
        with col8:
            st.metric("Extreme Returns (-)", extreme_returns_negative, help="Days with >50% loss")
        
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            volatility = returns.std() * np.sqrt(252)
            st.metric("Annualized Vol", f"{volatility*100:.1f}%")
        with col10:
            st.metric("Extreme Returns (Total)", extreme_returns_total, help="Days with >50% absolute return")
        with col11:
            st.empty()  # Placeholder
        with col12:
            st.empty()  # Placeholder
        
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
        
        # Mark extreme positive returns (>50% gains) in GREEN
        extreme_gains_mask = returns > 0.5
        if extreme_gains_mask.any():
            # Shift index by 1 to align with price (returns are calculated from previous day)
            extreme_gains_dates = returns[extreme_gains_mask].index
            extreme_gains_prices = symbol_prices.loc[extreme_gains_dates]
            fig.add_trace(
                go.Scatter(
                    x=extreme_gains_dates,
                    y=extreme_gains_prices.values,
                    mode='markers',
                    name='Extreme Gain (>50%)',
                    marker=dict(color='green', size=12, symbol='triangle-up', line=dict(width=2, color='darkgreen')),
                    hovertemplate='%{x}<br>Price: $%{y:.2f}<br>(Extreme gain)<extra></extra>'
                )
            )
        
        # Mark extreme negative returns (>50% losses) in ORANGE/RED
        extreme_losses_mask = returns < -0.5
        if extreme_losses_mask.any():
            extreme_losses_dates = returns[extreme_losses_mask].index
            extreme_losses_prices = symbol_prices.loc[extreme_losses_dates]
            fig.add_trace(
                go.Scatter(
                    x=extreme_losses_dates,
                    y=extreme_losses_prices.values,
                    mode='markers',
                    name='Extreme Loss (>50%)',
                    marker=dict(color='orange', size=12, symbol='triangle-down', line=dict(width=2, color='darkorange')),
                    hovertemplate='%{x}<br>Price: $%{y:.2f}<br>(Extreme loss)<extra></extra>'
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
        
        # Add vertical lines for extreme returns thresholds
        fig_returns.add_vline(
            x=50,
            line_dash="dash",
            line_color="green",
            annotation_text="Extreme Gain (+50%)",
            annotation_position="top right"
        )
        
        fig_returns.add_vline(
            x=-50,
            line_dash="dash",
            line_color="orange",
            annotation_text="Extreme Loss (-50%)",
            annotation_position="top left"
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
        
        if extreme_returns_total > 0:
            reasons.append(f"- **Extreme volatility**: {extreme_returns_total} days with >50% absolute return ({extreme_returns_positive} gains, {extreme_returns_negative} losses)")
        
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
        
        # Manual Validation Section
        st.markdown("---")
        st.markdown("### ✅ Manual Validation")
        
        is_validated = selected_symbol in manual_validations
        
        if is_validated:
            val_info = manual_validations[selected_symbol]
            val_date = datetime.fromisoformat(val_info['validated_at']).strftime('%Y-%m-%d %H:%M')
            
            st.success(f"✅ This stock was manually validated on {val_date}")
            st.markdown(f"**Validation Reason:** {val_info['reason']}")
            st.caption(f"Price threshold at validation: ${val_info['threshold_at_validation']}")
            
            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                if st.button("🗑️ Remove Validation", key=f"remove_{selected_symbol}"):
                    remove_manual_validation(selected_symbol)
                    st.success(f"Removed validation for {selected_symbol}")
                    st.rerun()
        else:
            st.warning(f"""
            ⚠️ **{selected_symbol}** is currently excluded from portfolio simulations.
            
            If after reviewing the data you believe this stock should be included despite the price threshold, 
            you can manually validate it here. The stock will remain excluded by default, but will be tracked 
            in your manual validations list for future reference.
            """)
            
            validation_reason = st.text_area(
                "Reason for manual validation",
                placeholder="E.g., 'Price spike was due to stock split correction, actual trading history is stable'",
                help="Document why you're overriding the automatic exclusion",
                key=f"reason_{selected_symbol}"
            )
            
            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                if st.button("✅ Validate Stock", key=f"validate_{selected_symbol}", type="primary"):
                    if validation_reason.strip():
                        save_manual_validation(selected_symbol, validation_reason.strip(), price_threshold)
                        st.success(f"✅ Manually validated {selected_symbol}")
                        st.info("💡 **Note:** This marks the stock as reviewed. To actually include it in simulations, you would need to adjust the price threshold or implement custom filtering logic.")
                        st.rerun()
                    else:
                        st.error("❌ Please provide a reason for validation")
            
            with col_btn2:
                st.caption("💡 Tip: Manual validations are saved to `data/manual_validations.json`")
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
