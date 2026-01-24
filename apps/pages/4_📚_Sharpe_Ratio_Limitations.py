"""
Sharpe Ratio Limitations - Educational Demo

Shows how investments with the same Sharpe ratio can have very different
return profiles, drawdowns, and risk characteristics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

# Page config
st.set_page_config(
    page_title="Sharpe Ratio Limitations",
    page_icon="📚",
    layout="wide",
)

st.title("📚 When Sharpe Ratio Fails")
st.markdown("**Understanding the limitations of risk-adjusted metrics**")

st.info("""
**The Sharpe Ratio** = (Return - Risk-Free Rate) / Volatility

While widely used, it has significant **blind spots**. This page demonstrates how 
investments with the **same Sharpe ratio** can have **very different** risk profiles.
""")

st.markdown("---")

# Sidebar controls
st.sidebar.header("⚙️ Simulation Settings")

target_sharpe = st.sidebar.slider(
    "Target Sharpe Ratio",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="All simulated investments will have approximately this Sharpe ratio"
)

n_days = st.sidebar.slider(
    "Simulation Period (Trading Days)",
    min_value=252,
    max_value=2520,
    value=1260,  # 5 years
    step=252,
    help="Number of trading days (252 = 1 year)"
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=1,
    max_value=9999,
    value=42,
    help="For reproducibility"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 What You'll Learn")
st.sidebar.markdown("""
1. Same Sharpe ≠ Same Risk
2. Volatility isn't the only risk
3. Path matters (drawdowns)
4. Distribution shape matters
5. Better alternatives
""")

# Set random seed
np.random.seed(random_seed)


# --- Generate Investments with Same Sharpe Ratio ---

def generate_investment(name, target_sharpe, n_days, 
                       vol_level="medium", skew=0, kurtosis=0, drawdown_events=0,
                       color="#1f77b4", rf=0.02):
    """
    Generate daily returns for an investment with EXACT target Sharpe ratio.
    
    The key insight: we generate returns with desired characteristics (skew, kurtosis, etc.)
    then NORMALIZE them to achieve the exact target Sharpe ratio.
    
    Parameters:
    - target_sharpe: The exact Sharpe ratio to achieve
    - vol_level: "low", "medium", or "high" volatility
    - skew: Skewness of returns (0 = normal, negative = left skew)
    - kurtosis: Excess kurtosis (0 = normal, positive = fat tails)
    - drawdown_events: Number of "crash" events to inject
    """
    
    # Set volatility based on level
    vol_map = {"low": 0.12, "medium": 0.18, "high": 0.30}
    annual_vol = vol_map.get(vol_level, 0.18)
    daily_vol = annual_vol / np.sqrt(252)
    
    # Generate base returns (zero mean, unit variance initially)
    returns = np.random.normal(0, daily_vol, n_days)
    
    # Add skewness (negative = occasional large losses)
    if skew < 0:
        crash_days = np.random.choice(n_days, size=int(abs(skew) * 50), replace=False)
        returns[crash_days] -= daily_vol * np.random.uniform(2, 5, len(crash_days))
    
    # Add kurtosis (fat tails - extreme events both ways)
    if kurtosis > 0:
        extreme_days = np.random.choice(n_days, size=int(kurtosis * 20), replace=False)
        extreme_shocks = np.random.choice([-1, 1], len(extreme_days)) * daily_vol * np.random.uniform(3, 6, len(extreme_days))
        returns[extreme_days] += extreme_shocks
    
    # Add drawdown events (crashes followed by recovery)
    if drawdown_events > 0:
        event_starts = np.random.choice(range(50, n_days - 50), size=drawdown_events, replace=False)
        for start in event_starts:
            duration = np.random.randint(10, 30)
            crash_size = np.random.uniform(0.10, 0.25)
            returns[start:start+duration] -= crash_size / duration
    
    # === NORMALIZE TO TARGET SHARPE RATIO ===
    # Sharpe = (mean - rf/252) / std * sqrt(252)
    # So: mean = rf/252 + target_sharpe * std / sqrt(252)
    
    current_std = returns.std()
    target_daily_mean = rf / 252 + target_sharpe * current_std / np.sqrt(252)
    
    # Shift returns to achieve target mean (preserves shape/distribution)
    returns = returns - returns.mean() + target_daily_mean
    
    # Create price series
    prices = 100 * np.cumprod(1 + returns)
    
    # Create DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'return': returns * 100  # Convert to percentage
    }).set_index('date')
    
    return {
        'name': name,
        'df': df,
        'color': color,
        'returns': returns,
        'prices': prices,
    }


# Calculate Sharpe Ratio
def calc_sharpe(returns, rf=0.02):
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - rf / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calc_sortino(returns, rf=0.02):
    """Calculate Sortino ratio (only penalizes downside)."""
    excess_returns = returns - rf / 252
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
    return np.sqrt(252) * excess_returns.mean() / downside_std


def calc_max_drawdown(prices):
    """Calculate maximum drawdown."""
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100


def calc_calmar(returns, prices):
    """Calculate Calmar ratio (return / max drawdown)."""
    annual_return = returns.mean() * 252 * 100
    max_dd = abs(calc_max_drawdown(prices))
    return annual_return / max_dd if max_dd != 0 else 0


# Generate investments with SAME Sharpe ratio but different characteristics
st.subheader(f"🎲 Simulated Investments (All with Sharpe Ratio ≈ {target_sharpe})")

investments = []

# 1. Steady Eddie - Low vol, low return, normal distribution
investments.append(generate_investment(
    name="Steady Eddie",
    target_sharpe=target_sharpe,
    n_days=n_days,
    vol_level="low",
    skew=0,
    kurtosis=0,
    color="#2ecc71"  # Green
))

# 2. Rollercoaster - High vol, high return, normal distribution
investments.append(generate_investment(
    name="Rollercoaster",
    target_sharpe=target_sharpe,
    n_days=n_days,
    vol_level="high",
    skew=0,
    kurtosis=0,
    color="#e74c3c"  # Red
))

# 3. Sneaky Losses - Medium vol, negative skew (occasional crashes)
investments.append(generate_investment(
    name="Sneaky Losses",
    target_sharpe=target_sharpe,
    n_days=n_days,
    vol_level="medium",
    skew=-2,
    kurtosis=0,
    color="#9b59b6"  # Purple
))

# 4. Fat Tails - Medium vol, high kurtosis (extreme events both ways)
investments.append(generate_investment(
    name="Fat Tails",
    target_sharpe=target_sharpe,
    n_days=n_days,
    vol_level="medium",
    skew=0,
    kurtosis=3,
    color="#f39c12"  # Orange
))

# 5. Drawdown Prone - Periodic crashes followed by recovery
investments.append(generate_investment(
    name="Crash & Recover",
    target_sharpe=target_sharpe,
    n_days=n_days,
    vol_level="medium",
    skew=0,
    kurtosis=0,
    drawdown_events=3,
    color="#3498db"  # Blue
))

# Verify all have same Sharpe ratio
st.markdown("### ✅ Sharpe Ratio Verification")
sharpe_cols = st.columns(len(investments))
for i, inv in enumerate(investments):
    actual_sharpe = calc_sharpe(inv['returns'])
    with sharpe_cols[i]:
        st.metric(
            inv['name'],
            f"{actual_sharpe:.2f}",
            delta=f"Target: {target_sharpe:.1f}",
            delta_color="off"
        )

st.success(f"✅ All investments have Sharpe Ratio ≈ **{target_sharpe:.1f}** — yet look how different they are!")

# --- VISUALIZATION 1: Cumulative Returns ---
st.markdown("---")
st.subheader("📈 Cumulative Returns")
st.markdown("**Same Sharpe Ratio, VERY Different Journeys**")

fig1 = go.Figure()

for inv in investments:
    fig1.add_trace(
        go.Scatter(
            x=inv['df'].index,
            y=inv['df']['price'],
            name=inv['name'],
            mode='lines',
            line=dict(color=inv['color'], width=2),
            hovertemplate=f"{inv['name']}<br>Date: %{{x}}<br>Value: $%{{y:.2f}}<extra></extra>"
        )
    )

fig1.update_layout(
    title="Growth of $100 Investment",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",
    height=500,
    showlegend=True,
    xaxis=dict(
        rangeslider=dict(visible=True, yaxis=dict(rangemode='auto')),
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="lightgray",
            activecolor="gray"
        ),
    ),
    yaxis=dict(fixedrange=False, autorange=True),
)

fig1.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5, 
               annotation_text="Starting Value")

st.plotly_chart(fig1, use_container_width=True)

# --- VISUALIZATION 2: Drawdowns ---
st.subheader("📉 Drawdowns (Peak-to-Trough Declines)")
st.markdown("**The pain you feel along the way**")

fig2 = go.Figure()

for inv in investments:
    prices = inv['df']['price'].values
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak * 100
    
    fig2.add_trace(
        go.Scatter(
            x=inv['df'].index,
            y=drawdown,
            name=inv['name'],
            mode='lines',
            line=dict(color=inv['color'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba{tuple(list(int(inv['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}",
            hovertemplate=f"{inv['name']}<br>Drawdown: %{{y:.2f}}%<extra></extra>"
        )
    )

fig2.update_layout(
    title="Drawdown Over Time",
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    hovermode="x unified",
    height=400,
    showlegend=True,
)

st.plotly_chart(fig2, use_container_width=True)

# --- VISUALIZATION 3: Return Distributions ---
st.subheader("📊 Return Distributions")
st.markdown("**Same average, different shapes**")

fig3 = make_subplots(rows=1, cols=len(investments), 
                     subplot_titles=[inv['name'] for inv in investments])

for i, inv in enumerate(investments, 1):
    returns = inv['returns'] * 100  # Convert to %
    
    fig3.add_trace(
        go.Histogram(
            x=returns,
            name=inv['name'],
            marker_color=inv['color'],
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=1, col=i
    )
    
    # Add vertical line at mean
    mean_ret = returns.mean()
    fig3.add_vline(x=mean_ret, line_dash="dash", line_color="black", 
                   row=1, col=i, opacity=0.5)

fig3.update_layout(
    height=350,
    title_text="Daily Return Distributions",
)

for i in range(1, len(investments) + 1):
    fig3.update_xaxes(title_text="Daily Return (%)", row=1, col=i)
    fig3.update_yaxes(title_text="Frequency" if i == 1 else "", row=1, col=i)

st.plotly_chart(fig3, use_container_width=True)

# --- METRICS COMPARISON TABLE ---
st.markdown("---")
st.subheader("📋 Comprehensive Metrics Comparison")

metrics_data = []

for inv in investments:
    returns = inv['returns']
    prices = inv['prices']
    
    sharpe = calc_sharpe(returns)
    sortino = calc_sortino(returns)
    max_dd = calc_max_drawdown(prices)
    calmar = calc_calmar(returns, prices)
    
    # Distribution stats
    skewness = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    # Best/worst days
    best_day = returns.max() * 100
    worst_day = returns.min() * 100
    
    # Final value
    final_value = prices[-1]
    total_return = (final_value / 100 - 1) * 100
    
    metrics_data.append({
        "Investment": inv['name'],
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd:.1f}%",
        "Calmar Ratio": f"{calmar:.2f}",
        "Total Return": f"{total_return:.1f}%",
        "Volatility (Ann.)": f"{returns.std() * np.sqrt(252) * 100:.1f}%",
        "Skewness": f"{skewness:.2f}",
        "Kurtosis": f"{kurt:.2f}",
        "Win Rate": f"{win_rate:.1f}%",
        "Best Day": f"{best_day:.2f}%",
        "Worst Day": f"{worst_day:.2f}%",
    })

df_metrics = pd.DataFrame(metrics_data)
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

# --- KEY INSIGHTS ---
st.markdown("---")
st.subheader("💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.error("""
    ### ❌ What Sharpe Ratio Misses
    
    1. **Skewness** - Doesn't penalize negative skew (crash risk)
    2. **Kurtosis** - Ignores fat tails (extreme events)
    3. **Drawdowns** - Doesn't care how deep you fall
    4. **Path Dependency** - Same end point, different journeys
    5. **Upside vs Downside** - Penalizes good volatility too!
    """)

with col2:
    st.success("""
    ### ✅ Better Alternatives
    
    1. **Sortino Ratio** - Only penalizes downside volatility
    2. **Calmar Ratio** - Return / Max Drawdown
    3. **Omega Ratio** - Considers entire return distribution
    4. **Max Drawdown** - Worst peak-to-trough loss
    5. **VaR / CVaR** - Tail risk measures
    """)

# --- EDUCATIONAL CONTENT ---
st.markdown("---")
st.subheader("📖 Understanding the Math")

with st.expander("🎓 Why Same Sharpe Doesn't Mean Same Risk"):
    st.markdown(f"""
    ### The Sharpe Ratio Formula
    
    $$\\text{{Sharpe Ratio}} = \\frac{{R_p - R_f}}{{\\sigma_p}}$$
    
    Where:
    - $R_p$ = Portfolio return
    - $R_f$ = Risk-free rate  
    - $\\sigma_p$ = Portfolio standard deviation (volatility)
    
    ### The Problem
    
    **Standard deviation treats all volatility equally:**
    - A +5% day and a -5% day contribute the same to volatility
    - But investors **hate** -5% days more than they **love** +5% days!
    
    **It assumes normal distributions:**
    - Real returns have **fat tails** (more extreme events than expected)
    - Real returns have **negative skew** (crashes are bigger than rallies)
    
    ### Example from This Simulation
    
    All five investments have **exactly the same** Sharpe ≈ {target_sharpe:.2f}, but:
    - **Steady Eddie**: Low returns, low stress, small drawdowns
    - **Rollercoaster**: High returns, high stress, scary drawdowns
    - **Sneaky Losses**: Looks calm, then suddenly crashes
    - **Fat Tails**: Occasional extreme surprises (both ways)
    - **Crash & Recover**: Periodic painful periods
    
    **Would you really be indifferent between these?**
    """)

with st.expander("📊 When to Use Each Metric"):
    st.markdown("""
    ### Choosing the Right Risk Metric
    
    | Situation | Best Metric |
    |-----------|-------------|
    | Comparing funds with similar strategies | Sharpe Ratio (okay for rough comparison) |
    | Evaluating downside risk | **Sortino Ratio** |
    | Capital preservation is key | **Max Drawdown** |
    | Trading strategies with asymmetric returns | **Calmar Ratio** |
    | Tail risk is important | **CVaR (Expected Shortfall)** |
    | Complete picture needed | **Multiple metrics together** |
    
    ### The Golden Rule
    
    > **Never rely on a single metric!**
    > 
    > Always look at:
    > 1. Return distribution (histogram)
    > 2. Drawdown chart
    > 3. Multiple risk-adjusted ratios
    > 4. Correlation to your goals
    """)

with st.expander("🔬 Real-World Examples"):
    st.markdown("""
    ### Famous Sharpe Ratio Failures
    
    **1. Long-Term Capital Management (LTCM) - 1998**
    - High Sharpe ratio before collapse
    - Sharpe didn't capture leverage and tail risk
    - Lost $4.6 billion in weeks
    
    **2. Madoff's Ponzi Scheme**
    - "Perfect" Sharpe ratio (too good to be true!)
    - Consistent returns were fabricated
    - Lesson: Unrealistically high Sharpe = red flag
    
    **3. 2008 Financial Crisis**
    - Many "low volatility" strategies had high Sharpe
    - When correlations spiked, all crashed together
    - Sharpe doesn't capture correlation risk
    
    **4. Hedge Fund Smoothing**
    - Some funds report monthly, smoothing daily volatility
    - Artificially inflates Sharpe ratio
    - Always check data frequency
    
    ### Rules of Thumb
    
    | Sharpe Ratio | Interpretation |
    |--------------|----------------|
    | < 0 | Losing money vs risk-free |
    | 0 - 0.5 | Below average |
    | 0.5 - 1.0 | Average |
    | 1.0 - 2.0 | Good |
    | 2.0 - 3.0 | Very good (be skeptical!) |
    | > 3.0 | Too good to be true? |
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 <b>Remember:</b> The best portfolio isn't the one with the highest Sharpe ratio—it's the one you can stick with through the bad times.</p>
</div>
""", unsafe_allow_html=True)

