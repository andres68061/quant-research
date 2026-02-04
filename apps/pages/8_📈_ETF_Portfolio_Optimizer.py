"""
ETF Portfolio Optimizer with Efficient Frontier Analysis.

This page replicates the R Markdown ETF portfolio template functionality:
- Efficient Frontier calculation
- Tangency Portfolio (Max Sharpe Ratio)
- Capital Allocation Line (CAL)
- Portfolio rebalancing simulation
- Comprehensive performance metrics
"""

from src.data.banxico_api import get_current_cetes28_rate, get_cetes28_returns
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Page configuration
st.set_page_config(
    page_title="ETF Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

st.title("📈 ETF Portfolio Optimizer")
st.markdown("""
Build optimal portfolios using Modern Portfolio Theory with:
- **Efficient Frontier** - Risk/return trade-offs
- **Tangency Portfolio** - Maximum Sharpe ratio
- **Capital Allocation Line** - Lending & borrowing
- **Rebalancing Simulation** - Annual rebalancing
- **Performance Metrics** - Sharpe, Alpha, Beta, Jensen, Treynor
""")

# ============================================================================
# LOAD DATA
# ============================================================================


@st.cache_data
def load_price_data():
    """Load prices for ETF portfolio analysis."""
    prices_path = ROOT / "data" / "factors" / "prices.parquet"

    if not prices_path.exists():
        st.error("❌ Prices file not found. Run backfill_all.py first.")
        st.stop()

    df = pd.read_parquet(prices_path)
    return df


df_prices = load_price_data()

# Default ETF list from R template
DEFAULT_ETFS = [
    'VOO',    # Vanguard S&P 500
    'SOXX',   # Semiconductors
    'ITA',    # Aerospace & Defense
    'DTEC',   # Digital Transformation
    'IXJ',    # Healthcare
    'IYK',    # Consumer Goods
    'AIRR',   # Airlines
    'UGA',    # US Gasoline
    'MLPX',   # Energy Infrastructure
    'GREK',   # Greece
    'ARGT',   # Argentina
    'GLD',    # Gold
]

# Benchmark options
BENCHMARK_ETFS = ['BIL', 'ACWI', 'VOO', 'VTI', '^GSPC']

# ============================================================================
# SIDEBAR: CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Portfolio Configuration")

# Asset selection
st.sidebar.markdown("### 📊 Select Assets")

available_etfs = [
    col for col in df_prices.columns if col in DEFAULT_ETFS or col in BENCHMARK_ETFS or not col.startswith('^')]
available_etfs = sorted(set(available_etfs))

selected_assets = st.sidebar.multiselect(
    "Portfolio Assets",
    available_etfs,
    default=[a for a in DEFAULT_ETFS if a in available_etfs][:8],
    help="Select ETFs for portfolio optimization"
)

if len(selected_assets) < 2:
    st.warning("⚠️ Please select at least 2 assets")
    st.stop()

# Date range
st.sidebar.markdown("### 📅 Date Range")

min_date = df_prices.index.min().date()
max_date = df_prices.index.max().date()

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.Timestamp('2019-08-30').date(),
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# Risk-free rate
st.sidebar.markdown("### 💰 Risk-Free Rate")

# Try to get current CETES 28 rate
if st.sidebar.button("🇲🇽 Fetch CETES 28 Rate", help="Get current rate from Banxico API"):
    try:
        with st.spinner("Fetching from Banxico..."):
            rate, date = get_current_cetes28_rate()
            st.sidebar.success(
                f"✅ CETES 28: {rate*100:.2f}% (as of {date.date()})")
            # Store in session state
            st.session_state['cetes_rate'] = rate * 100
            st.session_state['cetes_date'] = date
    except Exception as e:
        st.sidebar.error(f"❌ Failed: {str(e)}")

# Show cached rate if available
if 'cetes_rate' in st.session_state:
    st.sidebar.caption(
        f"📊 Last fetched: {st.session_state['cetes_rate']:.2f}% on {st.session_state['cetes_date'].date()}")
    default_rate = st.session_state['cetes_rate']
else:
    default_rate = 8.15

risk_free_annual = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=default_rate,
    step=0.1,
    help="Annualized risk-free rate (e.g., T-bills, CETES 28)"
) / 100

risk_free_daily = (1 + risk_free_annual) ** (1/252) - 1

# Borrowing rate
borrowing_annual = st.sidebar.number_input(
    "Borrowing Rate (%)",
    min_value=risk_free_annual * 100,
    max_value=30.0,
    value=(risk_free_annual * 100) + 3.0,
    step=0.1,
    help="Annual borrowing rate (typically risk-free + premium)"
) / 100

# Benchmark selection
st.sidebar.markdown("### 📊 Benchmark")

benchmark_assets = st.sidebar.multiselect(
    "Benchmark Assets",
    [a for a in BENCHMARK_ETFS if a in df_prices.columns],
    default=['BIL', 'ACWI'] if all(a in df_prices.columns for a in [
                                   'BIL', 'ACWI']) else ['^GSPC'],
    help="Assets for benchmark portfolio"
)

if benchmark_assets:
    benchmark_weights_input = {}
    st.sidebar.markdown("**Benchmark Weights:**")

    remaining = 100.0
    for i, asset in enumerate(benchmark_assets[:-1]):
        weight = st.sidebar.number_input(
            f"{asset} (%)",
            min_value=0.0,
            max_value=100.0,
            value=100.0 / len(benchmark_assets),
            step=1.0,
            key=f"bench_weight_{asset}"
        )
        benchmark_weights_input[asset] = weight / 100
        remaining -= weight

    # Last asset gets remaining weight
    benchmark_weights_input[benchmark_assets[-1]] = max(0, remaining) / 100
    st.sidebar.caption(f"{benchmark_assets[-1]}: {max(0, remaining):.1f}%")

    benchmark_weights = np.array(
        [benchmark_weights_input[a] for a in benchmark_assets])
else:
    benchmark_weights = None

# Rebalancing frequency
st.sidebar.markdown("### 🔄 Rebalancing")

rebalance_freq = st.sidebar.selectbox(
    "Frequency",
    ["Annual", "Quarterly", "Monthly"],
    index=0,
    help="How often to rebalance the portfolio"
)

# CETES 28 option
st.sidebar.markdown("### 🇲🇽 CETES 28 Integration")

use_cetes_returns = st.sidebar.checkbox(
    "Use Actual CETES 28 Returns",
    value=False,
    help="Use historical CETES 28 returns instead of constant risk-free rate for simulation"
)

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Convert dates to timezone-aware to match DataFrame index
if df_prices.index.tz is not None:
    # Make dates timezone-aware
    start_date_tz = pd.Timestamp(start_date).tz_localize(df_prices.index.tz)
    end_date_tz = pd.Timestamp(end_date).tz_localize(df_prices.index.tz)
else:
    start_date_tz = start_date
    end_date_tz = end_date

# Filter data by date range
df_prices_filtered = df_prices.loc[start_date_tz:end_date_tz, selected_assets].copy()

# Remove assets with insufficient data
min_data_points = 252  # At least 1 year
valid_assets = []
for asset in selected_assets:
    if df_prices_filtered[asset].notna().sum() >= min_data_points:
        valid_assets.append(asset)

if len(valid_assets) < 2:
    st.error(
        f"❌ Not enough data. Need at least 2 assets with {min_data_points}+ data points.")
    st.stop()

df_prices_filtered = df_prices_filtered[valid_assets].dropna()

if len(df_prices_filtered) < min_data_points:
    st.error(
        f"❌ Insufficient overlapping data. Need at least {min_data_points} days.")
    st.stop()

# Calculate returns
returns = df_prices_filtered.pct_change().dropna()

# Fetch CETES 28 returns if requested
cetes28_returns = None
if use_cetes_returns:
    try:
        with st.spinner("📥 Fetching CETES 28 returns from Banxico..."):
            cetes28_returns = get_cetes28_returns(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            # Align with portfolio dates
            cetes28_returns = cetes28_returns.reindex(
                returns.index, method='ffill')
            st.success(
                f"✅ Loaded {len(cetes28_returns)} days of CETES 28 returns")
    except Exception as e:
        st.error(f"❌ Failed to fetch CETES 28: {e}")
        st.info("Falling back to constant risk-free rate")
        use_cetes_returns = False

# Calculate statistics
mean_returns = returns.mean() * 252  # Annualized
cov_matrix = returns.cov() * 252     # Annualized
std_devs = returns.std() * np.sqrt(252)  # Annualized

# ============================================================================
# EFFICIENT FRONTIER CALCULATION
# ============================================================================


def portfolio_stats(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility."""
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Negative Sharpe ratio for minimization."""
    p_return, p_std = portfolio_stats(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std


def portfolio_volatility(weights, mean_returns, cov_matrix):
    """Portfolio volatility for minimization."""
    return portfolio_stats(weights, mean_returns, cov_matrix)[1]


def calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_portfolios=50):
    """Calculate efficient frontier portfolios."""
    n_assets = len(mean_returns)

    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Find minimum variance portfolio
    init_guess = np.array([1/n_assets] * n_assets)

    min_var_result = minimize(
        portfolio_volatility,
        init_guess,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    min_var_return, min_var_std = portfolio_stats(
        min_var_result.x, mean_returns, cov_matrix
    )

    # Find maximum return
    max_return = mean_returns.max()

    # Calculate efficient frontier
    target_returns = np.linspace(min_var_return, max_return, num_portfolios)

    efficient_portfolios = []

    for target_return in target_returns:
        constraints_with_return = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(
                x, mean_returns) - target_return}
        ]

        result = minimize(
            portfolio_volatility,
            init_guess,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_with_return,
            options={'maxiter': 1000}
        )

        if result.success:
            p_return, p_std = portfolio_stats(
                result.x, mean_returns, cov_matrix)
            sharpe = (p_return - risk_free_rate) / p_std

            efficient_portfolios.append({
                'weights': result.x,
                'return': p_return,
                'volatility': p_std,
                'sharpe': sharpe
            })

    return efficient_portfolios, (min_var_return, min_var_std)


def find_tangency_portfolio(mean_returns, cov_matrix, risk_free_rate):
    """Find tangency portfolio (max Sharpe ratio)."""
    n_assets = len(mean_returns)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = np.array([1/n_assets] * n_assets)

    result = minimize(
        neg_sharpe_ratio,
        init_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


# Calculate efficient frontier
with st.spinner("🔄 Calculating Efficient Frontier..."):
    efficient_portfolios, min_var_point = calculate_efficient_frontier(
        mean_returns.values,
        cov_matrix.values,
        risk_free_annual
    )

    # Find tangency portfolio
    tangency_weights = find_tangency_portfolio(
        mean_returns.values,
        cov_matrix.values,
        risk_free_annual
    )

    tangency_return, tangency_std = portfolio_stats(
        tangency_weights,
        mean_returns.values,
        cov_matrix.values
    )

    tangency_sharpe = (tangency_return - risk_free_annual) / tangency_std

# ============================================================================
# DISPLAY: EFFICIENT FRONTIER
# ============================================================================

st.markdown("---")
st.header("📊 Efficient Frontier Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Tangency Return",
        f"{tangency_return*100:.2f}%",
        help="Expected annual return of tangency portfolio"
    )

with col2:
    st.metric(
        "Tangency Volatility",
        f"{tangency_std*100:.2f}%",
        help="Annual volatility of tangency portfolio"
    )

with col3:
    st.metric(
        "Sharpe Ratio",
        f"{tangency_sharpe:.3f}",
        help="Risk-adjusted return (higher is better)"
    )

with col4:
    st.metric(
        "Risk-Free Rate",
        f"{risk_free_annual*100:.2f}%",
        help="Annual risk-free rate"
    )

# Plot Efficient Frontier
fig_ef = go.Figure()

# Extract data for plotting
ef_returns = [p['return'] * 100 for p in efficient_portfolios]
ef_vols = [p['volatility'] * 100 for p in efficient_portfolios]

# Efficient Frontier
fig_ef.add_trace(go.Scatter(
    x=ef_vols,
    y=ef_returns,
    mode='lines',
    name='Efficient Frontier',
    line=dict(color='blue', width=3),
    hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
))

# Tangency Portfolio
fig_ef.add_trace(go.Scatter(
    x=[tangency_std * 100],
    y=[tangency_return * 100],
    mode='markers',
    name='Tangency Portfolio',
    marker=dict(size=15, color='red', symbol='star'),
    hovertemplate='Tangency<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>'
))

# Individual Assets
fig_ef.add_trace(go.Scatter(
    x=std_devs.values * 100,
    y=mean_returns.values * 100,
    mode='markers+text',
    name='Individual Assets',
    marker=dict(size=10, color='green'),
    text=valid_assets,
    textposition='top center',
    hovertemplate='%{text}<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>'
))

# Capital Allocation Line (CAL)
max_vol = max(ef_vols) * 1.3
cal_vols = np.linspace(0, max_vol, 100)

# Lending CAL (up to tangency)
lending_returns = risk_free_annual * 100 + \
    (cal_vols / (tangency_std * 100)) * \
    (tangency_return - risk_free_annual) * 100

# Borrowing CAL (beyond tangency)
borrowing_slope = (tangency_return - borrowing_annual) / tangency_std
borrowing_returns = tangency_return * 100 + \
    borrowing_slope * (cal_vols - tangency_std * 100) * 100

# Combine CAL
cal_returns = np.where(
    cal_vols <= tangency_std * 100,
    lending_returns,
    borrowing_returns
)

fig_ef.add_trace(go.Scatter(
    x=cal_vols,
    y=cal_returns,
    mode='lines',
    name='CAL (Lending/Borrowing)',
    line=dict(color='green', width=2, dash='dash'),
    hovertemplate='CAL<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>'
))

fig_ef.update_layout(
    title="Efficient Frontier with Capital Allocation Line",
    xaxis_title="Volatility (Annual %)",
    yaxis_title="Expected Return (Annual %)",
    height=600,
    hovermode='closest',
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)

st.plotly_chart(fig_ef, use_container_width=True)

# ============================================================================
# DISPLAY: TANGENCY PORTFOLIO WEIGHTS
# ============================================================================

st.markdown("### 🎯 Tangency Portfolio Weights")

weights_df = pd.DataFrame({
    'Asset': valid_assets,
    'Weight (%)': tangency_weights * 100
}).sort_values('Weight (%)', ascending=False)

col1, col2 = st.columns([1, 1])

with col1:
    st.dataframe(
        weights_df.style.format({'Weight (%)': '{:.2f}%'}),
        use_container_width=True,
        hide_index=True
    )

with col2:
    # Pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=weights_df['Asset'],
        values=weights_df['Weight (%)'],
        hole=0.3,
        hovertemplate='%{label}<br>%{value:.2f}%<extra></extra>'
    )])

    fig_pie.update_layout(
        title="Portfolio Allocation",
        height=400
    )

    st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================================
# PORTFOLIO SIMULATION WITH REBALANCING
# ============================================================================

st.markdown("---")
st.header("📈 Portfolio Performance Simulation")


def simulate_portfolio_with_rebalancing(returns_df, weights, rebalance_freq='Annual'):
    """Simulate portfolio with periodic rebalancing."""
    dates = returns_df.index
    n_days = len(dates)

    portfolio_value = np.zeros(n_days)
    portfolio_value[0] = 1.0  # Start with $1

    current_weights = weights.copy()

    # Determine rebalancing dates
    if rebalance_freq == 'Annual':
        years = pd.Series(dates).dt.year.unique()
        rebal_dates = [dates[0]]
        for year in years[1:]:
            year_dates = dates[pd.Series(dates).dt.year == year]
            if len(year_dates) > 0:
                rebal_dates.append(year_dates[0])
    elif rebalance_freq == 'Quarterly':
        rebal_dates = dates[dates.is_quarter_start]
    else:  # Monthly
        rebal_dates = dates[dates.is_month_start]

    rebal_dates = set(rebal_dates)

    for t in range(n_days - 1):
        # Calculate portfolio return
        asset_returns = returns_df.iloc[t].values
        portfolio_return = np.dot(current_weights, asset_returns)

        # Update portfolio value
        portfolio_value[t + 1] = portfolio_value[t] * (1 + portfolio_return)

        # Rebalance if needed
        if dates[t] in rebal_dates:
            current_weights = weights.copy()
        else:
            # Drift weights
            current_weights = current_weights * (1 + asset_returns)
            current_weights = current_weights / current_weights.sum()

    return pd.Series(portfolio_value, index=dates)


# Simulate tangency portfolio
tangency_value = simulate_portfolio_with_rebalancing(
    returns[valid_assets],
    tangency_weights,
    rebalance_freq
)

# Simulate 50/50 portfolio (50% risk-free, 50% tangency)
optimal_50_weights = tangency_weights * 0.5
optimal_50_returns = returns[valid_assets].copy()

# Use CETES 28 returns if available, otherwise constant risk-free rate
if use_cetes_returns and cetes28_returns is not None:
    optimal_50_returns['RiskFree'] = cetes28_returns
else:
    optimal_50_returns['RiskFree'] = risk_free_daily

# Add risk-free asset
optimal_50_weights_full = np.append([0.5], optimal_50_weights)

optimal_50_value = simulate_portfolio_with_rebalancing(
    optimal_50_returns,
    optimal_50_weights_full,
    rebalance_freq
)

# Simulate benchmark if available
if benchmark_assets and benchmark_weights is not None:
    benchmark_returns = returns[[
        a for a in benchmark_assets if a in returns.columns]]

    if len(benchmark_returns.columns) == len(benchmark_assets):
        benchmark_value = simulate_portfolio_with_rebalancing(
            benchmark_returns,
            benchmark_weights,
            rebalance_freq
        )
    else:
        benchmark_value = None
else:
    benchmark_value = None

# Plot cumulative growth
fig_growth = go.Figure()

fig_growth.add_trace(go.Scatter(
    x=tangency_value.index,
    y=tangency_value.values,
    mode='lines',
    name='Tangency Portfolio',
    line=dict(color='blue', width=2)
))

fig_growth.add_trace(go.Scatter(
    x=optimal_50_value.index,
    y=optimal_50_value.values,
    mode='lines',
    name='50/50 Portfolio',
    line=dict(color='red', width=2)
))

if benchmark_value is not None:
    fig_growth.add_trace(go.Scatter(
        x=benchmark_value.index,
        y=benchmark_value.values,
        mode='lines',
        name='Benchmark',
        line=dict(color='purple', width=2)
    ))

# Add risk-free growth
if use_cetes_returns and cetes28_returns is not None:
    rf_growth = (1 + cetes28_returns).cumprod()
    rf_label = 'CETES 28'
else:
    rf_growth = (1 + risk_free_daily) ** np.arange(len(returns))
    rf_label = 'Risk-Free'

fig_growth.add_trace(go.Scatter(
    x=returns.index,
    y=rf_growth,
    mode='lines',
    name=rf_label,
    line=dict(color='green', width=2, dash='dash')
))

fig_growth.update_layout(
    title=f"Cumulative Growth with {rebalance_freq} Rebalancing",
    xaxis_title="Date",
    yaxis_title="Portfolio Value (Starting at $1)",
    height=600,
    hovermode='x unified'
)

st.plotly_chart(fig_growth, use_container_width=True)

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

st.markdown("---")
st.header("📊 Performance Metrics")


def calculate_performance_metrics(portfolio_returns, benchmark_returns, risk_free_returns):
    """Calculate comprehensive performance metrics."""
    # Annualize
    n_days = len(portfolio_returns)
    n_years = n_days / 252

    # Returns
    mean_return = portfolio_returns.mean() * 252

    # Volatility
    volatility = portfolio_returns.std() * np.sqrt(252)

    # Sharpe Ratio
    excess_returns = portfolio_returns - risk_free_returns
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    # Max Drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Beta and Alpha
    if benchmark_returns is not None:
        cov_matrix = np.cov(portfolio_returns, benchmark_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        benchmark_mean = benchmark_returns.mean() * 252
        alpha = mean_return - (risk_free_annual + beta *
                               (benchmark_mean - risk_free_annual))

        # Jensen's Alpha
        excess_port = portfolio_returns - risk_free_returns
        excess_bench = benchmark_returns - risk_free_returns

        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(excess_bench, excess_port)
        jensen_alpha = intercept * 252

        # Information Ratio
        excess_over_bench = portfolio_returns - benchmark_returns
        tracking_error = excess_over_bench.std() * np.sqrt(252)
        info_ratio = (excess_over_bench.mean() * 252) / \
            tracking_error if tracking_error > 0 else 0

        # Treynor Ratio
        treynor = (mean_return - risk_free_annual) / beta if beta != 0 else 0
    else:
        beta = np.nan
        alpha = np.nan
        jensen_alpha = np.nan
        info_ratio = np.nan
        treynor = np.nan

    # Holding period return
    total_return = cumulative.iloc[-1] - 1
    avg_annual_return = (1 + total_return) ** (1 / n_years) - 1

    return {
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Jensen Alpha': jensen_alpha,
        'Alpha': alpha,
        'Beta': beta,
        'Information Ratio': info_ratio,
        'Treynor Ratio': treynor,
        'Holding Period Return': total_return,
        'Avg Annual Return': avg_annual_return,
        'Volatility': volatility,
        'Mean Return': mean_return
    }


# Calculate returns for each portfolio
tangency_returns = tangency_value.pct_change().dropna()
optimal_50_returns_series = optimal_50_value.pct_change().dropna()

if benchmark_value is not None:
    benchmark_returns_series = benchmark_value.pct_change().dropna()
else:
    benchmark_returns_series = None

# Use CETES 28 returns if available, otherwise constant risk-free rate
if use_cetes_returns and cetes28_returns is not None:
    risk_free_returns_series = cetes28_returns.reindex(
        tangency_returns.index, method='ffill')
else:
    risk_free_returns_series = pd.Series(
        risk_free_daily, index=tangency_returns.index)

# Calculate metrics
tangency_metrics = calculate_performance_metrics(
    tangency_returns,
    benchmark_returns_series,
    risk_free_returns_series
)

optimal_50_metrics = calculate_performance_metrics(
    optimal_50_returns_series,
    benchmark_returns_series,
    risk_free_returns_series
)

if benchmark_value is not None:
    benchmark_metrics = calculate_performance_metrics(
        benchmark_returns_series,
        None,  # No benchmark for benchmark
        risk_free_returns_series
    )
else:
    benchmark_metrics = None

# Create metrics table
metrics_data = {
    'Metric': list(tangency_metrics.keys()),
    'Tangency': list(tangency_metrics.values()),
    '50/50 Portfolio': list(optimal_50_metrics.values()),
}

if benchmark_metrics:
    metrics_data['Benchmark'] = list(benchmark_metrics.values())

metrics_df = pd.DataFrame(metrics_data)

# Format the table


def format_metric(val):
    if pd.isna(val):
        return 'N/A'
    elif abs(val) < 0.01:
        return f'{val:.4f}'
    elif abs(val) < 1:
        return f'{val:.3f}'
    else:
        return f'{val:.2f}'


st.dataframe(
    metrics_df.style.format({
        col: format_metric for col in metrics_df.columns if col != 'Metric'
    }),
    use_container_width=True,
    hide_index=True
)

# ============================================================================
# DOWNLOAD SECTION
# ============================================================================

st.markdown("---")
st.header("📥 Download Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Download weights
    weights_csv = weights_df.to_csv(index=False)
    st.download_button(
        label="📊 Download Weights (CSV)",
        data=weights_csv,
        file_name=f"tangency_weights_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    # Download metrics
    metrics_csv = metrics_df.to_csv(index=False)
    st.download_button(
        label="📈 Download Metrics (CSV)",
        data=metrics_csv,
        file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col3:
    # Download portfolio values
    portfolio_values = pd.DataFrame({
        'Date': tangency_value.index,
        'Tangency': tangency_value.values,
        '50/50': optimal_50_value.values,
        'Benchmark': benchmark_value.values if benchmark_value is not None else np.nan
    })

    portfolio_csv = portfolio_values.to_csv(index=False)
    st.download_button(
        label="💼 Download Portfolio Values (CSV)",
        data=portfolio_csv,
        file_name=f"portfolio_values_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("""
💡 **Methodology:**
- **Efficient Frontier:** Calculated using mean-variance optimization (Markowitz)
- **Tangency Portfolio:** Maximum Sharpe ratio portfolio
- **CAL:** Capital Allocation Line with lending (risk-free) and borrowing rates
- **Rebalancing:** Periodic rebalancing to target weights
- **Metrics:** Industry-standard risk-adjusted performance measures
""")
