#!/usr/bin/env python3
"""
Sortino Momentum Analysis - Advanced statistical analysis of Sortino ratio persistence.

This page analyzes whether improving Sortino ratios tend to continue improving,
and provides regime indicators for portfolio monitoring.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.utils.portfolio import calculate_rolling_metrics

# Page configuration
st.set_page_config(
    page_title="Sortino Momentum Analysis",
    page_icon="💜",
    layout="wide",
)

st.title("💜 Sortino Momentum Analysis")
st.markdown("""
**Research Question:** If the rolling Sortino ratio has been improving, 
will it continue to improve? And with what probability?

This page uses three statistical methods to answer this question and identify regime changes.
""")

st.markdown("---")


@st.cache_data
def load_data():
    """Load prices and factors data."""
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
    
    if not factors_path.exists() or not prices_path.exists():
        st.error("❌ Required data files not found.")
        st.stop()
    
    df_factors = pd.read_parquet(factors_path)
    df_prices = pd.read_parquet(prices_path)
    return df_factors, df_prices


def calculate_sortino_slopes(rolling_sortino, X_days):
    """Calculate Sortino slopes over X days."""
    slope = rolling_sortino.diff(X_days) / X_days
    return slope


def analyze_momentum_grid_search(returns, sortino_window=252, min_signals=10):
    """
    Method 1: Grid search to find optimal X, K, and calculate Z (hit rate).
    
    Args:
        returns: Daily returns series
        sortino_window: Window for rolling Sortino calculation
        min_signals: Minimum number of signals required for valid result
    
    Returns:
        DataFrame with results for each (X, K) combination
    """
    # Calculate rolling Sortino
    rolling_metrics = calculate_rolling_metrics(returns, window=sortino_window)
    rolling_sortino = rolling_metrics['sortino_ratio'].dropna()
    
    # Grid search parameters
    lookback_windows = [5, 10, 15, 20, 30, 45, 60, 90]  # X values
    forecast_horizons = [5, 10, 15, 20, 30]              # K values
    baseline_window = 30  # Reference period for comparison
    
    results = []
    
    for X in lookback_windows:
        for K in forecast_horizons:
            # Calculate recent slope (last X days)
            recent_slope = calculate_sortino_slopes(rolling_sortino, X)
            
            # Calculate baseline slope (30 days before the X-day period)
            baseline_slope = calculate_sortino_slopes(rolling_sortino.shift(X), baseline_window)
            
            # Identify "strong momentum" periods
            strong_momentum = (recent_slope > baseline_slope) & recent_slope.notna() & baseline_slope.notna()
            
            # Check if momentum continued for next K days
            future_slope = calculate_sortino_slopes(rolling_sortino.shift(-K), K)
            continued = (future_slope > 0) & future_slope.notna()
            
            # Calculate hit rate Z
            valid_indices = strong_momentum[strong_momentum].index
            
            if len(valid_indices) >= min_signals:
                # Get continuation outcomes for signals
                outcomes = continued.loc[valid_indices]
                hits = outcomes.sum()
                total = len(outcomes)
                hit_rate = (hits / total * 100) if total > 0 else np.nan
                
                # Calculate confidence interval (binomial proportion)
                if total > 0:
                    se = np.sqrt(hit_rate/100 * (1 - hit_rate/100) / total)
                    ci_lower = max(0, hit_rate - 1.96 * se * 100)
                    ci_upper = min(100, hit_rate + 1.96 * se * 100)
                else:
                    ci_lower = ci_upper = np.nan
                
                results.append({
                    'X (lookback)': X,
                    'K (forecast)': K,
                    'Z (hit_rate)': hit_rate,
                    'CI_lower': ci_lower,
                    'CI_upper': ci_upper,
                    'Total_signals': total,
                    'Successful': hits,
                    'Failed': total - hits
                })
    
    df_results = pd.DataFrame(results)
    return df_results.sort_values('Z (hit_rate)', ascending=False)


def test_statistical_significance(returns, X, K, sortino_window=252, n_bootstraps=500):
    """
    Method 2: Bootstrap test for statistical significance.
    
    Tests if observed hit rate is significantly different from random chance.
    """
    # Calculate actual hit rate
    rolling_metrics = calculate_rolling_metrics(returns, window=sortino_window)
    rolling_sortino = rolling_metrics['sortino_ratio'].dropna()
    
    recent_slope = calculate_sortino_slopes(rolling_sortino, X)
    baseline_slope = calculate_sortino_slopes(rolling_sortino.shift(X), 30)
    strong_momentum = (recent_slope > baseline_slope) & recent_slope.notna() & baseline_slope.notna()
    future_slope = calculate_sortino_slopes(rolling_sortino.shift(-K), K)
    continued = (future_slope > 0) & future_slope.notna()
    
    valid_indices = strong_momentum[strong_momentum].index
    
    if len(valid_indices) < 10:
        return {
            'actual_hit_rate': np.nan,
            'random_mean': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n_signals': len(valid_indices)
        }
    
    outcomes = continued.loc[valid_indices]
    actual_hit_rate = outcomes.mean() * 100
    
    # Bootstrap: shuffle future outcomes
    bootstrap_hit_rates = []
    np.random.seed(42)
    
    for _ in range(n_bootstraps):
        shuffled_outcomes = outcomes.sample(frac=1, replace=True)
        bootstrap_hit_rates.append(shuffled_outcomes.mean() * 100)
    
    # Calculate p-value (two-tailed test)
    random_mean = np.mean(bootstrap_hit_rates)
    p_value = np.mean(np.abs(bootstrap_hit_rates - random_mean) >= np.abs(actual_hit_rate - random_mean))
    
    return {
        'actual_hit_rate': actual_hit_rate,
        'random_mean': random_mean,
        'random_std': np.std(bootstrap_hit_rates),
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_signals': len(valid_indices),
        'bootstrap_dist': bootstrap_hit_rates
    }


def prepare_ml_features(returns, sortino_window=252, forecast_horizon=10):
    """
    Method 3: Prepare features for machine learning prediction.
    
    Features:
    - Sortino level
    - Slopes at multiple timeframes (5, 10, 20, 30 days)
    - Sortino vs baseline
    - Recent volatility
    - Sharpe ratio
    
    Target:
    - Will Sortino rise in next K days?
    """
    rolling = calculate_rolling_metrics(returns, window=sortino_window)
    
    features = pd.DataFrame({
        'sortino': rolling['sortino_ratio'],
        'sharpe': rolling['sharpe_ratio'],
        'volatility': rolling['annualized_volatility'],
        'slope_5d': calculate_sortino_slopes(rolling['sortino_ratio'], 5),
        'slope_10d': calculate_sortino_slopes(rolling['sortino_ratio'], 10),
        'slope_20d': calculate_sortino_slopes(rolling['sortino_ratio'], 20),
        'slope_30d': calculate_sortino_slopes(rolling['sortino_ratio'], 30),
    }).dropna()
    
    # Calculate vs baseline
    features['vs_baseline'] = (
        calculate_sortino_slopes(rolling['sortino_ratio'], 20) - 
        calculate_sortino_slopes(rolling['sortino_ratio'].shift(20), 30)
    )
    
    # Create target
    K = forecast_horizon
    future_slope = calculate_sortino_slopes(rolling['sortino_ratio'].shift(-K), K)
    features['target'] = (future_slope > 0).astype(int)
    
    # Drop NaN
    features = features.dropna()
    
    return features


def analyze_ml_prediction(returns, sortino_window=252, forecast_horizon=10):
    """
    Method 3: Machine learning analysis using logistic regression.
    
    Uses time series cross-validation to avoid look-ahead bias.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        return {
            'error': 'sklearn not installed',
            'message': 'Install with: pip install scikit-learn'
        }
    
    # Prepare features
    features = prepare_ml_features(returns, sortino_window, forecast_horizon)
    
    if len(features) < 100:
        return {
            'error': 'insufficient_data',
            'message': f'Only {len(features)} samples available, need at least 100'
        }
    
    X = features.drop('target', axis=1)
    y = features['target']
    
    # Time series cross-validation (5 splits)
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    all_predictions = []
    all_actuals = []
    feature_importances = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        
        # Store feature importance (coefficients)
        feature_importances.append(np.abs(clf.coef_[0]))
    
    # Calculate mean feature importance
    mean_importance = np.mean(feature_importances, axis=0)
    feature_importance_dict = dict(zip(X.columns, mean_importance))
    feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'accuracies': accuracies,
        'feature_importance': feature_importance_dict,
        'n_samples': len(features),
        'positive_class_pct': y.mean() * 100,
        'all_predictions': all_predictions,
        'all_actuals': all_actuals
    }


def get_current_regime(returns, X, K, sortino_window=252):
    """Determine current Sortino momentum regime."""
    rolling_metrics = calculate_rolling_metrics(returns, window=sortino_window)
    rolling_sortino = rolling_metrics['sortino_ratio'].dropna()
    
    if len(rolling_sortino) < X + 30:
        return None
    
    # Calculate current slopes
    recent_slope = calculate_sortino_slopes(rolling_sortino, X).iloc[-1]
    baseline_slope = calculate_sortino_slopes(rolling_sortino.shift(X), 30).iloc[-1]
    
    current_sortino = rolling_sortino.iloc[-1]
    
    # Determine regime
    if pd.notna(recent_slope) and pd.notna(baseline_slope):
        is_strong_momentum = recent_slope > baseline_slope
        
        return {
            'current_sortino': current_sortino,
            'recent_slope': recent_slope,
            'baseline_slope': baseline_slope,
            'strong_momentum': is_strong_momentum,
            'slope_ratio': recent_slope / baseline_slope if baseline_slope != 0 else np.nan
        }
    
    return None


# ============================================================================
# MAIN APP
# ============================================================================

# Load data
df_factors, df_prices = load_data()

# Sidebar configuration
st.sidebar.header("🔧 Analysis Configuration")

# Date range selection
min_date = df_prices.index.min().date()
max_date = df_prices.index.max().date()

date_range = st.sidebar.date_input(
    "Analysis Period",
    value=(max_date - pd.Timedelta(days=365 * 5), max_date),
    min_value=min_date,
    max_value=max_date,
    help="Select time period for analysis"
)

if len(date_range) != 2:
    st.warning("⚠️ Please select both start and end dates")
    st.stop()

start_date, end_date = date_range

# Filter data
df_prices_filtered = df_prices[
    (df_prices.index.date >= start_date) & (df_prices.index.date <= end_date)
]

# Symbol selection
available_symbols = sorted(df_prices_filtered.columns.tolist())
selected_symbol = st.sidebar.selectbox(
    "Select Stock/Index",
    options=available_symbols,
    index=available_symbols.index('^GSPC') if '^GSPC' in available_symbols else 0,
    help="Choose a symbol to analyze"
)

# Calculate returns
returns = df_prices_filtered[selected_symbol].pct_change().dropna()

if len(returns) < 500:
    st.error(f"❌ Insufficient data: only {len(returns)} days available. Need at least 500.")
    st.stop()

st.sidebar.markdown("---")

# Analysis parameters
sortino_window = st.sidebar.slider(
    "Sortino Window (days)",
    min_value=63,
    max_value=504,
    value=252,
    step=21,
    help="Rolling window for Sortino calculation (252 = 1 year)"
)

st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

# ============================================================================
# DISPLAY
# ============================================================================

if not run_analysis:
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis**")
    st.stop()

with st.spinner("Running comprehensive momentum analysis..."):
    
    # Calculate rolling Sortino for visualization
    rolling_metrics = calculate_rolling_metrics(returns, window=sortino_window)
    rolling_sortino = rolling_metrics['sortino_ratio'].dropna()
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Symbol", selected_symbol)
    with col2:
        st.metric("Trading Days", len(returns))
    with col3:
        current_sortino = rolling_sortino.iloc[-1] if len(rolling_sortino) > 0 else np.nan
        st.metric("Current Sortino", f"{current_sortino:.2f}")
    with col4:
        mean_sortino = rolling_sortino.mean()
        st.metric("Avg Sortino", f"{mean_sortino:.2f}")
    
    st.markdown("---")
    
    # ========================================================================
    # METHOD 1: GRID SEARCH
    # ========================================================================
    
    st.header("📊 Method 1: Grid Search Optimization")
    st.markdown("""
    **Objective:** Find optimal lookback (X) and forecast (K) periods that maximize hit rate (Z).
    
    **Interpretation:**
    - **X (lookback)**: How many days to measure recent Sortino improvement
    - **K (forecast)**: How many days ahead to predict continued improvement  
    - **Z (hit rate)**: Probability that improvement continues
    """)
    
    with st.spinner("Running grid search..."):
        grid_results = analyze_momentum_grid_search(returns, sortino_window)
    
    if len(grid_results) == 0:
        st.warning("⚠️ No valid combinations found. Try a longer time period.")
    else:
        # Display top results
        st.markdown("### 🏆 Top 10 Combinations")
        
        display_cols = ['X (lookback)', 'K (forecast)', 'Z (hit_rate)', 'CI_lower', 'CI_upper', 'Total_signals', 'Successful', 'Failed']
        st.dataframe(
            grid_results[display_cols].head(10).style.format({
                'Z (hit_rate)': '{:.1f}%',
                'CI_lower': '{:.1f}%',
                'CI_upper': '{:.1f}%',
            }).background_gradient(subset=['Z (hit_rate)'], cmap='RdYlGn', vmin=45, vmax=70),
            use_container_width=True
        )
        
        # Best combination
        best = grid_results.iloc[0]
        X_best = int(best['X (lookback)'])
        K_best = int(best['K (forecast)'])
        Z_best = best['Z (hit_rate)']
        
        st.success(f"""
        **✨ Best Combination Found:**
        - **X = {X_best} days** (lookback period)
        - **K = {K_best} days** (forecast horizon)
        - **Z = {Z_best:.1f}%** (hit rate)
        - **95% CI: [{best['CI_lower']:.1f}%, {best['CI_upper']:.1f}%]**
        - **Signals: {int(best['Total_signals'])}** ({int(best['Successful'])} successful, {int(best['Failed'])} failed)
        
        **Plain English:** When Sortino rises faster than usual for {X_best} days, 
        it continues rising for the next {K_best} days approximately **{Z_best:.1f}% of the time**.
        """)
        
        # Heatmap
        st.markdown("### 🌡️ Hit Rate Heatmap")
        
        pivot = grid_results.pivot(index='K (forecast)', columns='X (lookback)', values='Z (hit_rate)')
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=50,
            text=pivot.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Hit Rate (%)"),
            hovertemplate='X=%{x} days<br>K=%{y} days<br>Z=%{z:.1f}%<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title="Hit Rate (Z%) by Lookback (X) and Forecast (K) Periods",
            xaxis_title="X - Lookback Period (days)",
            yaxis_title="K - Forecast Horizon (days)",
            height=500
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # METHOD 2: STATISTICAL SIGNIFICANCE
    # ========================================================================
    
    st.header("📈 Method 2: Statistical Significance Test")
    st.markdown("""
    **Objective:** Determine if the observed hit rate is statistically significant or just random luck.
    
    **Method:** Bootstrap resampling (500 iterations) to compare actual vs. random performance.
    """)
    
    if len(grid_results) > 0:
        with st.spinner("Running significance tests..."):
            sig_test = test_statistical_significance(returns, X_best, K_best, sortino_window)
        
        if sig_test['n_signals'] < 10:
            st.warning(f"⚠️ Only {sig_test['n_signals']} signals found. Need at least 10 for reliable testing.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Actual Hit Rate",
                    f"{sig_test['actual_hit_rate']:.1f}%",
                    delta=f"{sig_test['actual_hit_rate'] - sig_test['random_mean']:.1f}% vs random"
                )
            
            with col2:
                st.metric(
                    "Random Mean (Bootstrap)",
                    f"{sig_test['random_mean']:.1f}%",
                    delta=f"±{sig_test['random_std']:.1f}% std"
                )
            
            with col3:
                st.metric(
                    "P-Value",
                    f"{sig_test['p_value']:.4f}",
                    delta="Significant" if sig_test['significant'] else "Not Significant"
                )
            
            if sig_test['significant']:
                st.success(f"""
                ✅ **Result: STATISTICALLY SIGNIFICANT** (p < 0.05)
                
                The observed hit rate of {sig_test['actual_hit_rate']:.1f}% is significantly different 
                from random chance ({sig_test['random_mean']:.1f}%). This suggests genuine momentum persistence.
                """)
            else:
                st.warning(f"""
                ⚠️ **Result: NOT STATISTICALLY SIGNIFICANT** (p = {sig_test['p_value']:.4f})
                
                The observed hit rate of {sig_test['actual_hit_rate']:.1f}% could be due to random chance. 
                The pattern may not be reliable for prediction.
                """)
            
            # Bootstrap distribution
            fig_bootstrap = go.Figure()
            
            fig_bootstrap.add_trace(go.Histogram(
                x=sig_test['bootstrap_dist'],
                nbinsx=30,
                name='Bootstrap Distribution',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig_bootstrap.add_vline(
                x=sig_test['actual_hit_rate'],
                line_dash="solid",
                line_color="red",
                line_width=3,
                annotation_text=f"Actual: {sig_test['actual_hit_rate']:.1f}%",
                annotation_position="top right"
            )
            
            fig_bootstrap.add_vline(
                x=sig_test['random_mean'],
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Random: {sig_test['random_mean']:.1f}%",
                annotation_position="top left"
            )
            
            fig_bootstrap.update_layout(
                title="Bootstrap Distribution vs. Actual Hit Rate",
                xaxis_title="Hit Rate (%)",
                yaxis_title="Frequency",
                height=450,
                showlegend=False
            )
            
            st.plotly_chart(fig_bootstrap, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # METHOD 3: MACHINE LEARNING
    # ========================================================================
    
    st.header("🤖 Method 3: Machine Learning Prediction")
    st.markdown("""
    **Objective:** Use multiple features to predict Sortino momentum using logistic regression.
    
    **Features:** Sortino level, slopes at multiple timeframes, Sharpe ratio, volatility
    
    **Validation:** Time series cross-validation (5 splits) to prevent look-ahead bias
    """)
    
    with st.spinner("Training machine learning models..."):
        ml_results = analyze_ml_prediction(returns, sortino_window, forecast_horizon=K_best if len(grid_results) > 0 else 10)
    
    if 'error' in ml_results:
        st.error(f"❌ {ml_results['message']}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Accuracy", f"{ml_results['mean_accuracy']*100:.1f}%")
        with col2:
            st.metric("Std Accuracy", f"±{ml_results['std_accuracy']*100:.1f}%")
        with col3:
            st.metric("Training Samples", f"{ml_results['n_samples']}")
        with col4:
            st.metric("Positive Class", f"{ml_results['positive_class_pct']:.1f}%")
        
        # Interpretation
        accuracy = ml_results['mean_accuracy'] * 100
        
        if accuracy > 55:
            st.success(f"""
            ✅ **Model Performance: GOOD** ({accuracy:.1f}% accuracy)
            
            The ML model can predict Sortino momentum better than random (50%). 
            Multiple features contribute to predictive power.
            """)
        elif accuracy > 52:
            st.info(f"""
            ℹ️ **Model Performance: MODEST** ({accuracy:.1f}% accuracy)
            
            The ML model shows slight predictive ability but the edge is small. 
            Use as a supplementary indicator.
            """)
        else:
            st.warning(f"""
            ⚠️ **Model Performance: POOR** ({accuracy:.1f}% accuracy)
            
            The ML model performs close to random. Sortino momentum may not be predictable 
            using these features.
            """)
        
        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': list(ml_results['feature_importance'].keys()),
            'Importance': list(ml_results['feature_importance'].values())
        })
        
        fig_importance = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color='purple'
        ))
        
        fig_importance.update_layout(
            title="Feature Importance (Absolute Coefficient Values)",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Cross-validation scores
        st.markdown("### 📊 Cross-Validation Performance")
        
        fig_cv = go.Figure()
        
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(ml_results['accuracies']))],
            y=[acc * 100 for acc in ml_results['accuracies']],
            marker_color='purple',
            text=[f"{acc*100:.1f}%" for acc in ml_results['accuracies']],
            textposition='outside'
        ))
        
        fig_cv.add_hline(
            y=50,
            line_dash="dash",
            line_color="red",
            annotation_text="Random (50%)",
            annotation_position="right"
        )
        
        fig_cv.update_layout(
            title="Accuracy Across Cross-Validation Folds",
            xaxis_title="Fold",
            yaxis_title="Accuracy (%)",
            height=400,
            showlegend=False,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_cv, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # CURRENT REGIME INDICATOR
    # ========================================================================
    
    st.header("🎯 Current Regime Indicator")
    st.markdown("**What's happening RIGHT NOW with Sortino momentum?**")
    
    if len(grid_results) > 0:
        regime = get_current_regime(returns, X_best, K_best, sortino_window)
        
        if regime:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Sortino",
                    f"{regime['current_sortino']:.2f}"
                )
            
            with col2:
                st.metric(
                    f"Recent Slope ({X_best}d)",
                    f"{regime['recent_slope']:.4f}"
                )
            
            with col3:
                st.metric(
                    "Baseline Slope (30d)",
                    f"{regime['baseline_slope']:.4f}"
                )
            
            if regime['strong_momentum']:
                st.success(f"""
                ### 📈 STRONG POSITIVE MOMENTUM DETECTED
                
                **Status:** Sortino is rising faster than usual
                
                **Historical Evidence:** Based on {int(best['Total_signals'])} historical signals,
                when this pattern occurred, Sortino continued rising for the next {K_best} days 
                approximately **{Z_best:.1f}% of the time**.
                
                **Recommendation:** Strategy is currently in a favorable regime. Monitor closely.
                """)
            else:
                st.info(f"""
                ### 📊 NEUTRAL/NEGATIVE MOMENTUM
                
                **Status:** Sortino is NOT rising faster than usual
                
                **Context:** Current slope ({regime['recent_slope']:.4f}) is below baseline ({regime['baseline_slope']:.4f})
                
                **Recommendation:** Strategy may be entering a challenging period or consolidating. 
                Historical edge is less applicable in this regime.
                """)
            
            # Plot recent Sortino with regime
            fig_regime = go.Figure()
            
            recent_sortino = rolling_sortino.iloc[-252:]  # Last year
            
            fig_regime.add_trace(go.Scatter(
                x=recent_sortino.index,
                y=recent_sortino.values,
                mode='lines',
                name='Rolling Sortino',
                line=dict(color='purple', width=2)
            ))
            
            # Highlight current regime period
            highlight_start = len(recent_sortino) - X_best
            if highlight_start >= 0:
                highlight_sortino = recent_sortino.iloc[highlight_start:]
                fig_regime.add_trace(go.Scatter(
                    x=highlight_sortino.index,
                    y=highlight_sortino.values,
                    mode='lines',
                    name=f'Current Period ({X_best}d)',
                    line=dict(color='green' if regime['strong_momentum'] else 'orange', width=4)
                ))
            
            fig_regime.update_layout(
                title=f"Rolling Sortino - Last 252 Days (Highlighting Current {X_best}-Day Period)",
                xaxis_title="Date",
                yaxis_title="Sortino Ratio",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_regime, use_container_width=True)
        else:
            st.warning("⚠️ Unable to calculate current regime. Need more data.")
    
    st.markdown("---")
    
    # ========================================================================
    # SUMMARY & RECOMMENDATIONS
    # ========================================================================
    
    st.header("📋 Summary & Recommendations")
    
    if len(grid_results) > 0:
        st.markdown(f"""
        ### Key Findings for {selected_symbol}
        
        **1. Optimal Parameters:**
        - Lookback: **{X_best} days**
        - Forecast: **{K_best} days**
        - Hit Rate: **{Z_best:.1f}%** (95% CI: [{best['CI_lower']:.1f}%, {best['CI_upper']:.1f}%])
        
        **2. Statistical Validity:**
        - Significance: **{'YES (p < 0.05)' if sig_test.get('significant') else f'NO (p = {sig_test.get("p_value", np.nan):.4f})'}**
        - Interpretation: **{'Genuine pattern detected' if sig_test.get('significant') else 'May be random chance'}**
        
        **3. ML Performance:**
        - Accuracy: **{ml_results.get('mean_accuracy', 0)*100:.1f}%**
        - Assessment: **{'Predictive power exists' if ml_results.get('mean_accuracy', 0) > 0.55 else 'Limited predictive power'}**
        
        **4. Practical Application:**
        """)
        
        if Z_best > 60 and sig_test.get('significant', False):
            st.success("""
            ✅ **STRONG SIGNAL** - Consider using as a regime indicator:
            - High hit rate (>60%)
            - Statistically significant
            - Can inform position sizing or risk management
            - Use as confidence boost, not primary entry/exit signal
            """)
        elif Z_best > 55:
            st.info("""
            ℹ️ **MODEST SIGNAL** - Use with caution:
            - Moderate hit rate (55-60%)
            - May provide slight edge
            - Best used as supplementary indicator
            - Combine with other analysis
            """)
        else:
            st.warning("""
            ⚠️ **WEAK SIGNAL** - Limited practical value:
            - Low hit rate (<55%)
            - Close to random
            - Not recommended for decision making
            - Focus on other metrics
            """)
    
    st.markdown("---")
    st.caption("💡 **Note:** Past performance does not guarantee future results. Use these insights as one input among many in your decision-making process.")

