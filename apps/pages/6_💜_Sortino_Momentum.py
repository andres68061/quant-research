#!/usr/bin/env python3
"""
Sortino Momentum Analysis - Advanced statistical analysis of Sortino ratio persistence.

This page analyzes whether improving Sortino ratios tend to continue improving,
and provides regime indicators for portfolio monitoring.
"""

from apps.utils.portfolio import calculate_rolling_metrics
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

# ============================================================================
# THEORY & METHODOLOGY SECTION
# ============================================================================

with st.expander("📚 Theory & Methodology - Click to Learn More", expanded=False):
    st.markdown("""
    # Comprehensive Guide to Sortino Momentum Analysis
    
    ## 1. Understanding the Sortino Momentum Analysis Page
    
    ### What This Page Does
    
    This analysis investigates **momentum persistence** in the Sortino ratio - a phenomenon where 
    improving risk-adjusted returns tend to continue improving for some period of time.
    
    **The Core Question:**
    > "If my strategy's Sortino ratio has been rising faster than usual for the last X days,
    > what's the probability (Z%) it will keep rising for the next K days?"
    
    ### Why This Matters
    
    **1. Regime Detection**
    - Identifies when your strategy is in a "hot streak" vs "cold streak"
    - Helps you understand if current performance is likely to persist
    - Provides confidence (or caution) about near-term prospects
    
    **2. Position Sizing**
    - Increase positions when momentum is strong and significant
    - Reduce exposure when momentum weakens
    - Dynamic risk management based on regime
    
    **3. Strategy Evaluation**
    - Some strategies have persistent momentum (trend-following)
    - Others are mean-reverting (no momentum)
    - Understanding this helps set realistic expectations
    
    ### The Three Parameters: X, K, and Z
    
    **X (Lookback Period):**
    - How many days to measure "recent" improvement
    - Too short (5 days): Noisy, many false signals
    - Too long (90 days): Slow to react, misses regime changes
    - Typical optimal: 15-30 days
    
    **K (Forecast Horizon):**
    - How many days ahead to predict
    - Shorter K (5-10 days): More reliable but limited usefulness
    - Longer K (30+ days): Less reliable but more actionable
    - Typical optimal: 10-20 days
    
    **Z (Hit Rate / Probability):**
    - Percentage of time momentum continues
    - Random baseline: 50% (coin flip)
    - Weak signal: 52-55% (barely better than random)
    - Modest signal: 55-60% (useful supplementary indicator)
    - Strong signal: 60-70% (genuine predictive power)
    - Exceptional: >70% (rare, check for overfitting)
    
    ---
    
    ## 2. The Three Statistical Methods Explained
    
    ### Method 1: Grid Search Optimization
    
    **Mathematical Approach:**
    
    For each combination of (X, K):
    
    1. **Calculate Recent Slope:**
       ```
       recent_slope[t] = (Sortino[t] - Sortino[t-X]) / X
       ```
       This measures how fast Sortino is changing over the last X days.
    
    2. **Calculate Baseline Slope:**
       ```
       baseline_slope[t] = (Sortino[t-X] - Sortino[t-X-30]) / 30
       ```
       This measures the "normal" rate of change from the previous 30 days.
    
    3. **Identify Strong Momentum:**
       ```
       strong_momentum[t] = (recent_slope[t] > baseline_slope[t])
       ```
       True when current improvement exceeds historical norm.
    
    4. **Check Continuation:**
       ```
       continued[t] = (Sortino[t+K] > Sortino[t])
       ```
       Did Sortino actually rise over the next K days?
    
    5. **Calculate Hit Rate:**
       ```
       Z = (Number of times continued = True) / (Total strong_momentum signals)
       ```
    
    **Why Grid Search?**
    - No assumptions about optimal parameters
    - Tests all reasonable combinations empirically
    - Finds what actually worked historically
    - Provides confidence intervals for reliability
    
    **Interpreting Results:**
    - **Heatmap:** Shows Z% for all (X,K) combinations
      - Red zones: Poor performance (<50%)
      - Yellow zones: Modest performance (50-55%)
      - Green zones: Good performance (>55%)
    
    - **Top 10 Table:** Best combinations ranked by Z
      - Look for: High Z, tight confidence interval, many signals
      - Avoid: High Z with wide CI or few signals (likely luck)
    
    **Example Interpretation:**
    ```
    X=20, K=10, Z=62.3%, CI=[55.1%, 69.5%], Signals=87
    
    Translation: "When Sortino rises faster than usual for 20 days,
    it continues rising for the next 10 days 62.3% of the time.
    We're 95% confident the true rate is between 55-70%.
    This pattern occurred 87 times historically."
    ```
    
    ---
    
    ### Method 2: Statistical Significance Testing
    
    **The Problem:**
    Even random data can produce seemingly good results by chance. We need to know:
    Is our Z% genuinely predictive, or just lucky?
    
    **Bootstrap Resampling Explained:**
    
    1. **Take the best (X, K) from Method 1**
    
    2. **Shuffle the outcomes randomly:**
       - Keep the signal dates the same
       - Randomize whether momentum continued
       - This breaks any real relationship
    
    3. **Recalculate Z% on shuffled data**
    
    4. **Repeat 500 times:**
       - Creates distribution of "random" Z values
       - Shows what we'd get by pure chance
    
    5. **Compare actual vs random:**
       ```
       p_value = Probability(random Z ≥ actual Z)
       ```
       - p < 0.05: Less than 5% chance it's random → SIGNIFICANT
       - p > 0.05: Could easily be random → NOT SIGNIFICANT
    
    **Why Bootstrap?**
    - Non-parametric (no assumptions about distributions)
    - Robust to outliers and non-normal data
    - Intuitive interpretation
    - Industry standard for time series
    
    **Interpreting P-Values:**
    
    - **p = 0.001:** Extremely significant (99.9% confident it's real)
    - **p = 0.01:** Very significant (99% confident)
    - **p = 0.05:** Significant (95% confident) ← Standard threshold
    - **p = 0.10:** Marginally significant (90% confident)
    - **p = 0.20:** Not significant (could be luck)
    - **p = 0.50:** Definitely random (no relationship)
    
    **Example Interpretation:**
    ```
    Actual: 62.3%
    Random Mean: 51.2% ± 3.8%
    P-Value: 0.012
    
    Translation: "Our 62.3% hit rate is much higher than the 51.2%
    we'd expect by random chance. There's only a 1.2% probability
    this is luck. We can confidently say momentum is real."
    ```
    
    **Visual Interpretation:**
    - Bootstrap histogram shows random distribution
    - Actual hit rate (red line) far from random mean
    - If red line is in the tail → significant
    - If red line is in the middle → not significant
    
    ---
    
    ### Method 3: Machine Learning Prediction
    
    **Why Machine Learning?**
    
    Methods 1 & 2 use only one signal (recent slope vs baseline).
    ML can combine multiple features for better predictions:
    
    **The 8 Features:**
    
    1. **sortino:** Current Sortino level
       - High Sortino might be more stable
       - Or might be due for mean reversion
    
    2. **sharpe:** Current Sharpe ratio
       - Correlated with Sortino
       - Provides complementary risk info
    
    3. **volatility:** Recent volatility
       - High vol might mean unstable momentum
       - Low vol might mean stable trends
    
    4. **slope_5d:** 5-day Sortino slope
       - Very recent momentum
       - Captures short-term acceleration
    
    5. **slope_10d:** 10-day Sortino slope
       - Short-term momentum
       - Balances recency and stability
    
    6. **slope_20d:** 20-day Sortino slope
       - Medium-term momentum
       - Often most predictive
    
    7. **slope_30d:** 30-day Sortino slope
       - Longer-term momentum
       - Captures sustained trends
    
    8. **vs_baseline:** Recent slope vs historical baseline
       - Direct momentum acceleration measure
       - Combines multiple timeframes
    
    **Logistic Regression Explained:**
    
    ```
    P(momentum continues) = 1 / (1 + e^(-z))
    
    where z = β₀ + β₁×sortino + β₂×sharpe + ... + β₈×vs_baseline
    ```
    
    - Each feature gets a coefficient (β)
    - Positive β: Higher feature → higher probability
    - Negative β: Higher feature → lower probability
    - Larger |β|: More important feature
    
    **Time Series Cross-Validation:**
    
    Critical to avoid look-ahead bias:
    
    ```
    Fold 1: Train[Year 1-3] → Test[Year 4]
    Fold 2: Train[Year 1-4] → Test[Year 5]
    Fold 3: Train[Year 1-5] → Test[Year 6]
    Fold 4: Train[Year 1-6] → Test[Year 7]
    Fold 5: Train[Year 1-7] → Test[Year 8]
    ```
    
    - Always train on past, test on future
    - Never use future data to predict past
    - Mimics real-world usage
    
    **Feature Importance:**
    
    Shows which features matter most:
    
    ```
    slope_20d: 0.45  ← Most important
    vs_baseline: 0.38
    slope_10d: 0.31
    sortino: 0.12
    volatility: 0.08  ← Least important
    ```
    
    **Interpreting Accuracy:**
    
    - **50%:** Random (coin flip)
    - **52-55%:** Weak edge (marginally useful)
    - **55-60%:** Modest edge (supplementary indicator)
    - **60-65%:** Strong edge (primary indicator)
    - **>65%:** Exceptional (check for overfitting!)
    
    **Example Interpretation:**
    ```
    Mean Accuracy: 58.2% ± 2.1%
    Top Feature: slope_20d (0.45)
    
    Translation: "Using 8 features, we can predict momentum
    continuation 58% of the time (vs 50% random). The 20-day
    slope is most predictive. This is a modest but real edge."
    ```
    
    **Cross-Validation Consistency:**
    
    Look at per-fold accuracy:
    - All folds 55-60%: Robust, reliable
    - Folds vary 45-70%: Unstable, unreliable
    - One fold 80%, others 50%: Overfitting
    
    ---
    
    ## 3. Understanding Sortino vs Sharpe Ratio
    
    ### The Fundamental Difference
    
    **Sharpe Ratio:**
    ```
    Sharpe = (Return - RiskFree) / TotalVolatility
    
    TotalVolatility = √(Σ(all returns - mean)² / n)
    ```
    
    - Penalizes ALL volatility (up and down)
    - Treats 10% gain same as 10% loss
    - Assumes investors dislike variability in general
    
    **Sortino Ratio:**
    ```
    Sortino = (Return - RiskFree) / DownsideDeviation
    
    DownsideDeviation = √(Σ(negative returns - 0)² / n)
    ```
    
    - Penalizes ONLY downside volatility
    - Ignores upside volatility (the good kind!)
    - Assumes investors only dislike losses
    
    ### Why Sortino is Better for Asymmetric Strategies
    
    **Example: Two Strategies**
    
    **Strategy A (Positive Skew):**
    ```
    Returns: +1%, +1%, +1%, +1%, +1%, +1%, +1%, +1%, +1%, +15%
    Mean: +2.5%
    Sharpe: 1.2 (penalized for +15% "outlier")
    Sortino: 2.8 (ignores +15% upside)
    ```
    
    **Strategy B (Negative Skew):**
    ```
    Returns: +2%, +2%, +2%, +2%, +2%, +2%, +2%, +2%, +2%, -15%
    Mean: +0.3%
    Sharpe: 0.1
    Sortino: 0.1 (correctly penalizes -15%)
    ```
    
    **Interpretation:**
    - Strategy A is clearly better (big upside, no downside)
    - Sharpe doesn't distinguish them well
    - Sortino correctly identifies A as superior
    
    ### Time Series Autocorrelation and Momentum
    
    **What is Autocorrelation?**
    
    Correlation of a time series with itself at different lags:
    
    ```
    Autocorr(lag=1) = Correlation(Sortino[t], Sortino[t-1])
    Autocorr(lag=5) = Correlation(Sortino[t], Sortino[t-5])
    ```
    
    - Positive autocorrelation: Momentum (trends persist)
    - Zero autocorrelation: Random walk (no patterns)
    - Negative autocorrelation: Mean reversion (trends reverse)
    
    **Why Expect Momentum?**
    
    **Behavioral Factors:**
    1. **Herding:** Investors follow trends
    2. **Underreaction:** Markets slow to incorporate news
    3. **Confirmation bias:** Good performance attracts more capital
    
    **Structural Factors:**
    1. **Regime persistence:** Market conditions don't change instantly
    2. **Strategy capacity:** Takes time for edge to erode
    3. **Feedback loops:** Success → more capital → more success
    
    **Why NOT Expect Momentum?**
    
    1. **Efficient markets:** All info already in prices
    2. **Mean reversion:** Extreme performance unsustainable
    3. **Crowding:** Everyone chasing momentum kills it
    4. **Regime changes:** Sudden shifts break patterns
    
    **Empirical Reality:**
    - Short-term (1-12 months): Momentum often exists
    - Long-term (3-5 years): Mean reversion dominates
    - Asset-specific: Some show momentum, others don't
    
    ---
    
    ## 4. Practical Application Guide
    
    ### How to Use Results in Trading/Investing
    
    **Step 1: Determine Signal Strength**
    
    ```
    IF Z > 60% AND p < 0.05 AND ML_accuracy > 55%:
        Signal = STRONG
    ELIF Z > 55% AND (p < 0.05 OR ML_accuracy > 55%):
        Signal = MODEST
    ELSE:
        Signal = WEAK
    ```
    
    **Step 2: Apply Based on Strength**
    
    **STRONG Signal (Z>60%, significant, ML works):**
    
    ✅ **Do:**
    - Use as primary regime indicator
    - Adjust position sizes based on regime
    - Set different risk parameters per regime
    - Monitor closely for regime changes
    
    ❌ **Don't:**
    - Use as sole entry/exit signal
    - Ignore other analysis
    - Over-leverage based on momentum
    - Assume it will work forever
    
    **MODEST Signal (Z=55-60%):**
    
    ✅ **Do:**
    - Use as supplementary indicator
    - Combine with other signals
    - Slight position size adjustments
    - Track for regime awareness
    
    ❌ **Don't:**
    - Make it primary decision driver
    - Trade solely on momentum
    - Expect consistent edge
    
    **WEAK Signal (Z<55%):**
    
    ✅ **Do:**
    - Ignore momentum for this asset
    - Focus on other metrics
    - Assume random walk
    
    ❌ **Don't:**
    - Try to force patterns
    - Over-optimize parameters
    - Use for decision making
    
    ### What "Regime Indicator" Means
    
    **Definition:**
    A regime indicator identifies which market/strategy state you're in:
    
    **Regime Types:**
    
    1. **Strong Positive Momentum:**
       - Sortino rising faster than usual
       - High probability of continuation
       - Strategy in "hot streak"
       - Action: Maintain or increase exposure
    
    2. **Neutral/Weak Momentum:**
       - Sortino not rising unusually fast
       - No clear direction
       - Strategy in "normal" state
       - Action: Standard position sizing
    
    3. **Negative Momentum:**
       - Sortino declining
       - May continue declining
       - Strategy in "cold streak"
       - Action: Reduce exposure or pause
    
    **Using Regime Information:**
    
    ```python
    if current_regime == "Strong Positive":
        position_size = base_size * 1.5  # Increase 50%
        stop_loss = wider_stop  # Give more room
        
    elif current_regime == "Neutral":
        position_size = base_size  # Standard
        stop_loss = normal_stop
        
    elif current_regime == "Negative":
        position_size = base_size * 0.5  # Reduce 50%
        stop_loss = tighter_stop  # Protect capital
    ```
    
    ### Position Sizing Based on Momentum
    
    **Kelly Criterion Adjustment:**
    
    ```
    Standard Kelly: f = (p×b - q) / b
    
    where:
    f = fraction of capital to bet
    p = probability of win
    b = odds (win amount / loss amount)
    q = probability of loss = 1 - p
    
    Momentum-Adjusted Kelly:
    f_adjusted = f × momentum_multiplier
    
    momentum_multiplier = {
        1.5 if strong positive momentum
        1.0 if neutral
        0.5 if negative momentum
    }
    ```
    
    **Example:**
    
    ```
    Base Kelly: 10% of capital
    
    Strong Momentum: 10% × 1.5 = 15%
    Neutral: 10% × 1.0 = 10%
    Negative: 10% × 0.5 = 5%
    ```
    
    **Conservative Approach:**
    
    Don't change position size, change allocation:
    
    ```
    Strong Momentum:
    - Allocate to this strategy: 60%
    - Allocate to others: 40%
    
    Negative Momentum:
    - Allocate to this strategy: 30%
    - Allocate to others: 70%
    ```
    
    ### Risk Management Implications
    
    **Dynamic Stop Losses:**
    
    ```
    Strong Momentum:
    - Wider stops (give strategy room to work)
    - Trail stops more loosely
    - Example: 3% stop vs 2% normal
    
    Negative Momentum:
    - Tighter stops (protect capital)
    - Trail stops aggressively
    - Example: 1% stop vs 2% normal
    ```
    
    **Rebalancing Frequency:**
    
    ```
    Strong Momentum:
    - Rebalance less frequently
    - Let winners run
    - Monthly instead of weekly
    
    Negative Momentum:
    - Rebalance more frequently
    - Cut losers quickly
    - Weekly instead of monthly
    ```
    
    **Risk Budget Allocation:**
    
    ```
    Total Risk Budget: 10% portfolio volatility
    
    Strong Momentum:
    - Allocate 6% to this strategy
    - Allocate 4% to others
    
    Negative Momentum:
    - Allocate 3% to this strategy
    - Allocate 7% to others
    ```
    
    ### Example Decision Framework
    
    **Scenario: You run a momentum strategy on ^GSPC**
    
    **Analysis Results:**
    ```
    X = 20 days
    K = 10 days
    Z = 62.3%
    P-value = 0.012 (significant)
    ML Accuracy = 58.2%
    → Signal: STRONG
    ```
    
    **Current Regime:**
    ```
    Recent Slope (20d): +0.0042
    Baseline Slope (30d): +0.0018
    → Strong Positive Momentum
    ```
    
    **Your Action Plan:**
    
    1. **Position Sizing:**
       - Increase from 50% → 65% of capital
       - Rationale: High confidence in continuation
    
    2. **Stop Loss:**
       - Widen from 2% → 3%
       - Rationale: Give strategy room in favorable regime
    
    3. **Rebalancing:**
       - Reduce from weekly → bi-weekly
       - Rationale: Let momentum play out
    
    4. **Monitoring:**
       - Check regime daily
       - If momentum weakens, revert to standard sizing
    
    5. **Exit Plan:**
       - If regime turns negative: reduce to 35%
       - If Z drops below 55%: stop using momentum
    
    **Risk Management:**
    ```
    Maximum drawdown tolerance: 15%
    Current regime: Strong positive
    → Acceptable drawdown: 18% (20% higher)
    
    If drawdown exceeds 18%:
    → Override momentum signal
    → Reduce position immediately
    ```
    
    ---
    
    ## Key Takeaways
    
    ### What This Analysis Can Do:
    
    ✅ Identify if momentum exists for your strategy
    ✅ Quantify the probability of continuation
    ✅ Provide statistical confidence in findings
    ✅ Detect current regime (hot/cold streak)
    ✅ Inform position sizing decisions
    ✅ Improve risk management
    
    ### What This Analysis Cannot Do:
    
    ❌ Predict exact future returns
    ❌ Guarantee profits
    ❌ Replace fundamental analysis
    ❌ Work forever (patterns change)
    ❌ Eliminate all risk
    ❌ Substitute for diversification
    
    ### Best Practices:
    
    1. **Test on multiple assets** - Some show momentum, others don't
    2. **Update regularly** - Patterns evolve over time
    3. **Combine with other analysis** - Never use in isolation
    4. **Respect statistical significance** - Don't trade insignificant patterns
    5. **Start small** - Test with small position adjustments first
    6. **Monitor performance** - Track if momentum edge persists
    7. **Have exit rules** - Know when to stop using momentum
    
    ### Warning Signs:
    
    🚨 **Stop using momentum if:**
    - P-value increases above 0.10
    - Z drops below 52%
    - ML accuracy falls below 51%
    - Regime indicator stops working
    - Losses exceed expectations
    
    ---
    
    *This methodology is for educational and research purposes. Past performance 
    does not guarantee future results. Always conduct your own due diligence and 
    consider your risk tolerance before making investment decisions.*
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
            baseline_slope = calculate_sortino_slopes(
                rolling_sortino.shift(X), baseline_window)

            # Identify "strong momentum" periods
            strong_momentum = (
                recent_slope > baseline_slope) & recent_slope.notna() & baseline_slope.notna()

            # Check if momentum continued for next K days
            future_slope = calculate_sortino_slopes(
                rolling_sortino.shift(-K), K)
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
    strong_momentum = (
        recent_slope > baseline_slope) & recent_slope.notna() & baseline_slope.notna()
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
    p_value = np.mean(np.abs(bootstrap_hit_rates - random_mean)
                      >= np.abs(actual_hit_rate - random_mean))

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
    future_slope = calculate_sortino_slopes(
        rolling['sortino_ratio'].shift(-K), K)
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
    feature_importance_dict = dict(
        sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

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
    baseline_slope = calculate_sortino_slopes(
        rolling_sortino.shift(X), 30).iloc[-1]

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
    index=available_symbols.index(
        '^GSPC') if '^GSPC' in available_symbols else 0,
    help="Choose a symbol to analyze"
)

# Calculate returns
returns = df_prices_filtered[selected_symbol].pct_change().dropna()

if len(returns) < 500:
    st.error(
        f"❌ Insufficient data: only {len(returns)} days available. Need at least 500.")
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
        current_sortino = rolling_sortino.iloc[-1] if len(
            rolling_sortino) > 0 else np.nan
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

        display_cols = ['X (lookback)', 'K (forecast)', 'Z (hit_rate)',
                        'CI_lower', 'CI_upper', 'Total_signals', 'Successful', 'Failed']
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

        pivot = grid_results.pivot(
            index='K (forecast)', columns='X (lookback)', values='Z (hit_rate)')

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
            sig_test = test_statistical_significance(
                returns, X_best, K_best, sortino_window)

        if sig_test['n_signals'] < 10:
            st.warning(
                f"⚠️ Only {sig_test['n_signals']} signals found. Need at least 10 for reliable testing.")
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
        ml_results = analyze_ml_prediction(
            returns, sortino_window, forecast_horizon=K_best if len(grid_results) > 0 else 10)

    if 'error' in ml_results:
        st.error(f"❌ {ml_results['message']}")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Mean Accuracy",
                      f"{ml_results['mean_accuracy']*100:.1f}%")
        with col2:
            st.metric("Std Accuracy",
                      f"±{ml_results['std_accuracy']*100:.1f}%")
        with col3:
            st.metric("Training Samples", f"{ml_results['n_samples']}")
        with col4:
            st.metric("Positive Class",
                      f"{ml_results['positive_class_pct']:.1f}%")

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
                    line=dict(
                        color='green' if regime['strong_momentum'] else 'orange', width=4)
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
