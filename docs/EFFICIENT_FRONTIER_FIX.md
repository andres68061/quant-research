# Efficient Frontier Fix: Asset Filtering

## Issue Reported

User noticed the efficient frontier looked "weird" compared to the R Markdown template.

## Root Cause

**Critical difference in Modern Portfolio Theory implementation:**

### R Code (Correct)
```r
# Lines 388-396 in ETF_portfolio_template.Rmd
expret_risky_annualized <- (1 + expret_risky)^252 - 1

# Filter assets with expected returns >= risk-free rate
valid_risky_indices <- risky_indices[expret_risky_annualized >= risk_free_rate]
valid_risky_assets <- params$Assets[valid_risky_indices]
```

The R code **filters out assets with expected returns below the risk-free rate** before calculating the efficient frontier.

### Python Code (Before Fix)
```python
# Used ALL assets without filtering
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
# ... directly used for efficient frontier
```

## Why This Matters

### Economic Rationale
1. **Rational investors wouldn't hold risky assets with returns below the risk-free rate**
   - If an asset has expected return < risk-free rate, why take the risk?
   - You'd be better off holding the risk-free asset

2. **The efficient frontier should only include assets that offer a risk premium**
   - Risk premium = Expected Return - Risk-Free Rate
   - Only positive risk premiums make economic sense

3. **Assets below the risk-free rate distort the tangency portfolio calculation**
   - The tangency portfolio maximizes the Sharpe ratio
   - Including low-return assets creates a misleading "optimal" portfolio

### Mathematical Impact
- **Efficient Frontier Shape**: Including low-return assets creates a distorted curve
- **Tangency Portfolio**: The max Sharpe ratio point shifts incorrectly
- **Capital Allocation Line (CAL)**: Originates from wrong tangency point
- **50/50 Portfolio**: Uses incorrect tangency weights

## Solution Implemented

### 1. Filter Assets
```python
# Filter assets with expected returns >= risk-free rate
valid_assets_mask = mean_returns >= risk_free_annual
valid_assets_list = mean_returns[valid_assets_mask].index.tolist()

# Create filtered datasets
returns_risky = returns[valid_assets_list].copy()
mean_returns_risky = mean_returns[valid_assets_list]
cov_matrix_risky = cov_matrix.loc[valid_assets_list, valid_assets_list]
```

### 2. Use Filtered Data for Optimization
```python
# Calculate efficient frontier (using only risky assets)
efficient_portfolios, min_var_point = calculate_efficient_frontier(
    mean_returns_risky.values,
    cov_matrix_risky.values,
    risk_free_annual
)

# Find tangency portfolio
tangency_weights_risky = find_tangency_portfolio(
    mean_returns_risky.values,
    cov_matrix_risky.values,
    risk_free_annual
)
```

### 3. Create Full Weights Vector for Display
```python
# Create full weights vector (including zeros for excluded assets)
tangency_weights_full = pd.Series(0.0, index=valid_assets)
tangency_weights_full[valid_assets_list] = tangency_weights_risky
```

### 4. Update Simulation
```python
# Simulate tangency portfolio (using only risky assets)
tangency_value = simulate_portfolio_with_rebalancing(
    returns_risky,
    tangency_weights_risky,
    rebalance_freq
)
```

## User Experience Improvements

### Before Fix
- Efficient frontier looked distorted
- Tangency portfolio included suboptimal assets
- CAL didn't align properly
- No transparency about excluded assets

### After Fix
- Clean, economically rational efficient frontier
- Tangency portfolio maximizes Sharpe among valid assets
- CAL correctly originates from risk-free rate through tangency
- Info message shows excluded assets:
  ```
  ℹ️ Using 11 risky assets with returns ≥ risk-free rate. 
  Excluded: 1 asset(s).
  ```

## Verification

### R Code Behavior
```r
if (num_valid_risky_assets == 0) {
  stop("No risky assets have expected returns above the risk-free rate")
}
```

### Python Code Behavior (Now Matches)
```python
if len(valid_assets_list) < 2:
    st.error(
        f"❌ Not enough risky assets with returns above risk-free rate"
    )
    st.info(f"Assets below risk-free rate: {excluded_assets}")
    st.stop()
```

## Key Takeaways

1. **MPT Principle**: Only include assets with positive risk premiums
2. **Replication Accuracy**: Must match R code logic exactly
3. **Economic Rationality**: Filter reflects real investor behavior
4. **Transparency**: Show users which assets are excluded and why

## Files Modified

- `apps/pages/8_📈_ETF_Portfolio_Optimizer.py`
  - Added asset filtering logic
  - Updated efficient frontier calculation
  - Updated tangency portfolio calculation
  - Updated simulation logic
  - Added user info messages

## Related Fixes

- **CETES 28 Timezone Fix** (commit 29725bc): Ensured timezone consistency
- **ETF Portfolio Optimizer Creation** (commit 548651d): Initial implementation

## Status

✅ **FIXED** - Efficient frontier now correctly replicates R Markdown behavior
