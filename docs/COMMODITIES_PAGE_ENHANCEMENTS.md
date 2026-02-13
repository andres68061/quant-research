# Commodities Page Enhancements

## Overview

Enhanced the Commodities & Metals Analytics page (`apps/pages/2_📊_Metals_Analytics.py`) with 7 new analysis types based on practical quantitative workflow requirements.

**Date:** February 3, 2026

**Motivation:** Following the principle "Never report only one series" and incorporating proper quant workflow practices (log returns for modeling, NAV for reporting, geometric metrics for performance).

---

## What Was Added

### **Original Analysis Types (5):**
1. ✅ Price Trends
2. ✅ Returns Analysis (Arithmetic) - *renamed*
3. ✅ Correlation Matrix
4. ✅ Normalized Comparison
5. ✅ Seasonality Analysis

### **New Analysis Types (7):**

#### **1. Log Returns Analysis** 🔥
**Purpose:** Foundation for signal generation and risk modeling

**Features:**
- Time series comparison: Log vs Arithmetic returns
- Scatter plot showing relationship between log and arithmetic returns
- Statistics comparison table (mean, volatility, skewness, kurtosis)
- Difference analysis (Arithmetic - Log)
- Educational info box explaining why log returns matter

**Key Metrics:**
- Log returns are time-additive
- Better for multi-period analysis
- More appropriate for econometric models

---

#### **2. Cumulative Wealth (NAV)** 🔥
**Purpose:** Show actual dollar P&L over time - what investors see!

**Features:**
- Configurable initial investment amount (sidebar)
- NAV path chart showing portfolio value over time
- Performance table with dollar P&L and percentage returns
- CAGR calculation for each asset
- Bar chart comparing final portfolio values

**Key Metrics:**
- Start Value, End Value, P&L ($), Total Return (%), CAGR (%)
- Actual dollar values (not just percentages)

---

#### **3. Drawdown Analysis** 🔥
**Purpose:** Critical for understanding downside risk

**Features:**
- Drawdown time series (filled area chart)
- Maximum drawdown statistics table
- Recovery time analysis
- Drawdown duration distribution histogram
- Underwater plot showing current drawdown status

**Key Metrics:**
- Max Drawdown (%)
- Max Drawdown Date
- Recovery Time (days)
- Current Drawdown (%)

---

#### **4. Risk Metrics Dashboard** 🔥
**Purpose:** Comprehensive risk-adjusted performance beyond Sharpe

**Features:**
- Comprehensive metrics table with 13 columns
- Bar chart comparing Sharpe vs Sortino vs Calmar
- Educational expander explaining each metric

**Metrics Included:**
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio (downside deviation only)
- Maximum Drawdown
- Calmar Ratio (CAGR / Max DD)
- VaR 95% and 99% (Value at Risk)
- CVaR 95% and 99% (Conditional VaR / Expected Shortfall)
- Skewness
- Kurtosis

---

#### **5. Rolling Metrics** 📈
**Purpose:** Time-varying risk analysis and regime detection

**Features:**
- Configurable rolling window (sidebar slider)
- Rolling Sharpe ratio chart
- Rolling Sortino ratio chart
- Rolling volatility chart
- Rolling correlation (for pairs of assets)
- Reference lines at key levels (0, 1.0)

**Key Use Cases:**
- Identify regime changes
- Detect periods of elevated risk
- Monitor time-varying correlations

---

#### **6. Return Distribution** 📊
**Purpose:** Understand tail risk and deviation from normality

**Features:**
- Histogram comparison: Log vs Arithmetic returns
- Q-Q plots (Quantile-Quantile) for normality testing
- Distribution statistics table with Jarque-Bera test
- Educational expander on interpretation

**Key Metrics:**
- Mean, Median, Std Dev
- Skewness (asymmetry)
- Kurtosis (tail fatness)
- Jarque-Bera test (p-value for normality)

**Key Insights:**
- Fat tails = more extreme events than normal distribution predicts
- Negative skew = asymmetric downside risk
- Most risk models assume normality (often wrong!)

---

#### **7. Multi-Period Performance** 📅
**Purpose:** Performance summary across multiple time horizons

**Features:**
- Performance table across 8 time periods
- Color-coded cells (green positive, red negative)
- Bar charts for each period
- CAGR (annualized returns) table

**Time Periods:**
- 1 Month, 3 Months, 6 Months
- YTD (Year-to-Date)
- 1 Year, 3 Years, 5 Years
- Since Inception

**Key Insights:**
- Identify consistency vs recent trends
- Quick performance summary
- Compare short-term vs long-term performance

---

## Implementation Details

### **Code Structure:**
- All new sections follow existing pattern
- Uses Plotly for interactive charts
- Consistent styling with existing pages
- Comprehensive error handling

### **Performance:**
- Leverages existing cached data
- Efficient vectorized calculations with pandas/numpy
- No additional API calls required

### **Educational Components:**
- Info boxes explaining concepts
- Expanders with detailed interpretations
- Tooltips on charts and metrics

---

## Technical Highlights

### **Proper Return Calculations:**
```python
# Arithmetic returns
arith_returns = prices.pct_change()

# Log returns
log_returns = np.log(prices / prices.shift(1))

# NAV path (geometric compounding)
nav = (1 + returns).cumprod() * initial_capital

# CAGR (geometric)
cagr = ((final_value / initial_value) ** (1 / years) - 1)
```

### **Risk Metrics:**
```python
# Sortino ratio (downside deviation only)
downside_returns = returns[returns < 0]
downside_std = downside_returns.std()
sortino = (mean_return / downside_std) * np.sqrt(ann_factor)

# Calmar ratio
calmar = cagr / max_drawdown

# Value at Risk (VaR)
var_95 = np.percentile(returns, 5)

# Conditional VaR (CVaR / Expected Shortfall)
cvar_95 = returns[returns <= var_95].mean()
```

### **Drawdown Calculation:**
```python
cum_returns = (1 + returns).cumprod()
running_max = cum_returns.expanding().max()
drawdown = (cum_returns - running_max) / running_max
```

---

## User Experience

### **Sidebar Configuration:**
- Analysis type selector (12 options)
- Initial investment amount (for NAV)
- Rolling window size (for rolling metrics)

### **Color Coding:**
- Green = positive returns
- Red = negative returns
- Orange/Purple = different metric types
- Gray = reference lines

### **Interactive Features:**
- Date range filters (existing)
- Asset selection (existing)
- Data frequency (existing)
- Tooltips and hover information
- Expandable help sections

---

## Next Steps / Future Enhancements

### **Potential Additions:**
1. **Benchmark Comparison** - Compare commodities to S&P 500
2. **Alpha/Beta Analysis** - Regression against benchmark
3. **Monte Carlo Simulation** - Forward-looking risk scenarios
4. **Regime Detection** - Automated regime identification
5. **Tail Risk Metrics** - Beyond VaR (EVT, GPD)
6. **Factor Decomposition** - PCA on commodity returns

### **Data Enhancements:**
- Add futures data (not just ETFs)
- Include carry/roll returns
- Term structure analysis (contango/backwardation)

---

## Documentation References

### **Related Docs:**
- Practical quant workflow principles
- `PORTFOLIO_SIMULATION_FIXES_APPLIED.md` - Similar metrics implemented
- Portfolio Simulator page - Reference implementation

### **Key Principles Applied:**
1. ✅ Never report only one series (log + arithmetic + NAV)
2. ✅ Model with log returns
3. ✅ Report with geometric metrics (CAGR, not arithmetic mean)
4. ✅ Always include drawdown analysis
5. ✅ Multiple risk metrics (not just Sharpe)

---

## Testing

### **Validation:**
✅ Python syntax check passed
✅ All imports valid
✅ Consistent with existing code style
✅ Error handling for edge cases

### **To Test:**
```bash
# 1. Ensure commodities data exists
python scripts/fetch_commodities.py

# 2. Launch Streamlit
streamlit run apps/portfolio_simulator.py

# 3. Navigate to "Metals Analytics" page

# 4. Test each new analysis type:
- Log Returns Analysis
- Cumulative Wealth (NAV)
- Drawdown Analysis
- Risk Metrics Dashboard
- Rolling Metrics
- Return Distribution
- Multi-Period Performance
```

---

## Summary

**Total Analysis Types:** 12 (5 original + 7 new)

**Lines Added:** ~1,800 lines of production code

**Key Improvements:**
- 🔥 Professional-grade quant analysis
- 🔥 Multiple views of same data (never just one series)
- 🔥 Proper geometric calculations (CAGR, NAV)
- 🔥 Comprehensive risk metrics (beyond Sharpe)
- 🔥 Educational components for understanding

**Result:** A world-class commodities analytics page suitable for institutional portfolio management! 🚀
