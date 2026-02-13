# Testing Guide: Enhanced Commodities Page

## Quick Test Checklist

### **Prerequisites:**
```bash
# 1. Ensure commodities data exists
python scripts/fetch_commodities.py

# 2. Verify data file
ls -lh data/commodities/prices.parquet
```

### **Expected Output:**
- File should exist with recent modification date
- Size: varies based on data range

---

## Test Each New Analysis Type

### **1. Log Returns Analysis** ✨

**Test Steps:**
1. Navigate to Commodities page
2. Select "Log Returns Analysis" from dropdown
3. Select 2-3 commodities (e.g., Gold, Silver, Copper)

**Expected Results:**
- ✅ Time series chart showing log vs arithmetic returns
- ✅ Scatter plot with diagonal reference line
- ✅ Statistics table with both return types
- ✅ Difference chart showing Arithmetic - Log
- ✅ Info box explaining log returns

**What to Check:**
- Log returns should be slightly lower than arithmetic returns
- Difference increases with volatility
- Scatter points should cluster around diagonal

---

### **2. Cumulative Wealth (NAV)** ✨

**Test Steps:**
1. Select "Cumulative Wealth (NAV)" from dropdown
2. Change initial investment (sidebar): $10,000 → $100,000
3. Select multiple commodities

**Expected Results:**
- ✅ NAV path chart in dollars (not percentages)
- ✅ Performance table with P&L in dollars
- ✅ CAGR calculations
- ✅ Bar chart comparing final values
- ✅ Horizontal line at initial investment

**What to Check:**
- Y-axis should show dollar values
- Table should show both dollar and percentage returns
- CAGR should differ from arithmetic mean
- Green/red colors based on profit/loss

---

### **3. Drawdown Analysis** ✨

**Test Steps:**
1. Select "Drawdown Analysis" from dropdown
2. Use full date range
3. Select volatile commodities (e.g., Oil, Copper)

**Expected Results:**
- ✅ Drawdown time series (negative values, filled to zero)
- ✅ Max drawdown statistics table
- ✅ Recovery time analysis
- ✅ Drawdown duration histogram
- ✅ Underwater plot

**What to Check:**
- All drawdown values should be ≤ 0%
- Max drawdown should be the most negative value
- Recovery time should be "Not recovered" if still in drawdown
- Current DD should match latest value in chart

---

### **4. Risk Metrics Dashboard** ✨

**Test Steps:**
1. Select "Risk Metrics Dashboard" from dropdown
2. Select 3-4 commodities with different characteristics

**Expected Results:**
- ✅ Comprehensive table with 13 columns
- ✅ All metrics populated (no NaN)
- ✅ Bar chart comparing Sharpe vs Sortino vs Calmar
- ✅ Expander with interpretations

**What to Check:**
- Sortino ≥ Sharpe (usually)
- VaR 99% should be more negative than VaR 95%
- CVaR should be more negative than VaR
- Negative skew = left tail (more losses)
- Positive kurtosis = fat tails

---

### **5. Rolling Metrics** ✨

**Test Steps:**
1. Select "Rolling Metrics" from dropdown
2. Adjust rolling window (sidebar): 252 → 126 (shorter window)
3. Select 2 assets for correlation

**Expected Results:**
- ✅ Rolling Sharpe ratio chart
- ✅ Rolling Sortino ratio chart
- ✅ Rolling volatility chart
- ✅ Rolling correlation chart (if 2+ assets)
- ✅ Reference lines at key levels

**What to Check:**
- Shorter window = more volatile metrics
- Sharpe and Sortino should track similarly
- Volatility should spike during market stress
- Correlation should vary over time

---

### **6. Return Distribution** ✨

**Test Steps:**
1. Select "Return Distribution" from dropdown
2. Select volatile asset (e.g., Oil)

**Expected Results:**
- ✅ Histogram: Log vs Arithmetic overlaid
- ✅ Q-Q plot for each asset
- ✅ Distribution statistics table
- ✅ Jarque-Bera test results
- ✅ Educational expander

**What to Check:**
- Q-Q plot: points should deviate from line at tails
- Jarque-Bera p-value < 0.05 = NOT normal
- Kurtosis > 0 = fat tails (common in commodities)
- Histogram should show spread of returns

---

### **7. Multi-Period Performance** ✨

**Test Steps:**
1. Select "Multi-Period Performance" from dropdown
2. Select multiple commodities
3. Check if you have at least 3 years of data

**Expected Results:**
- ✅ Performance table (8 time periods)
- ✅ Color coding (green positive, red negative)
- ✅ Bar charts for each period
- ✅ CAGR table

**What to Check:**
- N/A for periods with insufficient data
- Colors should match signs (green = positive)
- CAGR should be lower than total return % for multi-year periods
- YTD should be year-to-date only

---

## Edge Cases to Test

### **1. Single Asset Selected**
- All charts should still work
- No correlation analysis (needs 2+ assets)

### **2. Short Date Range (< 1 year)**
- Rolling metrics may not have enough data
- Multi-period table will show many N/A

### **3. Missing Data**
- Should handle NaN gracefully
- Charts should skip missing periods

### **4. Weekly/Monthly Frequency**
- All metrics should adjust annualization factors
- Rolling window defaults should change

---

## Performance Test

### **Test with All 14 Commodities:**
```bash
# Expected load time for each analysis:
- Price Trends: < 2 seconds
- Log Returns: < 3 seconds (calculations)
- Cumulative Wealth: < 2 seconds
- Drawdown: < 3 seconds (expanding calculations)
- Risk Metrics: < 4 seconds (many calculations)
- Rolling Metrics: < 5 seconds (rolling calculations)
- Return Distribution: < 4 seconds (Q-Q plots)
- Multi-Period: < 3 seconds
```

---

## Visual Inspection Checklist

### **Charts Should Be:**
- ✅ Properly sized (height: 500-600px)
- ✅ Interactive (hover, zoom, pan)
- ✅ Legend visible and positioned well
- ✅ Axes labeled clearly
- ✅ Colors distinguishable
- ✅ Date range selector working

### **Tables Should Be:**
- ✅ Formatted with proper decimals
- ✅ Color-coded where appropriate
- ✅ Full width (use_container_width=True)
- ✅ No index column showing
- ✅ Headers descriptive

### **Info Boxes Should Be:**
- ✅ Blue background (st.info)
- ✅ Concise text (2-4 bullet points)
- ✅ Helpful context
- ✅ Not overwhelming

---

## Common Issues & Fixes

### **Issue: "No commodities data found"**
**Fix:**
```bash
python scripts/fetch_commodities.py
```

### **Issue: Charts not displaying**
**Fix:**
```bash
pip install plotly --upgrade
```

### **Issue: NaN in metrics tables**
**Fix:**
- Check date range has enough data
- Ensure at least 2 data points for calculations

### **Issue: Slow performance**
**Fix:**
- Reduce date range
- Use Weekly or Monthly frequency
- Select fewer commodities

---

## Acceptance Criteria

### **Must Pass:**
1. ✅ All 12 analysis types load without errors
2. ✅ Charts render properly
3. ✅ Tables show data (no all-NaN columns)
4. ✅ Color coding works
5. ✅ No Python exceptions in terminal
6. ✅ Educational content displays

### **Should Pass:**
7. ✅ Page loads in < 2 seconds (with cached data)
8. ✅ Responsive layout (no horizontal scroll)
9. ✅ Tooltips and help text work
10. ✅ Date range selector updates charts

---

## Regression Testing

### **Original Features Should Still Work:**
- ✅ Price Trends (unchanged)
- ✅ Returns Analysis (Arithmetic) - renamed but same functionality
- ✅ Correlation Matrix (unchanged)
- ✅ Normalized Comparison (unchanged)
- ✅ Seasonality Analysis (unchanged)

---

## Final Validation

### **Complete Test Run:**
```bash
# 1. Start fresh
conda activate quant

# 2. Launch app
streamlit run apps/portfolio_simulator.py

# 3. Navigate to Commodities page

# 4. Test EACH of the 12 analysis types

# 5. Test with different:
   - Number of commodities (1, 3, all)
   - Date ranges (1 year, 5 years, all)
   - Frequencies (Daily, Weekly, Monthly)

# 6. Check browser console for JS errors (F12)

# 7. Check terminal for Python errors
```

---

## Success Criteria

✅ **PASS if:**
- All 12 analysis types work
- No Python exceptions
- No JavaScript errors
- Charts are interactive
- Data is accurate
- Performance is acceptable

❌ **FAIL if:**
- Any analysis type crashes
- Charts don't render
- Data shows NaN everywhere
- Page takes > 10 seconds to load
- Layout is broken

---

## Documentation

After testing, document:
- ✅ Test date and time
- ✅ Browser used
- ✅ Any issues found
- ✅ Performance notes
- ✅ User feedback

---

**Testing Completed By:** _______________  
**Date:** _______________  
**Status:** ⬜ Pass | ⬜ Fail | ⬜ Partial  
**Notes:** _______________
