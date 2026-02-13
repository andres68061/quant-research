# Commodities Analytics - Quick Reference Card

## 🎯 12 Analysis Types at a Glance

### **1. Price Trends** 📈
- **What:** Raw price levels over time
- **Use Case:** Identify long-term trends
- **Key Output:** Price charts, min/max/mean statistics

### **2. Returns Analysis (Arithmetic)** 📊
- **What:** Traditional percentage returns
- **Use Case:** Portfolio optimization, quick performance check
- **Key Output:** Return distribution, Sharpe ratio

### **3. Log Returns Analysis** ✨ NEW
- **What:** Logarithmic returns (better for modeling)
- **Use Case:** Signal generation, time-series models, econometrics
- **Key Output:** Log vs arithmetic comparison, difference tracking
- **Why:** Time-additive, better for multi-period analysis

### **4. Cumulative Wealth (NAV)** ✨ NEW
- **What:** Dollar-based portfolio value over time
- **Use Case:** See actual dollar P&L, track real wealth
- **Key Output:** NAV curve, CAGR, dollar returns
- **Why:** What investors actually see in their accounts!

### **5. Drawdown Analysis** ✨ NEW
- **What:** Peak-to-trough decline analysis
- **Use Case:** Understand downside risk, capital preservation
- **Key Output:** Max drawdown %, recovery time, underwater plot
- **Why:** Often more important than volatility for risk management

### **6. Risk Metrics Dashboard** ✨ NEW
- **What:** 13 comprehensive risk-adjusted metrics
- **Use Case:** Complete risk assessment beyond Sharpe
- **Key Metrics:**
  - Sharpe (total risk)
  - Sortino (downside risk)
  - Calmar (CAGR/MaxDD)
  - VaR 95%, 99% (Value at Risk)
  - CVaR 95%, 99% (Expected Shortfall)
  - Skewness, Kurtosis
- **Why:** Different metrics reveal different risk dimensions

### **7. Rolling Metrics** ✨ NEW
- **What:** Time-varying risk metrics (configurable window)
- **Use Case:** Regime detection, identify changing correlations
- **Key Output:** Rolling Sharpe, Sortino, volatility, correlation
- **Why:** Risk is not constant - track how it changes

### **8. Return Distribution** ✨ NEW
- **What:** Statistical analysis of return characteristics
- **Use Case:** Understand tail risk, test normality assumptions
- **Key Output:** Histograms, Q-Q plots, Jarque-Bera test
- **Why:** Most returns are NOT normal - fat tails matter!

### **9. Correlation Matrix** 🔗
- **What:** Correlation heatmap between assets
- **Use Case:** Diversification analysis, identify clusters
- **Key Output:** Correlation matrix (−1 to +1)

### **10. Normalized Comparison** 📊
- **What:** All prices rebased to 100 at start
- **Use Case:** Compare relative performance visually
- **Key Output:** Normalized price chart, total return %

### **11. Seasonality Analysis** 🌙
- **What:** Monthly return patterns
- **Use Case:** Identify seasonal trends
- **Key Output:** Monthly average returns, heatmap, box plots

### **12. Multi-Period Performance** ✨ NEW
- **What:** Returns across 8 time horizons
- **Use Case:** Quick performance summary, check consistency
- **Key Periods:** 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y, Since Inception
- **Why:** Look beyond recent returns - consistency matters

---

## 🎯 Quick Decision Guide

**Need to...**

| Goal | Use This Analysis |
|------|------------------|
| Generate trading signals | **Log Returns** (modeling foundation) |
| Track dollar P&L | **Cumulative Wealth (NAV)** |
| Assess downside risk | **Drawdown Analysis** |
| Compare risk-adjusted returns | **Risk Metrics Dashboard** |
| Identify regime changes | **Rolling Metrics** |
| Understand tail risk | **Return Distribution** |
| Build diversified portfolio | **Correlation Matrix** |
| Compare asset performance | **Normalized Comparison** |
| Find seasonal patterns | **Seasonality Analysis** |
| Quick performance check | **Multi-Period Performance** |
| Understand long-term trends | **Price Trends** |
| Calculate portfolio weights | **Returns Analysis (Arithmetic)** |

---

## 📊 Key Metrics Explained

### **Return Metrics:**
- **Arithmetic Return:** Simple % change (good for single period)
- **Log Return:** ln(P₂/P₁) - time-additive (good for modeling)
- **CAGR:** Geometric mean - compounded annual growth rate

### **Risk Metrics:**
- **Volatility:** Standard deviation of returns (total risk)
- **Downside Deviation:** Std dev of negative returns only
- **Max Drawdown:** Largest peak-to-trough decline

### **Risk-Adjusted Ratios:**
- **Sharpe:** Return / Total Risk
- **Sortino:** Return / Downside Risk (better for asymmetric returns)
- **Calmar:** CAGR / Max Drawdown (focuses on worst case)

### **Tail Risk Metrics:**
- **VaR 95%:** Loss exceeded only 5% of the time
- **CVaR 95%:** Average loss when VaR is exceeded (worse than VaR)

### **Distribution Metrics:**
- **Skewness:** < 0 = more large losses (bad), > 0 = more large gains (good)
- **Kurtosis:** > 0 = fat tails (more extreme events), < 0 = thin tails

---

## 🔥 Pro Tips

### **1. Never Look at Just One Metric**
✅ Always compare: Sharpe + Sortino + Calmar + Max Drawdown

### **2. Log vs Arithmetic**
- Modeling/Signals → Use **Log Returns**
- Portfolio Construction → Use **Arithmetic Returns**
- Reporting → Use **CAGR** (geometric)

### **3. Understand Your Drawdowns**
- Max DD more important than volatility for most investors
- Always check recovery time
- Current DD tells you risk right now

### **4. Check Rolling Metrics**
- Static metrics hide regime changes
- Rolling window reveals time-varying risk
- Shorter window = more reactive, longer = more stable

### **5. Test for Normality**
- Q-Q plots should show if returns are normal
- Most commodities have fat tails (high kurtosis)
- Don't trust VaR if returns aren't normal!

### **6. Multi-Period Consistency**
- Look for consistent performance across periods
- Recent outperformance may not persist
- Check 3-5 year CAGR minimum

---

## ⚙️ Configuration Options

### **Sidebar Settings:**
- **Select Assets:** Choose 1-14 commodities
- **Data Frequency:** Daily / Weekly / Monthly
- **Rolling Window:** Adjustable (for rolling metrics)
- **Initial Investment:** Set starting capital (for NAV)

### **Date Range:**
- Filter any analysis to specific time period
- Compare performance across different market regimes
- Shorter periods = higher variance

---

## 📚 Learn More

- **Full Documentation:** `docs/COMMODITIES_ENHANCEMENT_SUMMARY.md`
- **Testing Guide:** `docs/COMMODITIES_TESTING_GUIDE.md`
- **Before/After Comparison:** `docs/COMMODITIES_BEFORE_AFTER.md`

---

## 🚀 Quick Start

```bash
# 1. Fetch data (first time)
python scripts/fetch_commodities.py

# 2. Launch app
streamlit run apps/portfolio_simulator.py

# 3. Navigate to: Commodities & Metals Analytics

# 4. Select analysis type from dropdown

# 5. Choose assets and explore!
```

---

## ❓ FAQ

**Q: Which analysis should I use first?**  
A: Start with **Multi-Period Performance** for overview, then **Risk Metrics Dashboard** for details.

**Q: What's the difference between Sharpe and Sortino?**  
A: Sharpe penalizes all volatility (up and down), Sortino only penalizes downside. Sortino is better for asymmetric returns.

**Q: Why are log returns different from arithmetic returns?**  
A: Log returns are time-additive and better for modeling. Arithmetic returns are better for portfolio construction. Use both!

**Q: What's a good Sharpe ratio?**  
A: > 1.0 is good, > 2.0 is excellent, > 3.0 is exceptional (or suspicious - check for data issues).

**Q: What's a good Calmar ratio?**  
A: > 0.5 is decent, > 1.0 is good, > 2.0 is excellent.

**Q: How do I interpret Max Drawdown?**  
A: It's the largest loss from peak. -20% DD means at one point you lost 20% from the highest value. Smaller (less negative) is better.

**Q: What does negative skewness mean?**  
A: More large losses than large gains - asymmetric downside risk. Not desirable for long positions.

**Q: What does high kurtosis mean?**  
A: Fat tails - more extreme events than normal distribution predicts. Higher crash risk and extreme gains.

---

**Last Updated:** February 3, 2026  
**Version:** 2.0 (Enhanced)  
**Status:** ✅ Production Ready
