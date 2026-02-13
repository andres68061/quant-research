# Commodity Data Availability Guide

## 🗓️ Understanding Data Start Dates

Different commodities have different data availability periods. This guide helps you select appropriate date ranges for analysis.

---

## 📊 Commodity Data Timeline

### **Precious Metals (Yahoo Finance ETFs)**
All available with extensive historical data:

| Symbol | Name | Source | Approx. Start Date | Data Quality |
|--------|------|--------|-------------------|--------------|
| GLD | Gold (SPDR ETF) | Yahoo Finance | **2004-11** | ✅ Excellent |
| SLV | Silver (iShares ETF) | Yahoo Finance | **2006-04** | ✅ Excellent |
| PPLT | Platinum (Aberdeen ETF) | Yahoo Finance | **2010-01** | ✅ Good |
| PALL | Palladium (Aberdeen ETF) | Yahoo Finance | **2010-01** | ✅ Good |

### **Energy Commodities (Alpha Vantage)**

| Symbol | Name | Source | Approx. Start Date | Data Quality |
|--------|------|--------|-------------------|--------------|
| WTI | Crude Oil (WTI) | Alpha Vantage | **2000+** | ✅ Good |
| BRENT | Crude Oil (Brent) | Alpha Vantage | **2000+** | ✅ Good |
| NATURAL_GAS | Natural Gas | Alpha Vantage | **2000+** | ⚠️ Variable |

### **Industrial Metals (Alpha Vantage)**

| Symbol | Name | Source | Approx. Start Date | Data Quality |
|--------|------|--------|-------------------|--------------|
| COPPER | Copper | Alpha Vantage | **Varies** | ⚠️ Check availability |
| ALUMINUM | Aluminum | Alpha Vantage | **Varies** | ⚠️ Check availability |

### **Agricultural Commodities (Alpha Vantage)**

| Symbol | Name | Source | Approx. Start Date | Data Quality |
|--------|------|--------|-------------------|--------------|
| WHEAT | Wheat | Alpha Vantage | **Varies** | ⚠️ Check availability |
| CORN | Corn | Alpha Vantage | **Varies** | ⚠️ Check availability |
| COFFEE | Coffee | Alpha Vantage | **Varies** | ⚠️ Check availability |
| COTTON | Cotton | Alpha Vantage | **Varies** | ⚠️ Check availability |
| SUGAR | Sugar | Alpha Vantage | **Varies** | ⚠️ Check availability |

---

## 🎯 Recommended Date Ranges

### **For Maximum Commodity Coverage:**

**Safest Range:** **2015 - Present**
- ✅ All commodities should have data
- ✅ Good data quality
- ✅ Sufficient history for analysis
- ✅ Minimal missing data issues

### **For Long-Term Analysis:**

**Conservative Range:** **2010 - Present**
- ✅ Most commodities available
- ⚠️ Some agricultural may have gaps
- ✅ Includes financial crisis recovery
- ✅ ~14 years of data

**Aggressive Range:** **2005 - Present**
- ⚠️ Only Gold, Silver, some energy
- ❌ Many commodities not yet available
- ✅ Longest possible history
- ⚠️ Expect exclusions

### **For Short-Term Analysis:**

**Recent Range:** **2020 - Present**
- ✅ All commodities available
- ✅ Includes COVID period
- ✅ Most recent market dynamics
- ⚠️ Shorter history (~4 years)

---

## ⚠️ Common Issues & Solutions

### **Issue 1: "No valid data for selected commodities"**

**Cause:** Selected date range is before commodities started trading

**Example:**
```
Selected: Copper, Wheat
Date Range: 2005-2010
Problem: These commodities may not have data before 2010
```

**Solutions:**
1. ✅ Select more recent date range (2015+)
2. ✅ Choose different commodities with longer history (Gold, Silver)
3. ✅ Check "Price Trends" first to see data availability

---

### **Issue 2: "Excluding X commodities with insufficient data"**

**Cause:** Some selected commodities don't have data in your date range

**Example:**
```
Selected: Gold, Copper, Wheat
Date Range: 2005-2015
Result: Only Gold has full data
Info: "Excluding 2 commodities: Copper, Wheat"
```

**Solutions:**
1. ✅ Analysis proceeds with available commodities (Gold)
2. ✅ Adjust date range to include all (2015+)
3. ✅ Or deselect commodities without data

---

### **Issue 3: Different commodities have different data lengths**

**Cause:** Natural - commodities started trading at different times

**Example:**
```
Date Range: 2005-2020
Gold: 15 years of data ✅
Silver: 14 years (starts 2006) ⚠️
Platinum: 10 years (starts 2010) ⚠️
```

**Impact:**
- Correlation analysis may be skewed
- Performance comparison not apples-to-apples
- Rolling metrics need sufficient overlap

**Solutions:**
1. ✅ Use "Normalized Comparison" - handles different starts
2. ✅ Select date range where ALL commodities have data (2010+)
3. ✅ Understand limitations when comparing

---

## 🔍 How to Check Data Availability

### **Method 1: Price Trends Analysis**

1. Go to **Price Trends** analysis
2. Select your commodities
3. Choose **full date range**
4. Look at the chart - you'll see when each commodity starts

**What to Look For:**
- Flat line or missing data at the beginning = not available yet
- Clear price movements = data available

### **Method 2: Check First Valid Date**

In **Price Trends** statistics table, look at the date range shown.

### **Method 3: Multi-Period Performance**

Select **Multi-Period Performance** and look for "N/A" values:
- Many N/A = limited data
- All values present = good coverage

---

## 💡 Best Practices

### **1. Start with Precious Metals**
Gold and Silver have the longest, most reliable data:
```
✅ Safe: GLD (2004+), SLV (2006+)
⚠️ Check: Everything else
```

### **2. Use Date Range Filters Wisely**

**For exploration:**
- Start with **2015 - Present** (safest)
- All analysis types will work
- No exclusions

**For historical analysis:**
- Use **2005 - Present** only with Gold/Silver
- Expect other commodities to be excluded
- Understand limitations

### **3. Test Before Deep Analysis**

**Workflow:**
1. Select commodities
2. Go to **Price Trends** first
3. See which have data in your date range
4. Adjust selection or date range
5. Then proceed to advanced analysis

### **4. Document Your Date Ranges**

When reporting results, always note:
- Date range used
- Which commodities included
- Any exclusions
- Data availability limitations

---

## 📈 Analysis-Specific Recommendations

### **Log Returns Analysis**
- **Minimum:** 2 data points (but realistically need 100+)
- **Recommended:** 1+ years of data
- **Best:** 3+ years for meaningful patterns

### **Drawdown Analysis**
- **Minimum:** 2 data points
- **Recommended:** 2+ years to capture full drawdown cycles
- **Best:** 5+ years to see recovery patterns

### **Rolling Metrics**
- **Minimum:** 20 data points
- **Recommended:** Window size + 100 points
- **Best:** 3+ years for 252-day rolling window

### **Return Distribution**
- **Minimum:** 30 data points
- **Recommended:** 252+ points (1 year daily)
- **Best:** 500+ points for reliable distribution analysis

### **Multi-Period Performance**
- **Minimum:** 2 data points
- **Recommended:** Enough to cover all desired periods (1M, 3M, 6M, etc.)
- **Best:** 5+ years to show all periods

---

## 🎯 Quick Reference

**Want to analyze from 2005?**
→ Use only: Gold (GLD)

**Want to analyze from 2007?**
→ Use: Gold (GLD), Silver (SLV)

**Want to analyze from 2010?**
→ Use: Gold, Silver, Platinum, Palladium, some energy

**Want to analyze from 2015?**
→ Use: Most commodities available

**Want to analyze from 2020?**
→ Use: All commodities available

---

## 🔧 Troubleshooting Commands

```bash
# Check what data exists
ls -lh data/commodities/prices.parquet

# Update commodity data
python scripts/update_commodities.py

# Fetch fresh data
python scripts/fetch_commodities.py

# See available commodities
python scripts/fetch_commodities.py --list
```

---

## 📚 Additional Resources

- **Commodity Data Sources:** See `src/data/commodities.py` for source configuration
- **API Limits:** Yahoo Finance (free), Alpha Vantage (25 requests/day on free tier)
- **Update Frequency:** Daily (run `update_commodities.py`)

---

## 🎓 Understanding the Error Messages

### **"Excluding X commodities with insufficient data"**
- **Severity:** Info ℹ️
- **Meaning:** Some commodities don't have data in your range
- **Action:** Analysis proceeds with available commodities
- **Fix:** Adjust date range or selection

### **"No valid data for selected commodities"**
- **Severity:** Warning ⚠️
- **Meaning:** None of your selected commodities have data
- **Action:** Analysis stops
- **Fix:** Change date range or choose different commodities

### **"Insufficient data for [analysis type]"**
- **Severity:** Warning ⚠️
- **Meaning:** Not enough data points for this analysis
- **Action:** Analysis stops
- **Fix:** Select longer date range

---

**Last Updated:** February 3, 2026  
**Version:** 1.0  
**Status:** ✅ Comprehensive Guide
