# Ratio Analysis Feature Guide

## 📊 New Feature: Commodity Ratio Analysis

**Added:** February 3, 2026  
**Version:** 2.1  
**Analysis Type:** #9 of 13

---

## 🎯 What Is It?

**Ratio Analysis** calculates and visualizes the price ratio between commodities, revealing relative value and trading opportunities.

**Classic Example:** Gold/Silver Ratio
- Historical average: 50-80
- Used by traders for centuries
- Mean reversion opportunities when extreme

---

## 🔧 How It Works

### **Scenario 1: Two Commodities Selected** ✨ SIMPLE

**Example:** Gold and Silver

**What You Get:**
1. ✅ Single ratio chart (Gold/Silver)
2. ✅ Mean line with ±1 SD bands
3. ✅ Current statistics (current, mean, min, max)
4. ✅ Distribution histogram
5. ✅ Trading signal (HIGH/LOW/NEUTRAL) with z-score
6. ✅ Interpretation guidance

**Perfect for:** Classic pairs like Gold/Silver, Oil/Gold, Copper/Gold

---

### **Scenario 2: Three Commodities Selected** 📊 AUTOMATIC

**Example:** Gold, Silver, Platinum

**What You Get:**
1. ✅ All 3 possible ratios displayed:
   - Gold/Silver
   - Gold/Platinum
   - Silver/Platinum
2. ✅ Normalized chart (base=100) for comparison
3. ✅ Statistics table for all 3 ratios
4. ✅ Z-scores for each ratio

**Perfect for:** Precious metals suite, energy suite

---

### **Scenario 3: Four+ Commodities Selected** 🎛️ INTERACTIVE

**Example:** Gold, Silver, Platinum, Palladium, Copper

**What You Get:**
1. ✅ **Select which pair** to analyze (dropdown menus)
2. ✅ Full analysis for selected pair:
   - Ratio chart with bands
   - Statistics
   - Distribution
   - Z-score
3. ✅ **Summary table** showing all possible ratios with current z-scores

**Perfect for:** Large portfolios, exploratory analysis

---

## 📈 Key Features

### **1. Mean Reversion Signals**

The system automatically calculates:
- **Mean ratio** (historical average)
- **Standard deviation** (volatility of ratio)
- **Z-score** (how many SDs from mean)

**Trading Signals:**
- **Z > +1**: Ratio is HIGH → Consider mean reversion DOWN
- **Z < -1**: Ratio is LOW → Consider mean reversion UP
- **|Z| < 1**: NEUTRAL → No strong signal

### **2. Visual Indicators**

**Chart Elements:**
- 🔵 **Blue line**: Actual ratio
- ⚪ **Gray dashed**: Mean (historical average)
- 🟢 **Green dotted**: +1 Standard Deviation
- 🔴 **Red dotted**: -1 Standard Deviation

**Distribution:**
- Shows full historical range
- Mean line for reference
- ±1 SD bands

### **3. Status Interpretation**

**HIGH Status** (Z > +1):
```
📈 Ratio is HIGH
Numerator is expensive relative to denominator
→ Mean reversion trade: Long denominator, Short numerator
→ Or momentum trade: Numerator outperforming
```

**LOW Status** (Z < -1):
```
📉 Ratio is LOW
Numerator is cheap relative to denominator
→ Mean reversion trade: Long numerator, Short denominator
→ Or momentum trade: Denominator outperforming
```

**NEUTRAL Status** (|Z| < 1):
```
⚖️ Ratio is NEUTRAL
Near historical average
→ No strong signal, wait for extremes
```

---

## 💡 Use Cases

### **1. Gold/Silver Ratio** (Classic)

**Typical Range:** 50-80

**Interpretation:**
- **Ratio = 80+**: Silver is cheap → Buy silver
- **Ratio = 50-**: Gold is cheap → Buy gold

**Why It Works:**
- Both are precious metals
- Supply/demand dynamics linked
- Historical mean reversion

### **2. Oil/Gold Ratio** (Inflation Indicator)

**What It Shows:**
- Energy prices vs monetary metal
- Inflation expectations
- Real vs nominal trends

**Use:**
- High ratio: Inflation concerns (oil expensive)
- Low ratio: Deflation concerns (gold expensive)

### **3. Copper/Gold Ratio** (Economic Growth)

**What It Shows:**
- Industrial vs safe-haven demand
- Economic growth expectations
- Risk-on vs risk-off

**Use:**
- Rising ratio: Economic expansion (copper demand ↑)
- Falling ratio: Economic slowdown (gold demand ↑)

### **4. Energy Spreads**

**Examples:**
- WTI/Brent (arbitrage opportunities)
- Oil/Natural Gas (energy substitution)

### **5. Agricultural Spreads**

**Examples:**
- Wheat/Corn (crop substitution)
- Coffee/Sugar (tropical commodities)

---

## 📊 Example Walkthrough

### **Example: Gold/Silver Analysis**

**Step 1: Select Commodities**
- ✅ Gold (GLD)
- ✅ Silver (SLV)

**Step 2: Choose "Ratio Analysis"**

**What You See:**

```
Current Ratio: 75.50
Mean: 70.20
Z-Score: +1.25

Status: 📈 RATIO IS HIGH

Interpretation:
Gold is expensive relative to silver

Trading Ideas:
- Mean reversion: Long silver, short gold
- Momentum: Gold outperforming, ride the trend
```

**Chart Shows:**
- Ratio fluctuating between 60-90 over time
- Currently above +1 SD (mean reversion opportunity)
- Historical context for decision making

---

## 🎓 Educational Content

Included in every analysis:

### **"Understanding Commodity Ratios" Expander**

Covers:
- ✅ What ratios are
- ✅ Famous ratios (Gold/Silver, Oil/Gold, Copper/Gold)
- ✅ Mean reversion strategy
- ✅ Momentum strategy
- ✅ Z-score interpretation
- ✅ Limitations and warnings

---

## ⚙️ Configuration

**No configuration needed!**

The system automatically:
- Detects number of commodities
- Chooses appropriate display mode
- Calculates statistics
- Provides signals

**For 4+ commodities:**
- Use dropdown menus to select pair
- Change selection anytime
- All calculations update instantly

---

## 🔥 Pro Tips

### **1. Use with Other Analyses**

**Workflow:**
1. Check **Correlation Matrix** first
2. Find highly correlated pairs
3. Analyze ratio for mean reversion
4. Check **Rolling Metrics** for regime changes

### **2. Combine with Fundamentals**

Ratios are statistical, not fundamental:
- ✅ High z-score + fundamental support = strong signal
- ⚠️ High z-score + fundamental divergence = caution

### **3. Time Frames Matter**

- **Short-term** (1 year): Noisy, many false signals
- **Medium-term** (3-5 years): Good for mean reversion
- **Long-term** (10+ years): Structural shifts visible

### **4. Not All Ratios Mean Revert**

Some ratios trend for years:
- Check distribution shape
- Look for bimodality (two regimes)
- Consider structural changes

### **5. Use Z-Score Thresholds**

Conservative approach:
- Enter: |Z| > 2.0 (very extreme)
- Exit: |Z| < 0.5 (near mean)

Aggressive approach:
- Enter: |Z| > 1.0
- Exit: |Z| < 0

---

## 📋 Technical Details

### **Calculation:**

```python
ratio = price_numerator / price_denominator
mean_ratio = ratio.mean()
std_ratio = ratio.std()
z_score = (current_ratio - mean_ratio) / std_ratio
```

### **Statistical Bands:**

- **Mean**: Historical average
- **+1 SD**: Mean + 1 standard deviation (~84th percentile)
- **-1 SD**: Mean - 1 standard deviation (~16th percentile)

### **Normalization (for 3 commodities):**

```python
normalized_ratio = (ratio / ratio.iloc[0]) * 100
```

Allows comparison of ratios with different scales.

---

## ⚠️ Limitations

### **1. Data Availability**

Ratios require overlapping data:
- Both commodities must have prices on same dates
- System filters to valid overlap automatically
- May reduce available history

### **2. Mean Reversion Assumption**

Not all ratios mean revert:
- Structural changes can create new regimes
- Backtesting historical mean may not predict future
- Use alongside fundamental analysis

### **3. Statistical vs Fundamental**

Z-scores are purely statistical:
- Don't capture supply/demand shifts
- Miss regulatory changes
- Ignore technological disruptions

### **4. Transaction Costs**

Real trading involves:
- Bid-ask spreads
- Futures roll costs (if using futures)
- Slippage
- Consider these in actual implementation

---

## 🚀 Getting Started

### **Quick Start:**

1. **Navigate to Commodities page**
2. **Select 2 commodities** (e.g., Gold, Silver)
3. **Choose "Ratio Analysis"** from dropdown
4. **Read the signals** (HIGH/LOW/NEUTRAL)
5. **Check distribution** for historical context

### **Advanced:**

1. **Select 4+ commodities**
2. **Choose "Ratio Analysis"**
3. **Use dropdowns** to explore different pairs
4. **Check summary table** for all z-scores
5. **Identify extremes** (|Z| > 2)
6. **Cross-reference** with fundamentals

---

## 📚 Related Features

**Complement Ratio Analysis with:**

- **Correlation Matrix**: Find related commodities
- **Rolling Metrics**: Detect regime changes
- **Normalized Comparison**: Visual relative performance
- **Multi-Period Performance**: Confirm trends across horizons

---

## 💬 FAQ

**Q: What's a good z-score for trading?**  
A: |Z| > 1.5 is interesting, |Z| > 2.0 is strong signal. But always verify with fundamentals!

**Q: How often should ratios mean revert?**  
A: Historically, ~60-70% of extremes revert within 3-6 months. Not guaranteed!

**Q: Can I use this for any two commodities?**  
A: Yes, but works best for related commodities (same sector, substitutes, or economic relationships).

**Q: What if the ratio keeps going up?**  
A: Structural changes can create new regimes. Mean from past 5 years may not be relevant anymore.

**Q: Should I always fade extremes?**  
A: No! Check fundamentals. An extreme ratio might be signaling a structural shift, not a temporary deviation.

**Q: How do I know if a ratio mean reverts?**  
A: Check the distribution - should be roughly bell-shaped. Wide, flat distributions don't mean revert well.

---

## 📊 Summary

**Ratio Analysis is perfect for:**
- ✅ Identifying relative value
- ✅ Finding mean reversion opportunities
- ✅ Understanding cross-commodity dynamics
- ✅ Classic pairs trading (Gold/Silver, etc.)
- ✅ Economic indicator analysis (Copper/Gold)

**Scales intelligently:**
- 2 commodities → Simple, focused analysis
- 3 commodities → All ratios shown
- 4+ commodities → Interactive selection

**Professional features:**
- Statistical bands (mean, ±SD)
- Z-score signals
- Distribution analysis
- Trading interpretations
- Educational content

---

**Status:** ✅ **PRODUCTION READY**

**Added:** February 3, 2026  
**Total Analysis Types:** 13  
**Lines of Code:** ~500 lines for Ratio Analysis

**Try it now!** Select Gold and Silver on the Commodities page and see the famous Gold/Silver ratio in action! 🚀
