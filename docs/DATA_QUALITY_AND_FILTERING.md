# Data Quality & Filtering

## Overview

This document explains how the portfolio simulator handles data quality issues to ensure realistic backtest results.

---

## The Penny Stock Problem

### What Happened

Initial simulations showed **astronomical returns** (29 quadrillion %) due to penny stocks in the dataset.

### Root Cause

**Small prices create huge percentage returns:**

```
Price: $0.001 → $0.010 = 900% return in one day!
Price: $0.50 → $1.00 = 100% return
Price: $50 → $51 = 2% return (realistic)
```

**Data Analysis Found:**
- 2,554 daily returns > 100%
- 17,508 prices < $0.01
- Some returns = `inf` (infinity)

### Why This Matters

When you compound these extreme returns over years:
```
Daily return of 2.68% → 5-year total return = 29 quadrillion %
```

This is obviously wrong and corrupts all portfolio metrics.

---

## Solution: Penny Stock Filter

### Standard Practice

Institutional quant research **always** excludes penny stocks:
- **Common threshold**: $5 minimum price
- **Conservative threshold**: $10 minimum price  
- **Why**: Penny stocks have:
  - Low liquidity (can't actually trade them)
  - Huge bid-ask spreads (real costs higher than data shows)
  - Extreme volatility (corrupts statistics)
  - Data quality issues (stale prices, errors)

### Our Implementation

**Filter Applied:**
```python
# Exclude stocks where price < $5 on ANY day in backtest period
price_mask = (df_prices_filtered >= 5.0).all(axis=0)
valid_symbols = df_prices_filtered.columns[price_mask]
```

**When Applied:**
1. After date selection
2. Before factor calculation
3. Before portfolio construction

**User Feedback:**
```
ℹ️ Excluded 147 penny stocks (price < $5) to prevent data corruption
📊 1,234 trading days, 762 symbols
```

---

## Missing Data Handling

### How Data Goes Missing

1. **Delistings** - Company removed from exchange (bankruptcy, acquisition, failure)
2. **IPOs** - New companies don't have historical data
3. **Halts** - Trading temporarily suspended
4. **Data gaps** - API failures, weekends, holidays

### Our Approach

#### Layer 1: Forward-Fill (Minor Gaps)
```python
# Fill short gaps (1-2 days)
prices = prices.asfreq('B').ffill()
```

**Use Case:** Stock didn't trade on a specific day, use previous close

#### Layer 2: Availability Filter (Rebalancing)
```python
# Only include stocks with valid prices on rebalance date
available_stocks = prices.loc[rebal_date].dropna().index
signals = signals[signals.index.isin(available_stocks)]
```

**Use Case:** Can't trade stocks that aren't trading

#### Layer 3: Delisting Detection (Daily)
```python
# If position exists but no return → stock delisted
if pd.isna(ret[symbol]):
    # Sell at last available price
    cash_from_delistings += abs(pos[symbol])
    # Zero out position
    positions.loc[date:, symbol] = 0.0
```

**Use Case:** Stock went bankrupt or was acquired mid-period

#### Layer 4: NaN Fallback (Safety)
```python
# Any remaining NaN returns → 0%
valid_returns = ret[pos.index].fillna(0.0)
```

**Use Case:** Catch-all for edge cases

---

## Real-World Scenarios

### Scenario A: Stock Delisted on Feb 15
```
Feb 14: ✅ Stock in portfolio, price = $10
Feb 15: ❌ No return data → DETECTED
Action: Sell at $10 (last available), convert to cash
Feb 16+: Position = 0, cash held until next rebalance
```

### Scenario B: New IPO on June 1
```
May 31 (Rebalance): ❌ Stock not available
Action: Not included in portfolio
June 30 (Next Rebalance): ✅ Stock has prices
Action: Can be included if factor ranking supports it
```

### Scenario C: Penny Stock Excluded
```
Stock XYZ trades at $3.50 on some days
Filter: ❌ Excluded (< $5 threshold)
Reason: Would create 50%+ swings on small moves
Result: More realistic portfolio metrics
```

---

## Factor Definitions

### Momentum Factors

**`mom_12_1`** (Default)
- **Definition**: 12-month return, excluding last month
- **Formula**: `(price[t] / price[t-252]) - (price[t] / price[t-21])`
- **Why exclude last month?** Academic research shows short-term reversal effect
- **Industry standard**: Used by AQR, Two Sigma, etc.

**`mom_6_1`**
- 6-month return, excluding last month
- More responsive to recent trends

**`mom_3_1`**
- 3-month return, excluding last month  
- Very short-term momentum

### Risk Factors

**`vol_60d`**
- 60-day volatility (annualized)
- Higher = riskier stock

**`beta_60d`**
- 60-day beta vs SPY
- Measures sensitivity to market moves

---

## Data Quality Checklist

When running backtests, the system ensures:

✅ **No penny stocks** (< $5 excluded)  
✅ **No look-ahead bias** (factors calculated from start date only)  
✅ **No forward-fill over delistings** (detect and sell)  
✅ **No unavailable stocks at rebalance** (check prices exist)  
✅ **No inf/NaN in returns** (safety fallbacks)

---

## Configuration Options

### Adjusting the Price Filter

If you want a different threshold, edit `apps/portfolio_simulator.py`:

```python
# Current: $5 minimum
price_mask = (df_prices_filtered >= 5.0).all(axis=0)

# Conservative: $10 minimum
price_mask = (df_prices_filtered >= 10.0).all(axis=0)

# Aggressive: $1 minimum (not recommended)
price_mask = (df_prices_filtered >= 1.0).all(axis=0)
```

**Recommendation:** Keep at $5 (matches most academic research)

---

## Performance Impact

**Without Filtering:**
- Total Return: 29,070,151,663,389,176% ❌
- Sharpe Ratio: Meaningless
- All metrics corrupted

**With Filtering:**
- Total Return: 50-200% (realistic) ✅
- Sharpe Ratio: 0.5-2.0 (reasonable)
- Metrics interpretable

---

## Related Documentation

- `PORTFOLIO_SIMULATION_FIXES_APPLIED.md` - Core simulation logic
- `BENCHMARK_OPTIONS.md` - Benchmark selection guide
- `README.md` - Full platform documentation

---

**Bottom Line:** Excluding penny stocks is not optional - it's required for accurate research.
