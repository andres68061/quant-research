# Portfolio Simulation Issues & Fixes

## Current Problems

### 1. **Look-Ahead Bias in Factor Calculation**
**Problem:** Factors are calculated using ALL historical data, including future data.
- User selects date range: 2020-2024
- But factors are calculated from 1927-2026 (entire history)
- This creates look-ahead bias!

**Fix:** Calculate factors ONLY from start_date onwards.

### 2. **Mid-Period IPOs Not Handled**
**Problem:** If stock starts trading mid-quarter, it immediately gets included.
- Stock IPOs on Feb 15
- Rebalancing is monthly (end of month)
- Current logic: Stock could be in portfolio on Feb 15!

**Fix:** Only include stocks on rebalancing dates.

### 3. **Delistings Not Handled**
**Problem:** Delisted stocks just disappear from returns.
- Stock delisted on June 15
- Last trading day price: $5
- Current logic: Position just vanishes (no sell recorded)

**Fix:** Sell on last available trading date.

### 4. **No Portfolio Value Tracking**
**Problem:** Returns are calculated but no actual portfolio value.
- Can't see "$100 grew to $150"
- Can't properly handle cash waiting for reinvestment

**Fix:** Track portfolio value starting at $100.

---

## Proposed Solution

### Flow:

```
1. User selects start_date and end_date
   ↓
2. Filter price data to start_date onwards
   ↓
3. Calculate factors ONLY on available data (no look-ahead!)
   ↓
4. On each rebalance_date:
   a. Get universe of stocks with data at rebalance_date
   b. Calculate factor ranks on that date
   c. Select top/bottom % based on ranks
   d. Calculate equal weights (1/N for long, -1/N for short)
   e. Assign positions
   ↓
5. Between rebalancing:
   a. Hold positions
   b. If stock delists → sell at last price, move to cash
   c. Cash earns 0% (or risk-free rate)
   ↓
6. On next rebalance_date:
   a. Sell old positions
   b. Reinvest cash + proceeds into new positions
   c. Calculate turnover and transaction costs
   ↓
7. Track portfolio value:
   - Start: $100
   - Daily: $100 × (1 + r_1) × (1 + r_2) × ... × (1 + r_t)
```

### Example Timeline:

```
2020-01-31 (Rebalance):
  Available stocks: ['AAPL', 'MSFT', 'GOOGL', ..., 500 stocks]
  Calculate momentum on these 500
  Select top 20%: ['AAPL', 'TSLA', 'NVDA', ...]
  Assign weights: 1/100 each = 1% each
  Portfolio value: $100

2020-02-01 to 2020-02-29 (Hold):
  Daily returns applied to positions
  If 'TSLA' delisted on Feb 15:
    - Sell TSLA at last price
    - Move proceeds to cash (earns 0%)
    - Cash = 1% of portfolio
  Portfolio value: $100 × (1 + daily_returns)

2020-02-29 (Rebalance):
  Available stocks: ['AAPL', 'MSFT', 'GOOGL', ..., 499 stocks]
                    (TSLA missing if delisted)
  Calculate momentum on these 499
  Select top 20%: ['AAPL', 'NVDA', 'AMD', ...]
  Sell old positions (except cash)
  Reinvest all capital (including cash from TSLA)
  Assign new weights: 1/100 each = 1% each
  Calculate turnover = sum(|old_weight - new_weight|) / 2
  Apply transaction costs
```

---

## Implementation Changes Needed

### 1. Add Portfolio Value Tracking

```python
def calculate_portfolio_returns_v2(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    rebalance_freq: str = "M",
    transaction_cost: float = 0.001,
    initial_value: float = 100.0,
) -> pd.DataFrame:
    """
    Calculate portfolio returns with proper handling of:
    - Look-ahead bias prevention
    - Mid-period IPOs
    - Delistings
    - Portfolio value tracking
    """
    # Filter to start_date onwards
    prices = prices[prices.index >= start_date]
    signals = signals[signals.index.get_level_values('date') >= start_date]
    
    # Calculate returns
    returns = prices.pct_change()
    
    # Track portfolio value
    portfolio_value = initial_value
    
    # ... implementation ...
```

### 2. Handle Delistings

```python
def handle_delistings(positions, returns, date):
    """
    Check if any holdings delisted today.
    If delisted: sell at last available price, move to cash.
    """
    for symbol in positions[date]:
        if symbol not in returns.columns or pd.isna(returns.loc[date, symbol]):
            # Stock delisted - check if we have a last price
            if date > 0 and not pd.isna(returns.loc[date-1, symbol]):
                # Sell at yesterday's price (last available)
                proceed = positions.loc[date-1, symbol] * (1 + returns.loc[date-1, symbol])
                cash += proceed
                positions.loc[date, symbol] = 0
```

### 3. Rebalancing Logic

```python
def rebalance_portfolio(
    date,
    prices,
    factors,
    factor_col,
    top_pct,
    bottom_pct,
    portfolio_value,
    cash
):
    """
    Rebalance portfolio on this date.
    
    Steps:
    1. Get universe (stocks with data on this date)
    2. Calculate factor ranks
    3. Select top/bottom percentiles
    4. Calculate weights
    5. Sell old positions
    6. Buy new positions
    7. Calculate costs
    """
    # Get available stocks
    available = prices.loc[date].dropna().index
    
    # Get factors for available stocks
    date_factors = factors.xs(date, level='date')
    date_factors = date_factors[date_factors.index.isin(available)]
    
    # Rank and select
    ranks = date_factors[factor_col].rank(ascending=False)
    n_stocks = len(ranks)
    n_long = int(n_stocks * top_pct)
    n_short = int(n_stocks * bottom_pct)
    
    longs = ranks[ranks <= n_long].index
    shorts = ranks[ranks > (n_stocks - n_short)].index
    
    # Calculate weights
    total_capital = portfolio_value + cash
    weight_long = 1.0 / len(longs) if len(longs) > 0 else 0
    weight_short = -1.0 / len(shorts) if len(shorts) > 0 else 0
    
    # Assign positions
    new_positions = pd.Series(0.0, index=available)
    new_positions[longs] = weight_long
    new_positions[shorts] = weight_short
    
    return new_positions
```

### 4. Streamlit UI Changes

**Current:**
```python
# User selects date range at the END
date_range = st.sidebar.date_input(...)
```

**Should be:**
```python
# User selects date range at the START (before factor selection)
st.sidebar.header("1️⃣ Date Range Selection")
start_date = st.sidebar.date_input("Start Date", ...)
end_date = st.sidebar.date_input("End Date", ...)

st.sidebar.header("2️⃣ Strategy Configuration")
# ... factor selection, etc. ...

# Note: Factors will be calculated from start_date onwards!
st.info(f"Factors will be calculated from {start_date} onwards (no look-ahead bias)")
```

---

## Starting Portfolio Value

**Question:** "btw we simulate our portfolios to have a starting value of 100 right?"

**Answer:** Yes! Industry standard is:
- **$100** - most common (easy percentages)
- **$1** - also common (simple returns)
- **$10,000** - sometimes used (realistic portfolio size)

**Current Implementation:**
- ❌ No portfolio value tracking
- ✅ Just returns (which can be converted to $100 → $X)

**Should implement:**
```python
# Start with $100
portfolio_value = pd.Series(index=dates)
portfolio_value[0] = 100.0

# Each day: compound returns
for t in range(1, len(dates)):
    portfolio_value[t] = portfolio_value[t-1] * (1 + net_return[t])

# Result: $100 → $X over time
```

This makes it easy to say: "$100 invested grew to $250 (150% total return)"

---

## Priority Fixes

1. **HIGH:** Add date selection as FIRST step in UI
2. **HIGH:** Calculate factors from start_date onwards (no look-ahead)
3. **MEDIUM:** Handle delistings properly (sell on last date)
4. **MEDIUM:** Track portfolio value ($100 start)
5. **LOW:** Handle mid-period IPOs (already handled by rebalancing logic)

---

## Next Steps

1. Review and approve approach
2. Implement portfolio_v2 with proper logic
3. Update Streamlit UI to prioritize date selection
4. Add portfolio value visualization
5. Test with known scenarios (e.g., 2008 financial crisis with Lehman)
