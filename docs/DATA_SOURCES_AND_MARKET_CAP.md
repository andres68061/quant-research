# Data Sources & Market Cap Strategy

## Current Data Sources

### 1. **Stock Prices** (Primary Source: Yahoo Finance via `yfinance`)
- **Location**: `data/factors/prices.parquet`
- **Update Mechanism**: `scripts/update_daily.py`
- **Coverage**: 508 stocks currently
- **Update Frequency**: Daily (incremental - only fetches new dates)
- **API**: FREE, unlimited for price data
- **Data Fields**: Open, High, Low, Close, Volume, Adjusted Close

**How it works:**
```python
# In src/data/stock_data.py
import yfinance as yf
ticker = yf.Ticker("AAPL")
hist = ticker.history(period="max")  # Full history
```

### 2. **Commodities** (Alpha Vantage + Yahoo Finance ETFs)
- **Location**: `data/commodities/prices.parquet`
- **Update Mechanism**: `scripts/update_commodities.py`
- **Sources**:
  - Alpha Vantage: WTI Oil, Brent Oil, Natural Gas, Wheat, Corn, Coffee
  - Yahoo Finance: GLD (Gold), SLV (Silver), PPLT (Platinum), PALL (Palladium)
- **Update Frequency**: Daily automatic updates via cron
- **API Limits**: Alpha Vantage - 25 requests/day

### 3. **Economic Indicators** (FRED API)
- **Source**: Federal Reserve Economic Data (FRED)
- **API Key**: `FRED_API_KEY` in `.env`
- **Location**: `data/factors/macro.parquet`
- **Data**: Interest rates (DFF, DGS10), Inflation (CPIAUCSL), GDP, Unemployment (UNRATE)
- **Update Frequency**: Daily (data updates monthly/quarterly on FRED)
- **API**: FREE, 120 requests/minute

**How it works:**
```python
# In src/data/factors/macro.py
from fredapi import Fred
fred = Fred(api_key=FRED_API_KEY)
series = fred.get_series('CPIAUCSL')  # CPI data
```

### 4. **S&P 500 Historical Constituents** (CSV File)
- **Source**: Pre-downloaded CSV (Kaggle/Github dataset)
- **Location**: `data/S&P 500 Historical Components & Changes(01-17-2026).csv`
- **Coverage**: 1,194 unique tickers from 1996-2026
- **Update**: Manual (CSV provided by external source)

---

## Market Cap Data Strategy

### The Problem
We need **historical market cap** for:
- Cap-weighted benchmarks
- Proper S&P 500 weighting (not just equal weight)
- Avoiding survivorship bias

### Three Approaches

## ❌ **Option 1: Yahoo Finance (Current/Latest Only)**

**API:**
```python
import yfinance as yf
ticker = yf.Ticker("AAPL")
market_cap = ticker.info['marketCap']  # Latest only!
```

**Pros:**
- ✅ Free, unlimited requests
- ✅ Already integrated

**Cons:**
- ❌ Only gives **CURRENT** market cap
- ❌ No historical market cap available
- ❌ Can't reconstruct past weights

**Verdict:** ❌ **NOT suitable** - we need historical data, not just current!

---

## ✅ **Option 2: FMP (Financial Modeling Prep) - RECOMMENDED**

**Your Situation:**
- ✅ You have `FMP_API_KEY` in `.env` already!
- ✅ Free tier: **250 requests/day**
- ✅ **Includes historical market cap**

**API Endpoints:**

### A. Historical Market Cap (Best Option)
```python
import requests
import os

FMP_API_KEY = os.getenv('FMP_API_KEY')

# Get historical market cap for a ticker
url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}"
params = {'apikey': FMP_API_KEY, 'limit': 1000}
response = requests.get(url, params=params)
data = response.json()

# Returns: [{"date": "2024-01-26", "marketCap": 3000000000000}, ...]
```

### B. Company Profile (Latest Market Cap)
```python
url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
params = {'apikey': FMP_API_KEY}
response = requests.get(url, params=params)
data = response.json()[0]

# Returns: {"mktCap": 3000000000000, "price": 180.5, ...}
```

**Fetching Strategy for 1,194 Tickers:**

```
Day 1: Fetch 250 tickers (250 requests)
Day 2: Fetch 250 tickers (250 requests)
Day 3: Fetch 250 tickers (250 requests)
Day 4: Fetch 250 tickers (250 requests)
Day 5: Fetch 194 tickers (194 requests)

Total: 5 days to fetch all 1,194 tickers
```

**Implementation Plan:**
```python
# scripts/fetch_market_caps.py
import time
from datetime import datetime

def fetch_batch(tickers, batch_size=250):
    """Fetch market caps for a batch of tickers"""
    for i, ticker in enumerate(tickers):
        if i >= batch_size:
            print(f"Reached daily limit ({batch_size}). Resume tomorrow!")
            break
        
        # Fetch market cap
        market_cap = fetch_fmp_market_cap(ticker)
        save_to_parquet(ticker, market_cap)
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
```

---

## ⚠️ **Option 3: Alpha Vantage Company Overview**

**Your Situation:**
- ✅ You have `ALPHAVANTAGE_API_KEY` in `.env`
- ❌ Limit: **25 requests/day** (very restrictive!)
- ⚠️ Only gives **latest** market cap (not historical)

**API:**
```python
url = f"https://www.alphavantage.co/query"
params = {
    'function': 'OVERVIEW',
    'symbol': ticker,
    'apikey': ALPHAVANTAGE_API_KEY
}
response = requests.get(url, params=params)
data = response.json()
market_cap = data['MarketCapitalization']  # Latest only
```

**Time to fetch 1,194 tickers:**
```
1,194 tickers ÷ 25 per day = 48 days! ❌
```

**Verdict:** ❌ **NOT recommended** - Too slow and no historical data

---

## 🎯 **Recommended Solution: FMP Historical Market Cap**

### Implementation Steps:

1. **Create Market Cap Fetcher** (`src/data/market_caps.py`)
2. **Create Fetch Script** (`scripts/fetch_market_caps_fmp.py`)
3. **Store in Parquet** (`data/market_caps/historical.parquet`)
4. **Update Portfolio Simulator** to use cap-weighted S&P 500

### Storage Format:
```
data/market_caps/
├── historical.parquet    # Long format: date, ticker, market_cap
└── latest.parquet        # Wide format: ticker, market_cap (latest)
```

### Data Structure:
```
date        ticker  market_cap
2024-01-01  AAPL    3000000000000
2024-01-01  MSFT    2800000000000
2024-01-02  AAPL    3010000000000
...
```

---

## Your Questions Answered

### Q1: "Will price history follow the same daily automatic update logic?"
**A:** ✅ **YES!** Once fetched via `add_symbol.py`, all tickers are automatically added to:
- `data/factors/prices.parquet` (new columns)
- Daily updates via `scripts/update_daily.py` (incremental fetch)
- Cron job at 6 PM daily

### Q2: "We should be transparent in how many are successfully fetched"
**A:** ✅ **DONE!** The fetch script shows:
- Success count: `0/10 successful`
- Failed symbols list
- Reason for failure (delisted/invalid/no data)

### Q3: "If we fetch from yfinance, will it give latest market cap instead of historical?"
**A:** ❌ **YES** - Yahoo Finance `ticker.info['marketCap']` only gives **current/latest** market cap, not historical series.

### Q4: "Which one do we need?"
**A:** We need **HISTORICAL market cap** to:
- Reconstruct past S&P 500 weights (cap-weighted benchmark)
- Know that AAPL was 2% of index in 2010, not 7% like today
- Avoid look-ahead bias (using today's weights for past backtests)

### Q5: "What do you mean I already have Alpha Vantage company overview?"
**A:** You have `ALPHAVANTAGE_API_KEY` in your `.env` file! This gives access to their Company Overview endpoint, which includes market cap. BUT: 
- Only 25 requests/day (too slow)
- Only current market cap (not historical)

### Q6: "How should the API request be so we get all data in a few days considering we have 250 requests per day"
**A:** See detailed FMP strategy above. Key points:
- Use `historical-market-capitalization` endpoint
- Fetch 250 tickers/day
- 5 days total for all 1,194 tickers
- Store results incrementally to resume if interrupted

---

## Next Steps

Would you like me to:
1. ✅ **Continue fetching S&P 500 prices** (some will fail for delisted companies - that's expected!)
2. 🆕 **Implement FMP market cap fetcher** (5-day batch process)
3. 🆕 **Add cap-weighted S&P 500 benchmark** (using market cap data)

Let me know which you'd like to prioritize!

