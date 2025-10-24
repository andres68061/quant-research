# Quick Reference Guide

## Daily Commands

### Update Data (Weekly/Monthly)
```bash
python scripts/update_daily.py
```
**What it does**: Fetches only new data since last update  
**When to run**: Weekly/monthly, or via cron job  
**Output**: Updated Parquet files with new dates

### Add New Stock(s)
```bash
python scripts/add_symbol.py NVDA
python scripts/add_symbol.py TSLA COIN PLTR  # Multiple at once
```
**What it does**: Fetches full history for new symbol(s)  
**When to use**: Adding stocks to your universe  
**Output**: Updated prices.parquet with new column(s)

### Full Backfill (Initial Setup)
```bash
python scripts/backfill_all.py --years 10
```
**What it does**: Fetches full history for all S&P 500 stocks  
**When to run**: First time setup, or when adding new fundamentals  
**Output**: Creates all Parquet files from scratch

## File Locations

### Data Files
```
data/factors/
├── prices.parquet          # Wide format: date × symbols
├── factors_price.parquet   # Long format: price factors
├── factors_all.parquet     # Combined factors
├── macro.parquet           # Raw macro indicators
└── macro_z.parquet         # Standardized macro
```

### Cache Files (Auto-managed)
```
data/.cache/
└── fmp/                    # FMP fundamentals by year
```

### Scripts
```
scripts/
├── backfill_all.py         # Initial setup / full refresh
├── update_daily.py         # Incremental updates
└── add_symbol.py           # Add new stocks
```

## Python Usage

### Load Prices
```python
import pandas as pd

# Read from Parquet
prices = pd.read_parquet('data/factors/prices.parquet')
print(prices.shape)  # (25521, 504)
```

### Query with DuckDB
```python
import duckdb

con = duckdb.connect('data/factors/factors.duckdb')

# Get prices for specific symbol
aapl = con.sql("""
    SELECT * FROM prices 
    WHERE symbol = 'AAPL' 
    AND date >= '2024-01-01'
""").df()

# Get factors for analysis
factors = con.sql("""
    SELECT * FROM factors_all 
    WHERE date >= '2024-01-01'
    ORDER BY date, symbol
""").df()
```

### Incremental Update Programmatically
```python
from pathlib import Path
from src.utils.io import read_parquet
from src.data.factors.prices import update_prices_panel_incremental

# Load existing
prices = read_parquet(Path('data/factors/prices.parquet'))

# Update with new data
updated_prices = update_prices_panel_incremental(prices)

# Save
from src.utils.io import write_parquet
write_parquet(updated_prices, Path('data/factors/prices.parquet'))
```

## Cron Jobs

### Daily Update (6 PM)
```bash
0 18 * * * cd /path/to/quant && /opt/anaconda3/envs/quant/bin/python scripts/update_daily.py >> logs/update.log 2>&1
```

### Weekly Backup (Sunday 2 AM)
```bash
0 2 * * 0 cd /path/to/quant && tar -czf backups/factors_$(date +\%Y\%m\%d).tar.gz data/factors/*.parquet
```

## Troubleshooting

### "No existing prices found"
```bash
# Run initial backfill first
python scripts/backfill_all.py --years 10
```

### Timezone Warning
Already fixed! Prices are timezone-aware, script handles it.

### Symbol Not Found
Check that symbol is valid on Yahoo Finance:
```python
import yfinance as yf
ticker = yf.Ticker('AAPL')
print(ticker.info)
```

### Out of Date Cache
```bash
# Force refresh FMP cache
python scripts/backfill_all.py --refresh-cache
```

## Common Patterns

### Get Last N Days
```python
import pandas as pd
prices = pd.read_parquet('data/factors/prices.parquet')
last_30_days = prices.iloc[-30:]
```

### Filter by Symbols
```python
symbols_of_interest = ['AAPL', 'MSFT', 'GOOGL']
subset = prices[symbols_of_interest]
```

### Merge Prices with Factors
```python
prices_long = prices.stack().to_frame('close')
prices_long.index.names = ['date', 'symbol']

factors = pd.read_parquet('data/factors/factors_price.parquet')
merged = prices_long.join(factors, how='inner')
```

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│         USER WORKFLOWS                   │
├─────────────────────────────────────────┤
│                                          │
│  Initial Setup                           │
│  backfill_all.py  → Full History        │
│                                          │
│  Regular Updates                         │
│  update_daily.py  → New Dates Only      │
│                                          │
│  Add Stocks                              │
│  add_symbol.py    → Full History         │
│                                          │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│         DATA STORAGE                     │
├─────────────────────────────────────────┤
│                                          │
│  Parquet Files (Source of Truth)        │
│  - prices.parquet                        │
│  - factors_price.parquet                 │
│  - factors_all.parquet                   │
│  - macro.parquet                         │
│                                          │
│  DuckDB (Query Interface)                │
│  - SQL views over Parquet                │
│                                          │
│  Cache (Rate Limit Protection)           │
│  - data/.cache/fmp/                      │
│                                          │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│         ANALYSIS                         │
├─────────────────────────────────────────┤
│                                          │
│  Jupyter Notebooks                       │
│  Python Scripts                          │
│  DuckDB Queries                          │
│  Pandas/NumPy Analysis                   │
│                                          │
└─────────────────────────────────────────┘
```

## Next Steps

1. **Read Full Documentation**: [INCREMENTAL_UPDATES.md](INCREMENTAL_UPDATES.md)
2. **Explore Data**: Open `notebooks/04_browse_databases.ipynb`
3. **Set Up Cron**: Automate weekly updates
4. **Add Your Stocks**: Use `add_symbol.py` for your watchlist
5. **Build Strategies**: Start quantamental research!

