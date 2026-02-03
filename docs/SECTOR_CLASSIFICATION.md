# Sector Classification System

## Overview

This system manages sector and industry classifications for all stocks in the database using Yahoo Finance data. Classifications are stored in Parquet format and automatically refreshed quarterly to minimize API calls while keeping data current.

## Data Source

**Yahoo Finance Classification** (via `yfinance`)

We use Yahoo Finance's sector/industry taxonomy instead of official GICS (Global Industry Classification Standard) because:
- ✅ Free and accessible via `yfinance` API
- ✅ Covers all stocks with reasonable accuracy
- ✅ Automatically updated by Yahoo Finance
- ✅ Good enough for research-grade analysis

**Note:** Yahoo's classification is *similar* to GICS but not identical. For official GICS codes, you would need a paid MSCI subscription.

## Data Structure

### Storage Location
```
data/sectors/sector_classifications.parquet
```

### Schema
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `symbol` | string | Stock ticker | `AAPL` |
| `sector` | string | Yahoo Finance sector | `Technology` |
| `industry` | string | Yahoo Finance industry | `Consumer Electronics` |
| `industryKey` | string | Industry key code | `consumer-electronics` |
| `sectorKey` | string | Sector key code | `technology` |
| `quoteType` | string | Asset type from Yahoo | `EQUITY`, `ETF`, `INDEX` |
| `last_updated` | string (ISO) | Last fetch timestamp | `2026-01-26T10:30:00` |

### Unknown Labels
Stocks without classification data are labeled as `"Unknown"` in all fields. This can happen for:
- Delisted stocks
- Very new IPOs
- API failures

### Quote Types
The `quoteType` field identifies the asset type:
- `EQUITY` - Regular stocks
- `ETF` - Exchange-traded funds
- `INDEX` - Market indices (^GSPC, ^IXIC, etc.)
- `MUTUALFUND` - Mutual funds
- `CRYPTOCURRENCY` - Crypto assets
- `Unknown` - Data not available

## Refresh Policy

### Initial Fetch
When a stock is first added to the database:
```bash
python scripts/add_symbol.py NVDA
```
→ Automatically fetches sector classification

### Quarterly Refresh
Sector classifications are refreshed every **90 days (3 months)** for:
- Stocks with stale data (>90 days old)
- Stocks marked as "Unknown" (retry failed fetches)

### Automatic Updates
The daily update script checks for stale sectors:
```bash
python scripts/update_daily.py
```
→ Automatically refreshes sectors >90 days old

### Manual Refresh
Force refresh all sectors:
```bash
python scripts/update_sectors.py
```

Force refresh specific symbols:
```bash
python scripts/fetch_sectors.py --symbols AAPL MSFT --force
```

## Usage

### 1. Initial Setup (First Time)

Fetch sectors for all existing stocks:
```bash
python scripts/fetch_sectors.py
```

This will:
- Read all symbols from `data/factors/prices.parquet`
- Fetch sector/industry for each symbol
- Save to `data/sectors/sector_classifications.parquet`
- Show sector breakdown

**Example Output:**
```
📊 FETCH SECTOR CLASSIFICATIONS
================================================================================

📈 Reading symbols from data/factors/prices.parquet...
   Found 909 symbols

📥 Fetching sectors for 909 new symbols...
   Progress: 10/909 symbols
   Progress: 20/909 symbols
   ...
   ✅ Fetched sector for AAPL: Technology

✅ Total symbols with classifications: 909

📈 Sector Breakdown:

   Technology                 150 (16.5%) ████████
   Financials                 120 (13.2%) ██████
   Healthcare                 110 (12.1%) ██████
   Consumer Cyclical           90 ( 9.9%) ████
   Industrials                 85 ( 9.4%) ████
   ...
```

### 2. Adding New Stocks

When adding a new stock, sector is automatically fetched:
```bash
python scripts/add_symbol.py NVDA
```

**Output includes:**
```
📊 Fetching sector classifications for new symbols...
   NVDA: Technology - Semiconductors
```

### 3. Quarterly Updates

#### Option A: Automatic (Recommended)
Run daily update script (includes quarterly sector refresh):
```bash
python scripts/update_daily.py
```

If sectors are stale (>90 days):
```
📊 Checking sector classifications...
   Found 45 symbols needing quarterly refresh
   Updating sectors (this may take a few minutes)...
   ✅ Updated 45 sector classifications
```

#### Option B: Manual
Explicitly run quarterly update:
```bash
python scripts/update_sectors.py
```

**Options:**
```bash
# Custom refresh period (60 days instead of 90)
python scripts/update_sectors.py --refresh-days 60

# Only retry Unknown symbols
python scripts/update_sectors.py --retry-unknown
```

### 4. Programmatic Access

#### Get sector for a symbol
```python
from src.data.sector_classification import get_sector_for_symbol

sector, industry = get_sector_for_symbol('AAPL')
print(f"{sector} - {industry}")
# Output: Technology - Consumer Electronics
```

#### Get all symbols in a sector
```python
from src.data.sector_classification import get_symbols_by_sector

tech_stocks = get_symbols_by_sector('Technology')
print(f"Found {len(tech_stocks)} tech stocks")
# Output: Found 150 tech stocks
```

#### Load all classifications
```python
from src.data.sector_classification import load_sector_classifications

df = load_sector_classifications()
print(df.head())
```

#### Get sector summary
```python
from src.data.sector_classification import get_sector_summary

summary = get_sector_summary()
print(summary)
#         sector  count  percentage
# 0   Technology    150       16.5
# 1   Financials    120       13.2
# ...
```

#### Add/update sectors for specific symbols
```python
from src.data.sector_classification import add_or_update_sectors

# Add new symbols
df = add_or_update_sectors(['NVDA', 'TSLA'])

# Force refresh existing symbols
df = add_or_update_sectors(['AAPL', 'MSFT'], force_refresh=True)
```

## Yahoo Finance Sectors

Common sectors you'll see:

1. **Technology** - Software, semiconductors, IT services
2. **Healthcare** - Pharmaceuticals, biotech, medical devices
3. **Financial Services** / **Financials** - Banks, insurance, asset management
4. **Consumer Cyclical** - Retail, automotive, leisure
5. **Consumer Defensive** - Food, beverages, household products
6. **Industrials** - Aerospace, construction, machinery
7. **Energy** - Oil & gas, renewable energy
8. **Basic Materials** - Chemicals, metals, mining
9. **Real Estate** - REITs, real estate services
10. **Communication Services** - Telecom, media, entertainment
11. **Utilities** - Electric, gas, water utilities

## Rate Limiting

To avoid hitting Yahoo Finance rate limits:

- **Default delay:** 0.5 seconds between requests
- **Batch fetching:** Processes symbols sequentially with delays
- **Quarterly refresh:** Only updates stale data (not all symbols)
- **Smart caching:** Doesn't re-fetch recent data

**Estimated time:**
- 100 symbols: ~1 minute
- 500 symbols: ~5 minutes
- 1000 symbols: ~10 minutes

## Troubleshooting

### Problem: All symbols showing as "Unknown"

**Cause:** API failures or network issues

**Solution:**
```bash
# Retry unknown symbols
python scripts/update_sectors.py --retry-unknown
```

### Problem: Specific symbol has wrong sector

**Cause:** Yahoo Finance data may be outdated or incorrect

**Solution:**
```bash
# Force refresh that symbol
python scripts/fetch_sectors.py --symbols TICKER --force
```

### Problem: Sector file doesn't exist

**Cause:** Haven't run initial fetch

**Solution:**
```bash
# Run initial setup
python scripts/fetch_sectors.py
```

### Problem: Rate limit errors

**Cause:** Fetching too many symbols too quickly

**Solution:**
- Wait a few minutes and retry
- The script already includes 0.5s delays
- For very large batches, increase delay in code

## Integration with Existing Workflows

### Backfill Process
```bash
# 1. Backfill prices (existing)
python scripts/backfill_all.py --years 10

# 2. Fetch sectors (new step)
python scripts/fetch_sectors.py
```

### Daily Updates
```bash
# Single command handles everything
python scripts/update_daily.py
```
This now includes:
- Price updates
- Macro updates
- Factor rebuilding
- **Sector refresh (quarterly, automatic)**
- DuckDB view updates

### Adding New Stocks
```bash
# Single command handles everything
python scripts/add_symbol.py NVDA TSLA
```
This now includes:
- Price history fetch
- Factor calculation
- **Sector classification fetch**
- DuckDB view updates

## File Organization

```
quant/
├── data/
│   └── sectors/                          # Sector data directory
│       └── sector_classifications.parquet # Main sector file
│
├── src/
│   └── data/
│       └── sector_classification.py      # Core module
│
├── scripts/
│   ├── fetch_sectors.py                  # Initial setup
│   ├── update_sectors.py                 # Quarterly refresh
│   ├── add_symbol.py                     # Updated to fetch sectors
│   └── update_daily.py                   # Updated with sector refresh
│
└── docs/
    └── SECTOR_CLASSIFICATION.md          # This file
```

## Best Practices

### ✅ Do:
- Run `fetch_sectors.py` after initial backfill
- Let `update_daily.py` handle quarterly refreshes automatically
- Use sector data for portfolio construction and analysis
- Check for "Unknown" sectors and retry periodically

### ❌ Don't:
- Fetch sectors more frequently than quarterly (wastes API calls)
- Assume Yahoo sectors match official GICS exactly
- Ignore "Unknown" labels (they indicate missing data)
- Fetch sectors without rate limiting (will hit API limits)

## Future Enhancements

Possible improvements:
- [ ] Add sector-based portfolio constraints
- [ ] Sector rotation strategies
- [ ] Sector momentum analysis
- [ ] Sector correlation matrices
- [ ] Integration with portfolio simulator UI
- [ ] Sector performance dashboards
- [ ] Historical sector changes tracking

## References

- **Yahoo Finance:** https://finance.yahoo.com/
- **yfinance Documentation:** https://github.com/ranaroussi/yfinance
- **GICS Standard (official):** https://www.msci.com/gics
- **S&P Sector Indices:** https://www.spglobal.com/spdji/en/indices/equity/sp-500/

---

**Last Updated:** 2026-01-26  
**Maintainer:** Quant Research Team
