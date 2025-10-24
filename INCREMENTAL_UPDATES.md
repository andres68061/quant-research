# Incremental Update System

This document explains the incremental update architecture for the quantamental research platform.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 DATA STORAGE ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  PARQUET FILES (Source of Truth)                            │
│  data/factors/                                               │
│  ├─ prices.parquet       (wide: date × symbols)             │
│  ├─ factors_price.parquet (long: date, symbol, factors)     │
│  ├─ factors_all.parquet   (long: combined factors)          │
│  ├─ macro.parquet         (date indexed)                    │
│  └─ macro_z.parquet       (standardized macro)              │
│                                                               │
│  CACHE LAYER (API Rate Limit Protection)                    │
│  data/.cache/                                                │
│  └─ fmp/                  (FMP fundamentals by year)         │
│                                                               │
│  DUCKDB (Query Interface - No Storage)                      │
│  data/factors/factors.duckdb                                 │
│  └─ Creates SQL views over Parquet files                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Parquet = Database
- All data permanently stored in Parquet format
- Columnar storage: fast queries, excellent compression
- Wide format for prices (date × symbols)
- Long format for factors (date, symbol, factor columns)

### 2. Incremental Updates
- **Existing stocks**: Fetch only new dates since last update
- **New stocks**: Fetch full history when first added
- Minimizes API calls and download time

### 3. Cache Layer
- API responses cached to avoid rate limits
- FMP fundamentals cached by year in `data/.cache/fmp/`
- Can add cache for other APIs (Finnhub, FRED, etc.)

### 4. DuckDB Views
- DuckDB doesn't store data
- Creates SQL views pointing to Parquet files
- Convenient for SQL-based analysis

## Workflows

### Initial Setup (First Time)

```bash
# 1. Run full backfill to populate initial data
python scripts/backfill_all.py --years 10

# This fetches:
# - Full price history for all S&P 500 stocks
# - Macro indicators from FRED
# - FMP fundamentals (uses cache)
# - Computes all factors
# - Creates Parquet files and DuckDB views
```

### Weekly/Monthly Updates (Incremental)

```bash
# 2. Run incremental update to fetch only new data
python scripts/update_daily.py

# This does:
# - Reads existing Parquet files
# - Finds last date in each dataset
# - Fetches only new data since last date
# - Appends to existing Parquet files
# - Rebuilds factors for new dates
# - Updates DuckDB views
```

**Output example:**
```
================================================================================
🔄 INCREMENTAL DATA UPDATE
================================================================================

📈 Updating prices from data/factors/prices.parquet...
   Last date in prices: 2024-10-17
   Fetching new data since 2024-10-17...
   ✅ Added 5 new dates
   New last date: 2024-10-24

📊 Updating macro from data/factors/macro.parquet...
   Last date in macro: 2024-10-17
   ℹ️  Macro data is up to date

📉 Rebuilding price factors...
   ✅ Rebuilt price factors: (12860064, 5)

🦆 Updating DuckDB views at data/factors/factors.duckdb...
   ✅ Registered view: prices
   ✅ Registered view: factors_price
   ✅ Registered view: factors_all

================================================================================
✅ Incremental update completed successfully!
================================================================================
```

### Adding New Stocks

```bash
# 3. Add individual stocks to the universe
python scripts/add_symbol.py NVDA
python scripts/add_symbol.py TSLA COIN PLTR

# This does:
# - Fetches FULL history for new symbol(s)
# - Adds as new column(s) to prices.parquet
# - Rebuilds all factors (including new symbols)
# - Updates DuckDB views
```

**Output example:**
```
================================================================================
➕ ADDING 1 NEW SYMBOL(S)
================================================================================
Symbols: NVDA

📈 Reading existing prices from data/factors/prices.parquet...
   Current symbols: 504
   Adding 1 new symbols: NVDA

📥 Fetching full history for new symbols...
   Fetching NVDA...
   ✅ Added NVDA: 5247 data points

💾 Saving updated prices...
   ✅ Saved: (25516, 505)

📉 Rebuilding all factors (this may take a moment)...
   ✅ Rebuilt price factors: (12885580, 5)

🦆 Updating DuckDB views...
   ✅ DuckDB views updated

================================================================================
✅ Successfully added 1 new symbol(s)!
================================================================================
```

### Adding New Fundamentals

When you want to add new fundamental ratios (e.g., FCF yield, EV/EBITDA):

1. **Update `fundamentals_fmp.py`**:
   - Add new ratio mapping in `col_map`
   - FMP cache will reuse existing year files

2. **Run backfill to populate**:
   ```bash
   # Use --refresh-cache to re-fetch if needed
   python scripts/backfill_all.py --refresh-cache
   ```

3. The new fundamental will be:
   - Fetched for all symbols
   - Added to `fundamentals.parquet`
   - Included in `factors_all.parquet`

## Script Reference

### `scripts/backfill_all.py`
**Purpose**: Initial data population or full refresh

**Usage**:
```bash
python scripts/backfill_all.py --years 10
python scripts/backfill_all.py --refresh-cache  # Force re-fetch cached FMP data
```

**When to use**:
- First time setup
- After adding new fundamental columns
- When you need to rebuild everything from scratch

### `scripts/update_daily.py`
**Purpose**: Incremental updates for existing data

**Usage**:
```bash
python scripts/update_daily.py
```

**When to use**:
- Daily/weekly/monthly data updates
- Automated via cron job
- After market close to get latest prices

**Cron example** (run daily at 6 PM):
```bash
0 18 * * * cd /path/to/quant && /path/to/conda/envs/quant/bin/python scripts/update_daily.py >> logs/update.log 2>&1
```

### `scripts/add_symbol.py`
**Purpose**: Add new stocks to universe

**Usage**:
```bash
python scripts/add_symbol.py NVDA
python scripts/add_symbol.py TSLA COIN PLTR  # Multiple symbols
```

**When to use**:
- Adding stocks to your research universe
- Stock joins S&P 500 or your watchlist
- Ad-hoc analysis of new ticker

## Data Flow

### Price Data Flow
```
Yahoo Finance API
    ↓
fetch_daily_close_incremental(symbol, since_date)
    ↓
update_prices_panel_incremental(existing_panel)
    ↓
prices.parquet (updated with new dates)
    ↓
build_price_factors(prices_panel)
    ↓
factors_price.parquet
    ↓
DuckDB views updated
```

### New Symbol Flow
```
Yahoo Finance API
    ↓
fetch_daily_close(symbol)  ← Full history
    ↓
add_symbol_to_panel(existing_panel, symbol)
    ↓
prices.parquet (new column added)
    ↓
build_price_factors(prices_panel)  ← Rebuild all
    ↓
factors_price.parquet (updated)
```

## API Rate Limits & Caching

### Yahoo Finance
- **Rate limits**: None (generous)
- **Caching**: Not needed for prices
- **Incremental updates**: Reduce load on yfinance servers

### FMP (Financial Modeling Prep)
- **Rate limits**: 250 calls/day (free tier)
- **Caching**: ✅ Cached in `data/.cache/fmp/`
- **Cache key**: `{period}_{year}.json`
- **Refresh**: Use `--refresh-cache` flag

### Future APIs (Finnhub, FRED, etc.)
Follow the FMP pattern:
1. Create cache directory in `data/.cache/{api_name}/`
2. Check cache before API call
3. Save response to cache
4. Implement `refresh_cache` flag

## Utilities Reference

### `src/utils/io.py`
Core Parquet manipulation functions:

```python
# Read/Write
read_parquet(path)                    # Read with existence check
write_parquet(df, path)               # Write with dir creation

# Introspection
get_last_date_from_parquet(path)      # Get most recent date
get_existing_symbols_from_parquet(path)  # Get symbol list

# Incremental Updates
append_rows_to_parquet(path, new_rows)    # Add new dates
add_columns_to_parquet(path, new_cols)    # Add new symbols/factors

# Merging
merge_parquet_files(path1, path2, output)  # Join on index

# DuckDB
connect_duckdb(db_path)               # Connect to DuckDB
register_parquet(con, name, path)     # Create view
```

### `src/data/factors/prices.py`
Price fetching and panel management:

```python
# Full history
fetch_daily_close(symbol)             # Fetch all history

# Incremental
fetch_daily_close_incremental(symbol, since_date)

# Panel operations
build_prices_panel(symbols)           # Build from scratch
update_prices_panel_incremental(existing_panel)  # Add new dates
add_symbol_to_panel(existing_panel, symbol)      # Add new symbol
```

## Best Practices

### 1. Regular Updates
- Run `update_daily.py` weekly/monthly
- Automate with cron job
- Check logs for failures

### 2. Before Adding Many Symbols
- Use `add_symbol.py` for 1-10 stocks
- Use `backfill_all.py` with custom universe CSV for 100+ stocks

### 3. Cache Management
- FMP cache lives in `data/.cache/fmp/`
- Ignored by git (in `.gitignore`)
- Safe to delete - will re-fetch as needed

### 4. Disaster Recovery
- Parquet files are your source of truth
- Back up `data/factors/*.parquet` regularly
- Can rebuild DuckDB from Parquet anytime

### 5. Performance
- Parquet files are fast for columnar queries
- DuckDB is optimized for analytics
- Incremental updates minimize API calls

## Troubleshooting

### "No existing prices found"
**Solution**: Run `backfill_all.py` first to create initial data

### "Already up to date"
**Normal**: No new data available since last update

### Symbol fetch fails
**Check**: 
- Symbol exists and is valid
- Internet connection
- Yahoo Finance service status

### Factors shape mismatch
**Solution**: Run full backfill to rebuild all factors consistently

## Next Steps

### Phase 1 (Complete) ✅
- [x] Delete SQLite infrastructure
- [x] Build incremental update utilities
- [x] Enhance prices.py with incremental functions
- [x] Create update_daily.py script
- [x] Create add_symbol.py script

### Phase 2 (Future)
- [ ] Add incremental updates for FMP fundamentals
- [ ] Add caching for other APIs (Finnhub, FRED)
- [ ] Build add_fundamental.py script
- [ ] Add data validation checks
- [ ] Build monitoring/alerting for failures

### Phase 3 (Advanced)
- [ ] Parallel symbol fetching
- [ ] Delta encoding for Parquet
- [ ] Partitioned Parquet files by year
- [ ] Cloud storage integration (S3)

