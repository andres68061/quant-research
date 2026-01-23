# Quant - Quantamental Research Platform

> A comprehensive quantitative & fundamental analysis platform with interactive web applications, automated data pipelines, and 14+ commodities/economic indicators tracking.

---

## 🚀 Quick Start (3 Steps)

```bash
# 1. Activate environment
conda activate quant

# 2. Initial data backfill (first time only)
python scripts/backfill_all.py --years 10

# 3. Launch interactive app
./run_portfolio_simulator.sh
# Opens at http://localhost:8501
```

**Done!** You now have:
- ✅ Portfolio backtest simulator
- ✅ 14 commodities & metals analytics (Gold, Silver, Oil, Copper, etc.)
- ✅ Economic indicators dashboard (Interest rates, CPI, GDP, Unemployment)
- ✅ 504 stocks with 10+ years of history
- ✅ Automated daily updates

---

## 📋 Table of Contents

1. [Features](#-features)
2. [Project Structure](#-project-structure)
3. [Installation & Setup](#-installation--setup)
4. [Interactive Apps](#-interactive-apps)
5. [Data Management](#-data-management)
6. [Python API](#-python-api)
7. [Command Reference](#-command-reference)
8. [Automation Setup](#-automation-setup)
9. [GitHub Deployment](#-github-deployment)
10. [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### Interactive Web Applications
- **Portfolio Simulator**: Backtest strategies with 5 weighting schemes and synthetic benchmarks
- **Commodities Analytics**: Track 14 assets (precious metals, energy, agriculture)
- **Economic Dashboard**: Monitor interest rates, inflation, GDP, unemployment with recession highlighting

### Data Infrastructure
- **Incremental Updates**: Fetch only new data, minimizing API calls
- **Multi-Source Integration**: Yahoo Finance, Alpha Vantage, FRED
- **Parquet Storage**: Fast columnar format, excellent compression
- **DuckDB Queries**: SQL interface over Parquet files

### Analysis Capabilities
- Factor-based strategies (momentum, value, volatility)
- Custom portfolio construction with multiple weighting schemes
- Transaction cost modeling
- Comprehensive performance metrics (Sharpe, Sortino, Alpha, Max Drawdown)

---

## 📁 Project Structure

```
quant/
├── apps/                          # Interactive Streamlit apps
│   ├── portfolio_simulator.py    # Main page: Portfolio backtester
│   ├── pages/
│   │   ├── 2_📊_Metals_Analytics.py      # Commodities & metals
│   │   └── 3_📉_Economic_Indicators.py   # Economic data
│   └── utils/
│       ├── portfolio.py           # Portfolio calculation utilities
│       └── metrics.py             # Performance metrics
│
├── src/                           # Core Python modules
│   ├── data/                      # Data fetching & processing
│   │   ├── stock_data.py          # Yahoo Finance fetcher
│   │   ├── database.py            # StockDatabase class
│   │   └── factors/               # Factor computation
│   │       ├── prices.py          # Price-based factors
│   │       ├── fundamentals_fmp.py
│   │       └── macro.py           # FRED economic data
│   ├── analysis/                  # Analysis modules
│   ├── models/                    # ML models
│   └── utils/                     # Utilities
│       └── io.py                  # Parquet I/O functions
│
├── scripts/                       # Command-line scripts
│   ├── backfill_all.py            # Initial setup / full refresh
│   ├── update_daily.py            # Incremental updates
│   └── add_symbol.py              # Add new stocks
│
├── notebooks/                     # Jupyter notebooks
│   └── 04_browse_databases.ipynb
│
├── data/                          # Data storage (not in git)
│   ├── factors/                   # Parquet files (source of truth)
│   │   ├── prices.parquet         # Wide: date × symbols (504 stocks)
│   │   ├── factors_price.parquet  # Price factors (momentum, vol, beta)
│   │   ├── factors_all.parquet    # Combined factors
│   │   ├── macro.parquet          # Economic indicators
│   │   └── factors.duckdb         # SQL query interface
│   └── .cache/                    # API response cache
│       └── fmp/                   # FMP fundamentals by year
│
├── config/                        # Configuration
│   └── settings.py                # API keys, settings
│
├── logs/                          # Application logs
│   └── update.log                 # Incremental update logs
│
├── requirements.txt               # Python dependencies
├── .env                           # API keys (not in git)
├── .env.example                   # API key template
└── README.md                      # This file
```

---

## 🔧 Installation & Setup

### 1. Environment Setup

```bash
# Create conda environment
conda create -n quant python=3.11
conda activate quant

# Install dependencies
pip install -r requirements.txt

# Test environment
python test_environment.py
```

**Python Interpreter**: `/opt/anaconda3/envs/quant/bin/python`

### 2. API Keys Configuration

Create a `.env` file in the project root:

```bash
# Required for economic indicators
FRED_API_KEY=your_key_here

# Required for commodities (energy, agriculture)
ALPHAVANTAGE_API_KEY=your_key_here

# Optional (for fundamentals)
FMP_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Optional (for additional data sources)
OPENAI_API_KEY=your_key_here
BEA_API_KEY=your_key_here
```

**Get Free API Keys:**
- **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html (120 requests/min)
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key (25 requests/day)
- **FMP**: https://site.financialmodelingprep.com/developer/docs/ (250 requests/day)

**Security**: The `.env` file is in `.gitignore` and won't be committed to git.

### 3. Initial Data Setup

```bash
# Fetch full history for S&P 500 stocks (first time only)
python scripts/backfill_all.py --years 10

# This creates:
# - data/factors/prices.parquet (504 stocks, 10 years)
# - data/factors/factors_price.parquet (momentum, volatility, beta)
# - data/factors/macro.parquet (FRED economic indicators)
# - data/factors/factors.duckdb (SQL query interface)
```

**Expected time**: 10-15 minutes for initial backfill

---

## 📊 Interactive Apps

### Launch All Apps

```bash
# Easy way (recommended)
./run_portfolio_simulator.sh

# Or manually
streamlit run apps/portfolio_simulator.py
```

Opens at: **http://localhost:8501**

---

### Page 1: Portfolio Simulator

**Backtest trading strategies with comprehensive metrics**

#### Strategy Types

**1. Factor-Based Strategies**
- Rank stocks by factors (momentum, value, volatility, beta)
- Create long/short or long-only portfolios
- Example: Long top 20% momentum, short bottom 20%

```
Available Factors:
- mom_12_1: 12-month momentum (skip last month)
- mom_6: 6-month momentum  
- vol_252: 252-day volatility
- beta: Market beta
- And more...
```

**2. Equal Weight Portfolio**
- Invest equally in all stocks
- Periodic rebalancing
- Diversification baseline

**3. Custom Selection**
- Choose specific stocks
- 5 weighting schemes:
  - **Equal Weight**: Each stock gets 1/N
  - **Manual Weights**: Specify exact allocations
  - **Cap-Weighted**: Weight by market cap (uses price as proxy)
  - **Share Count**: Input number of shares
  - **Harmonic**: Inverse price weighting

#### Benchmark Options

**1. S&P 500 (Cap-Weighted)**
- Standard market benchmark
- Large-cap focused
- Uses ^GSPC ticker

**2. Equal Weight Universe**
- All stocks weighted equally
- Removes size bias
- Similar to RSP ETF ($40B+ AUM)

**3. Synthetic (Custom Mix)**
- Blend any two components:
  - S&P 500 (^GSPC)
  - NASDAQ Composite (^IXIC)
  - Russell 2000 (^RUT)
  - Dow Jones (^DJI)
  - Equal Weight Universe
- Example: 60% S&P 500 + 40% Equal Weight

**Add More Indices:**
```bash
python scripts/add_symbol.py '^IXIC' '^RUT' '^DJI'
```
*Note: Use quotes around symbols starting with ^*

#### Performance Metrics

**Return Metrics:**
- Total Return
- Annualized Return

**Risk Metrics:**
- Annualized Volatility
- Max Drawdown

**Risk-Adjusted:**
- Sharpe Ratio (return per unit of total risk)
- Sortino Ratio (return per unit of downside risk)
- Calmar Ratio (return per unit of drawdown)

**Relative Metrics (vs Benchmark):**
- Alpha (excess return vs CAPM prediction)
- Beta (sensitivity to benchmark)
- Information Ratio (consistency of outperformance)

#### Example Workflows

**Test Momentum Strategy:**
```
Strategy Type: Factor-Based
Factor: mom_12_1
Top %: 20% (long)
Bottom %: 20% (short)
Benchmark: S&P 500
Rebalancing: Monthly
Transaction Cost: 10 bps
```

**Custom Tech Portfolio:**
```
Strategy Type: Custom Selection
Stocks: AAPL, MSFT, GOOGL, NVDA
Weighting: Manual Weights
  - AAPL: 40%
  - MSFT: 30%
  - GOOGL: 20%
  - NVDA: 10%
Benchmark: Synthetic (70% NASDAQ + 30% S&P)
Rebalancing: Quarterly
```

---

### Page 2: Commodities & Metals Analytics

**Track 14 assets across precious metals, energy, and agriculture**

#### Data Sources
- **Yahoo Finance**: Precious metals ETFs (no API key required!)
- **Alpha Vantage**: Energy, industrial metals, agricultural commodities

#### Assets Covered

**Precious Metals (Yahoo Finance ETFs)**
- 🥇 Gold (GLD) - SPDR Gold Trust
- 🥈 Silver (SLV) - iShares Silver Trust
- ⚪ Platinum (PPLT) - Aberdeen Physical Platinum
- ⚫ Palladium (PALL) - Aberdeen Physical Palladium

**Energy (Alpha Vantage)**
- 🛢️ Crude Oil (WTI) - West Texas Intermediate
- 🛢️ Crude Oil (Brent) - Brent crude oil
- 🔥 Natural Gas - Natural gas spot price

**Industrial Metals (Alpha Vantage)**
- 🟫 Copper - Copper spot price
- ⚪ Aluminum - Aluminum spot price

**Agricultural (Alpha Vantage)**
- 🌾 Wheat - Wheat price
- 🌽 Corn - Corn price
- ☕ Coffee - Coffee price
- 🪴 Cotton - Cotton price
- 🍬 Sugar - Sugar price

#### Analysis Types

1. **Price Trends**: Historical charts, statistics
2. **Returns Analysis**: Cumulative returns, Sharpe ratios
3. **Correlation Matrix**: Inter-commodity correlations
4. **Normalized Comparison**: Rebased to 100 for easy comparison

#### Why ETFs for Precious Metals?
- No API key required
- Highly liquid
- Accurate tracking of bullion prices
- Full historical data
- Real-time market pricing

---

### Page 3: Economic Indicators

**Monitor key economic indicators with recession highlighting**

#### Categories

**Interest Rates**
- DFF: Federal Funds Rate
- DGS10: 10-Year Treasury Rate
- DGS2: 2-Year Treasury Rate

**Inflation**
- CPIAUCSL: Consumer Price Index (CPI)
- PCEPI: Personal Consumption Expenditures

**GDP & Growth**
- GDP: Gross Domestic Product
- GDPC1: Real GDP

**Employment**
- UNRATE: Unemployment Rate
- PAYEMS: Nonfarm Payrolls

#### Features
- Historical trends with recession shading (NBER dates)
- Year-over-Year % change calculation
- Correlation analysis between indicators
- Download data as CSV

---

## 💾 Data Management

### Storage Architecture

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

### Incremental Update System

**Key Concepts:**
- **Existing stocks**: Fetch only new dates since last update
- **New stocks**: Fetch full history when first added
- **Minimizes API calls** and download time

### Workflows

#### Initial Setup (First Time)
```bash
python scripts/backfill_all.py --years 10
# Fetches full history for S&P 500 stocks
# Creates Parquet files and DuckDB views
```

#### Weekly/Monthly Updates (Incremental)
```bash
python scripts/update_daily.py
# Reads existing Parquet files
# Finds last date in each dataset
# Fetches only new data since last date
# Appends to existing Parquet files
# Rebuilds factors for new dates
```

**Output Example:**
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

#### Adding New Stocks
```bash
python scripts/add_symbol.py NVDA TSLA
# Fetches FULL history for new symbols
# Adds as new columns to prices.parquet
# Rebuilds all factors
```

---

## 🐍 Python API

### Quick Data Access

```python
import pandas as pd
import duckdb

# Read from Parquet
prices = pd.read_parquet('data/factors/prices.parquet')
print(prices.shape)  # (25521, 504)

# Query with DuckDB
con = duckdb.connect('data/factors/factors.duckdb')
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

### Stock Data Fetcher

```python
from src.data.stock_data import StockDataFetcher

# Create fetcher
fetcher = StockDataFetcher()

# Fetch data
data = fetcher.fetch_stock_data('AAPL', period='1y')

# Calculate returns
data_with_returns = fetcher.calculate_returns(data)

# Get statistics
stats = fetcher.get_basic_statistics(data_with_returns)
```

### Enhanced Fetcher with Caching

```python
from src.data.enhanced_stock_data import EnhancedStockDataFetcher

# Create enhanced fetcher
fetcher = EnhancedStockDataFetcher()

# Get data (uses cache if available)
data = fetcher.get_stock_data('AAPL', period='1y')

# Incremental update (only fetches new data)
updated_data = fetcher.get_stock_data_incremental('AAPL', period='1y')

# Multiple stocks
symbols = ['AAPL', 'MSFT', 'GOOGL']
multiple_data = fetcher.get_multiple_stocks(symbols, period='1y')
```

### Database Operations

```python
from src.data.database import StockDatabase

# Create database instance
with StockDatabase() as db:
    # Store data
    db.store_stock_data('AAPL', data, source='yfinance')
    
    # Retrieve data
    data = db.get_stock_data('AAPL', source='yfinance')
    
    # Get statistics
    stats = db.get_database_stats()
```

### Parquet Utilities

```python
from src.utils.io import (
    read_parquet,
    write_parquet,
    get_last_date_from_parquet,
    append_rows_to_parquet
)

# Read with existence check
prices = read_parquet(Path('data/factors/prices.parquet'))

# Get most recent date
last_date = get_last_date_from_parquet(Path('data/factors/prices.parquet'))

# Add new dates
append_rows_to_parquet(Path('data/factors/prices.parquet'), new_rows)
```

---

## 📟 Command Reference

### Daily Commands

```bash
# Update data (weekly/monthly)
python scripts/update_daily.py

# Add new stock(s)
python scripts/add_symbol.py NVDA
python scripts/add_symbol.py TSLA COIN PLTR  # Multiple at once

# Full backfill (initial setup)
python scripts/backfill_all.py --years 10

# Launch interactive app
./run_portfolio_simulator.sh

# Explore data in Jupyter
jupyter lab
# Open notebooks/04_browse_databases.ipynb
```

### File Locations

```
data/factors/
├── prices.parquet          # Wide format: date × symbols
├── factors_price.parquet   # Long format: price factors
├── factors_all.parquet     # Combined factors
├── macro.parquet           # Raw macro indicators
└── macro_z.parquet         # Standardized macro

data/.cache/
└── fmp/                    # FMP fundamentals by year (auto-managed)
```

### Common Patterns

```python
# Get last N days
prices = pd.read_parquet('data/factors/prices.parquet')
last_30_days = prices.iloc[-30:]

# Filter by symbols
symbols_of_interest = ['AAPL', 'MSFT', 'GOOGL']
subset = prices[symbols_of_interest]

# Merge prices with factors
prices_long = prices.stack().to_frame('close')
prices_long.index.names = ['date', 'symbol']
factors = pd.read_parquet('data/factors/factors_price.parquet')
merged = prices_long.join(factors, how='inner')
```

---

## ⏰ Automation Setup

### Cron Job Configuration

**Automatic daily updates at 6:00 PM**

```bash
# Edit crontab
crontab -e

# Add this line:
0 18 * * * cd /Users/andres/Downloads/Cursor/quant && /opt/anaconda3/envs/quant/bin/python scripts/update_daily.py >> logs/update.log 2>&1
```

**Schedule Options:**
```bash
0 18 * * *       # Every day at 6 PM
0 18 * * 1       # Every Monday at 6 PM (weekly)
0 18 * * 1-5     # Weekdays only at 6 PM
0 2 1 * *        # First of month at 2 AM (monthly)
```

### Monitoring

```bash
# View latest log
tail -20 logs/update.log

# Watch updates in real-time
tail -f logs/update.log

# Check cron job status
crontab -l
```

### Expected Behavior

**When data is up-to-date:**
```
📈 Prices are already up to date!
📊 Macro data is up to date
📉 Skipping factor rebuild (no new price data)
✅ Data is already up to date - no changes needed
```

**When new data is available:**
```
📈 Added 5 new dates (2025-10-24 → 2025-10-29)
📊 New macro data available through 2025-10-29
📉 Rebuilt price factors: (12,862,584 rows)
✅ Incremental update completed successfully!
```

---

## 🔒 GitHub Deployment

### Pre-Push Checklist

**✅ Your Project is GitHub-Ready!**

**What's Protected:**
1. **API Keys** (.env file is IGNORED)
2. **Data Files** (400 MB ignored)
3. **Logs** (logs/ directory ignored)
4. **Cache** (.cache/ ignored)

### Verification Commands

```bash
# Check what files git sees
git status

# Verify .env is ignored
git check-ignore -v .env
# Output: .gitignore:4:*.env    .env  ✅

# Verify data is ignored
git check-ignore -v data/factors/prices.parquet
# Output: .gitignore:47:data/   data/factors/prices.parquet  ✅

# List all tracked files (should be ~50-100, not thousands)
git ls-files | wc -l
```

### Push to GitHub

```bash
# Stage all files
git add .

# Review what's staged
git status

# Commit
git commit -m "Initial commit: Quantamental research platform"

# Create GitHub repo at https://github.com/new
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

**Total tracked size**: ~2 MB ✅  
**Total ignored size**: ~400 MB ✅

### Setup Instructions (For Contributors)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Create conda environment
conda create -n quant python=3.11
conda activate quant

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure .env
cp .env.example .env
# Edit .env with your actual API keys

# 5. Run initial data backfill
python scripts/backfill_all.py --years 10

# 6. Set up cron job (optional)
crontab -e
# Add: 0 18 * * * cd /path/to/repo && /path/to/python scripts/update_daily.py >> logs/update.log 2>&1
```

---

## 🐛 Troubleshooting

### Installation Issues

**"No module named 'streamlit'"**
```bash
conda activate quant
pip install streamlit plotly
```

**"Data directory not found"**
```bash
python scripts/backfill_all.py --years 10
```

### API Key Issues

**"FRED API Key not found"**
```bash
# Add to .env file:
FRED_API_KEY=your_key_here
```
Get free key: https://fred.stlouisfed.org/docs/api/api_key.html

**"Alpha Vantage API Key not found"**
```bash
# Add to .env file:
ALPHAVANTAGE_API_KEY=your_key_here
```
Get free key: https://www.alphavantage.co/support/#api-key

**Note:** Precious metals (Gold, Silver, Platinum, Palladium) work without any API key!

### App Issues

**App is slow**
- Reduce date range (use last 5 years)
- Use monthly/quarterly rebalancing
- Close and restart (first run caches data)

**Charts not displaying**
```bash
conda activate quant
pip install plotly
```

**Synthetic benchmark only shows 2 options**
```bash
# Add more indices
python scripts/add_symbol.py '^IXIC' '^RUT' '^DJI'
# Then restart the app
```

### Data Issues

**"Symbol not found"**
Check symbol is valid on Yahoo Finance:
```python
import yfinance as yf
ticker = yf.Ticker('AAPL')
print(ticker.info)
```

**"Already up to date"**
Normal - no new data available since last update

**Factors shape mismatch**
Run full backfill to rebuild consistently:
```bash
python scripts/backfill_all.py --years 10
```

### Cron Job Issues

**Updates don't run**
```bash
# 1. Check if cron is running
ps aux | grep cron

# 2. Test command manually
cd /Users/andres/Downloads/Cursor/quant && \
/opt/anaconda3/envs/quant/bin/python scripts/update_daily.py

# 3. Check permissions
ls -la scripts/update_daily.py
```

---

## 📚 Key Packages

**Data Science**
- numpy, pandas, scipy

**Visualization**
- matplotlib, seaborn, plotly

**Financial Analysis**
- yfinance, pandas-datareader, ta-lib

**Machine Learning**
- scikit-learn, statsmodels

**Development**
- jupyter, pytest, python-dotenv

**Web Apps**
- streamlit (interactive dashboards)
- fredapi (Federal Reserve data)

---

## 🎯 Best Practices

### ✅ Do This
- Test multiple date ranges for robustness
- Use realistic transaction costs (10 bps for institutions)
- Choose appropriate benchmarks (match strategy characteristics)
- Look at multiple metrics (don't rely on Sharpe alone)
- Run `update_daily.py` weekly/monthly
- Automate updates with cron job
- Back up `data/factors/*.parquet` regularly

### ❌ Avoid This
- Over-optimizing on one period (curve-fitting)
- Ignoring transaction costs (unrealistic results)
- Daily rebalancing (too expensive in practice)
- Wrong benchmark (comparing tech stocks to S&P 500)
- Looking only at returns (ignore risk metrics)

---

## 🔮 Future Enhancements

**Phase 2:**
- [ ] Add incremental updates for FMP fundamentals
- [ ] Add caching for other APIs (Finnhub, FRED)
- [ ] Build data validation checks
- [ ] Build monitoring/alerting for failures

**Phase 3:**
- [ ] Parallel symbol fetching
- [ ] Delta encoding for Parquet
- [ ] Partitioned Parquet files by year
- [ ] Cloud storage integration (S3)

**App Enhancements:**
- [ ] Crypto Analytics page
- [ ] Sector Rotation analysis
- [ ] Correlation Dashboard (cross-asset)
- [ ] Custom Indicators builder
- [ ] Real-time Alerts setup
- [ ] Export to PDF reports

---

## 📊 Summary

| Feature | Status |
|---------|--------|
| Multi-page app architecture | ✅ Complete |
| Portfolio Simulator | ✅ 3 strategies, 5 weighting schemes |
| Commodities Analytics | ✅ 14 assets (precious metals, energy, agriculture) |
| Economic Indicators | ✅ 4 categories, recession highlighting |
| Incremental updates | ✅ Efficient data management |
| DuckDB integration | ✅ SQL queries over Parquet |
| API integrations | ✅ Yahoo Finance, Alpha Vantage, FRED |
| Automated updates | ✅ Cron job ready |
| Documentation | ✅ Comprehensive (you're reading it!) |

---

## 📖 License

[Add license information here]

---

## 🤝 Contributing

[Add contribution guidelines here]

---

**Welcome to your comprehensive quantitative analytics platform!** 📈📊📉

For questions or support, review the troubleshooting section or check the API documentation above.

**Happy backtesting!** 🚀
