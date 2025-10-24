# Quant Project

A quantitative analysis and trading project built with Python.

## Environment Setup

This project uses a conda environment named `quant`.

### Activate Environment
```bash
conda activate quant
```

### Deactivate Environment
```bash
conda deactivate
```

## Project Structure

```
quant/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── __init__.py
│   ├── analysis/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── notebooks/
└── config/
    └── __init__.py
```

## Installation

1. Activate the conda environment:
   ```bash
   conda activate quant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Test the environment:
   ```bash
   python test_environment.py
   ```

## Python Interpreter

The quant environment uses Python 3.11.13 located at:
```
/opt/anaconda3/envs/quant/bin/python
```

To use this interpreter in your IDE:
- **VS Code/Cursor**: Select the interpreter at the path above
- **PyCharm**: Add the interpreter from the conda environment
- **Jupyter**: The environment is already configured for Jupyter notebooks

## Usage

### Quick Start

1. **Initial data backfill** (first time only):
   ```bash
   python scripts/backfill_all.py --years 10
   ```
   This fetches full history for S&P 500 stocks, macro data, and fundamentals.

2. **Weekly/monthly updates** (incremental):
   ```bash
   python scripts/update_daily.py
   ```
   This fetches only new data since last update.

3. **Add new stocks**:
   ```bash
   python scripts/add_symbol.py NVDA TSLA
   ```

4. **Explore data in Jupyter**:
   ```bash
   jupyter lab
   # Open notebooks/04_browse_databases.ipynb
   ```

5. **Query with DuckDB**:
   ```python
   import duckdb
   con = duckdb.connect('data/factors/factors.duckdb')
   con.sql("SELECT * FROM prices WHERE symbol='AAPL' LIMIT 10").df()
   ```

### Project Structure

- `src/data/` - Data fetching, database, and processing modules
- `src/analysis/` - Analysis and modeling modules  
- `src/models/` - Machine learning and statistical models
- `src/utils/` - Utility functions and helpers
- `tests/` - Unit tests and test data
- `notebooks/` - Jupyter notebooks for exploration
- `config/` - Configuration files and API keys
- `data/` - Database files and ML datasets
- `logs/` - Application logs
- `results/` - Generated visualizations and analysis results

### Key Packages Available

- **Data Science**: numpy, pandas, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Financial Analysis**: yfinance, pandas-datareader, ta-lib
- **Machine Learning**: scikit-learn, statsmodels
- **Development**: jupyter, pytest, python-dotenv

### Data Storage System

The project uses a Parquet-based storage architecture optimized for quantamental research:

- **Parquet Files (Source of Truth)**: All data stored in columnar Parquet format at `data/factors/`
  - `prices.parquet` - Wide format (date × symbols): 504 stocks, 100 years of history
  - `factors_price.parquet` - Price-based factors (momentum, volatility, beta)
  - `factors_all.parquet` - Combined price + fundamental factors
  - `macro.parquet` & `macro_z.parquet` - Macroeconomic indicators
  
- **DuckDB (Query Interface)**: SQL views over Parquet files at `data/factors/factors.duckdb`
  - No data storage - just a query layer
  - Fast SQL analytics on Parquet files
  
- **API Cache Layer**: Protects against rate limits
  - FMP fundamentals cached in `data/.cache/fmp/`
  - Avoid redundant API calls
  
- **Incremental Updates**: Efficient data management
  - Weekly/monthly updates fetch only new dates
  - New stocks fetch full history on demand
  - Minimizes API calls and bandwidth

See [INCREMENTAL_UPDATES.md](INCREMENTAL_UPDATES.md) for complete documentation.

### API Integration

- **Finnhub API**: Real-time quotes, company profiles, news, and market data
- **Yahoo Finance**: Historical price data and financial information
- **Configurable Sources**: Easy to add new data providers

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]

### Secrets and environment variables

The project loads environment variables via `python-dotenv`. Create a local `.env` at the repository root (not committed) or export vars in your shell.

Required variables:
- `FINNHUB_API_KEY`
- `ALPHAVANTAGE_API_KEY`
- `FMP_API_KEY`
- `OPENAI_API_KEY`
- `FRED_API_KEY`
- `BEA_API_KEY`
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `REDDIT_USERNAME`, `REDDIT_PASSWORD`

Example `.env`:

```dotenv
FINNHUB_API_KEY=...
ALPHAVANTAGE_API_KEY=...
FMP_API_KEY=...
OPENAI_API_KEY=...
FRED_API_KEY=...
BEA_API_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=...
REDDIT_USERNAME=...
REDDIT_PASSWORD=...
```

Alternatively, export variables in your shell and/or add them to `~/.zshrc`.
