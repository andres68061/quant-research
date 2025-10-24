# Quant Project API Documentation

## Overview
This document provides comprehensive API documentation for the quant project's modules and functions.

## Core Modules

### 1. StockDataFetcher (`src/data/stock_data.py`)

#### Class: `StockDataFetcher`
A class for fetching and processing stock market data from Yahoo Finance.

**Methods:**

##### `fetch_stock_data(symbol, period="1y", interval="1d", start_date=None, end_date=None)`
- **Purpose**: Fetch stock data for a given symbol
- **Parameters**:
  - `symbol` (str): Stock symbol (e.g., 'AAPL', 'MSFT')
  - `period` (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
  - `interval` (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
  - `start_date` (str, optional): Start date in 'YYYY-MM-DD' format
  - `end_date` (str, optional): End date in 'YYYY-MM-DD' format
- **Returns**: `pd.DataFrame` with columns [Open, High, Low, Close, Volume, Dividends, Stock Splits]
- **Example**:
  ```python
  fetcher = StockDataFetcher()
  data = fetcher.fetch_stock_data('AAPL', period='1y')
  ```

##### `calculate_returns(data)`
- **Purpose**: Calculate various return metrics from price data
- **Parameters**: `data` (pd.DataFrame): Stock price data
- **Returns**: `pd.DataFrame` with additional columns:
  - `Daily_Return`: Daily percentage returns
  - `Cumulative_Return`: Cumulative returns
  - `Log_Return`: Logarithmic returns
  - `Volatility_30d`: 30-day rolling volatility
  - `MA_20`: 20-day moving average
  - `MA_50`: 50-day moving average

##### `get_basic_statistics(data)`
- **Purpose**: Calculate basic statistics for stock data
- **Parameters**: `data` (pd.DataFrame): Stock data with returns calculated
- **Returns**: `Dict[str, float]` containing:
  - `current_price`: Latest closing price
  - `highest_price`: Maximum price in period
  - `lowest_price`: Minimum price in period
  - `price_range`: Price range
  - `mean_daily_return`: Average daily return
  - `std_daily_return`: Standard deviation of daily returns
  - `annualized_volatility`: Annualized volatility
  - `sharpe_ratio`: Sharpe ratio
  - `total_return`: Total return percentage
  - `avg_volume`: Average trading volume

### 2. StockDatabase (`src/data/database.py`)

#### Class: `StockDatabase`
A database class for storing and managing stock market data using SQLite.

**Methods:**

##### `store_stock_data(symbol, data, source='yfinance')`
- **Purpose**: Store stock data in the database
- **Parameters**:
  - `symbol` (str): Stock symbol
  - `data` (pd.DataFrame): Stock data to store
  - `source` (str): Data source (e.g., 'yfinance', 'finnhub')
- **Returns**: `bool` - True if successful

##### `get_stock_data(symbol, start_date=None, end_date=None, source='yfinance')`
- **Purpose**: Retrieve stock data from the database
- **Parameters**:
  - `symbol` (str): Stock symbol
  - `start_date` (str, optional): Start date in 'YYYY-MM-DD' format
  - `end_date` (str, optional): End date in 'YYYY-MM-DD' format
  - `source` (str): Data source to retrieve from
- **Returns**: `pd.DataFrame` with stock data or None if not found

##### `get_database_stats()`
- **Purpose**: Get database statistics
- **Returns**: `Dict` containing:
  - `total_records`: Total number of records
  - `total_symbols`: Number of unique symbols
  - `total_sources`: Number of data sources
  - `date_range`: Dictionary with 'start' and 'end' dates

### 3. EnhancedStockDataFetcher (`src/data/enhanced_stock_data.py`)

#### Class: `EnhancedStockDataFetcher`
Enhanced stock data fetcher with intelligent caching and multiple data sources.

**Methods:**

##### `get_stock_data(symbol, period="1y", source="auto", force_refresh=False)`
- **Purpose**: Get stock data with intelligent caching
- **Parameters**:
  - `symbol` (str): Stock symbol
  - `period` (str): Data period
  - `source` (str): Data source ('auto', 'yfinance', 'finnhub', 'database')
  - `force_refresh` (bool): Force refresh from external source
- **Returns**: `pd.DataFrame` with stock data and calculated returns

##### `get_stock_data_incremental(symbol, period="1y", source="auto")`
- **Purpose**: Get stock data with incremental updates (only fetch new data)
- **Parameters**:
  - `symbol` (str): Stock symbol
  - `period` (str): Data period
  - `source` (str): Data source
- **Returns**: `pd.DataFrame` with complete stock data (existing + new)

##### `get_multiple_stocks(symbols, period="1y", source="auto")`
- **Purpose**: Get data for multiple stocks efficiently
- **Parameters**:
  - `symbols` (List[str]): List of stock symbols
  - `period` (str): Data period
  - `source` (str): Data source
- **Returns**: `Dict[str, pd.DataFrame]` with symbol as key and data as value

## Convenience Functions

### `fetch_stock_data(symbol, period="1y")`
Quick function to fetch stock data.
```python
from src.data.stock_data import fetch_stock_data
data = fetch_stock_data('AAPL', period='1y')
```

### `get_stock_analysis(symbol, period="1y")`
Get comprehensive stock analysis including data and statistics.
```python
from src.data.stock_data import get_stock_analysis
analysis = get_stock_analysis('AAPL', period='1y')
# Returns: {'data': DataFrame, 'statistics': Dict}
```

## Usage Examples

### Basic Data Fetching
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

### Enhanced Data Fetching with Caching
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

## Error Handling

All functions return `None` or empty DataFrames on error. Check the logs for detailed error messages.

## Performance Tips

1. **Use caching**: The `EnhancedStockDataFetcher` automatically caches data
2. **Incremental updates**: Use `get_stock_data_incremental()` for updates
3. **Batch operations**: Use `get_multiple_stocks()` for multiple symbols
4. **Database queries**: Use date ranges to limit database queries

## Configuration

See `config/settings.py` for configuration options including:
- API keys
- Database paths
- Data source settings
- Technical indicator parameters
- Risk management settings
