from typing import List, Optional

import pandas as pd
import yfinance as yf


def _yahoo_daily(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.Series]:
    """
    Fetch daily adjusted close from Yahoo Finance via yfinance.
    
    Uses Adjusted Close to account for dividends and splits, providing
    total return performance suitable for backtesting and portfolio analysis.
    
    Args:
        symbol: Stock ticker symbol
        start: Start date in 'YYYY-MM-DD' format (inclusive). If None, fetches all history.
        end: End date in 'YYYY-MM-DD' format (inclusive). If None, fetches to today.
        
    Returns:
        Series with adjusted close prices indexed by date, or None if unavailable
    """
    try:
        t = yf.Ticker(symbol)
        
        if start is not None or end is not None:
            # Fetch specific date range
            hist = t.history(start=start, end=end, auto_adjust=False)
        else:
            # Fetch full history
            hist = t.history(period='max', auto_adjust=False)
        
        if hist.empty:
            return None
        
        # Use Adjusted Close for total return (accounts for dividends and splits)
        # Fallback to Close if Adj Close not available (shouldn't happen with yfinance)
        if 'Adj Close' in hist.columns:
            close = hist['Adj Close']
        else:
            close = hist['Close']
        
        close.index = pd.to_datetime(close.index)
        return close.sort_index()
    except Exception:
        return None


def fetch_daily_close(symbol: str) -> Optional[pd.Series]:
    """
    Fetch daily adjusted close exclusively from Yahoo Finance via yfinance (full history).
    
    Returns adjusted close prices suitable for backtesting and performance analysis.
    """
    return _yahoo_daily(symbol)


def fetch_daily_close_incremental(symbol: str, since_date: pd.Timestamp) -> Optional[pd.Series]:
    """
    Fetch daily adjusted close since a specific date (incremental update).
    
    Args:
        symbol: Stock ticker symbol
        since_date: Fetch data from this date onwards (inclusive)
        
    Returns:
        Series with new adjusted close prices, or None if unavailable
    """
    # Add 1 day to since_date to avoid duplicates
    start_date = (since_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return _yahoo_daily(symbol, start=start_date)


def build_prices_panel(symbols: List[str]) -> pd.DataFrame:
    """
    Return wide DataFrame: index=date (B), columns=symbol, values=close.
    Fetches full history for all symbols.
    """
    series_list = []
    for sym in symbols:
        s = fetch_daily_close(sym)
        if s is None or s.empty:
            continue
        s = s.asfreq('B').ffill()
        s.name = sym
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    panel = pd.concat(series_list, axis=1)
    panel.index.name = 'date'
    return panel


def update_prices_panel_incremental(
    existing_panel: pd.DataFrame,
    symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Update existing prices panel with new data since the last date.

    Args:
        existing_panel: Existing prices DataFrame (wide format: date × symbols)
        symbols: List of symbols to update. If None, updates all symbols in existing_panel.

    Returns:
        Updated prices DataFrame with new dates appended
    """
    if existing_panel is None or existing_panel.empty:
        # No existing data, fetch full history
        return build_prices_panel(symbols or [])

    # Get last date from existing panel
    last_date = existing_panel.index.max()

    # Determine which symbols to update
    symbols_to_update = symbols if symbols is not None else existing_panel.columns.tolist()

    # Fetch incremental data for each symbol
    new_data = {}
    for sym in symbols_to_update:
        new_series = fetch_daily_close_incremental(sym, last_date)
        if new_series is not None and not new_series.empty:
            new_series = new_series.asfreq('B').ffill()
            new_data[sym] = new_series

    if not new_data:
        # No new data fetched
        return existing_panel

    # Create new rows DataFrame
    new_rows = pd.DataFrame(new_data)
    new_rows.index.name = 'date'

    # Concatenate with existing data
    combined = pd.concat([existing_panel, new_rows])

    # Remove duplicate dates (keep last)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()

    return combined


def add_symbol_to_panel(existing_panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add a new symbol to existing prices panel by fetching its full history.

    Args:
        existing_panel: Existing prices DataFrame (wide format: date × symbols)
        symbol: New symbol to add

    Returns:
        Updated prices DataFrame with new symbol as a column
    """
    # Fetch full history for new symbol
    new_series = fetch_daily_close(symbol)

    if new_series is None or new_series.empty:
        # Failed to fetch, return existing panel unchanged
        return existing_panel

    # Convert to business day frequency and forward fill
    new_series = new_series.asfreq('B').ffill()
    new_series.name = symbol

    if existing_panel is None or existing_panel.empty:
        # No existing data, create new panel
        df = new_series.to_frame()
        df.index.name = 'date'
        return df

    # Add as new column
    existing_panel[symbol] = new_series

    return existing_panel


def fetch_symbols_batch(symbols: List[str]) -> pd.DataFrame:
    """
    Fetch multiple symbols in a single batch using yfinance.download().
    
    This is MUCH more efficient than sequential fetching - uses parallel threads!
    Can fetch 50-100 symbols in one request instead of 50-100 separate requests.
    
    Args:
        symbols: List of ticker symbols to fetch
        
    Returns:
        DataFrame with date index and symbols as columns (adjusted close prices)
    """
    if not symbols:
        return pd.DataFrame()
    
    try:
        # yfinance.download() can fetch multiple symbols at once with parallel threads
        # Returns DataFrame with MultiIndex columns: (symbol, price_type)
        data = yf.download(
            symbols,
            period='max',
            interval='1d',
            auto_adjust=False,
            prepost=False,
            threads=True,  # Enable parallel downloading - KEY OPTIMIZATION!
            progress=False
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # Handle single symbol case (different structure)
        if len(symbols) == 1:
            symbol = symbols[0]
            if 'Adj Close' in data.columns:
                result = data[['Adj Close']].copy()
                result.columns = [symbol]
                result.index.name = 'date'
                return result.sort_index()
            elif 'Close' in data.columns:
                result = data[['Close']].copy()
                result.columns = [symbol]
                result.index.name = 'date'
                return result.sort_index()
            return pd.DataFrame()
        
        # Multiple symbols: extract Adj Close for each
        # Columns are MultiIndex: (price_type, symbol) - note the order!
        adj_close_data = {}
        for symbol in symbols:
            try:
                # MultiIndex is (Price, Ticker), so ('Adj Close', symbol)
                if ('Adj Close', symbol) in data.columns:
                    adj_close_data[symbol] = data[('Adj Close', symbol)]
                elif ('Close', symbol) in data.columns:
                    # Fallback to Close if Adj Close not available
                    adj_close_data[symbol] = data[('Close', symbol)]
            except (KeyError, TypeError):
                # Symbol not found or no data
                continue
        
        if not adj_close_data:
            return pd.DataFrame()
        
        result = pd.DataFrame(adj_close_data)
        result.index.name = 'date'
        
        # Ensure timezone-naive index for consistency with existing data
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        
        return result.sort_index()
        
    except Exception as e:
        print(f"Error in batch fetch: {e}")
        return pd.DataFrame()


def add_symbols_batch(existing_panel: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    Add multiple symbols to existing prices panel using batch fetching.
    
    Much more efficient than adding symbols one by one!
    
    Args:
        existing_panel: Existing prices DataFrame (wide format: date × symbols)
        symbols: List of new symbols to add
        
    Returns:
        Updated prices DataFrame with new symbols as columns
    """
    if not symbols:
        return existing_panel
    
    # Filter out symbols that already exist
    new_symbols = [s for s in symbols if s not in existing_panel.columns]
    
    if not new_symbols:
        return existing_panel
    
    # Fetch all new symbols in one batch
    new_data = fetch_symbols_batch(new_symbols)
    
    if new_data.empty:
        return existing_panel
    
    # Convert to business day frequency and forward fill
    new_data = new_data.asfreq('B').ffill()
    
    if existing_panel is None or existing_panel.empty:
        new_data.index.name = 'date'
        return new_data
    
    # Merge with existing data
    combined = pd.concat([existing_panel, new_data], axis=1)
    combined = combined.sort_index()
    
    return combined