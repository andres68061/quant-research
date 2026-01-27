from typing import List, Optional

import pandas as pd
import yfinance as yf


def _yahoo_daily(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.Series]:
    """
    Fetch daily close from Yahoo Finance via yfinance.
    
    Args:
        symbol: Stock ticker symbol
        start: Start date in 'YYYY-MM-DD' format (inclusive). If None, fetches all history.
        end: End date in 'YYYY-MM-DD' format (inclusive). If None, fetches to today.
        
    Returns:
        Series with close prices indexed by date, or None if unavailable
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
        
        close = hist['Close']
        close.index = pd.to_datetime(close.index)
        return close.sort_index()
    except Exception:
        return None


def fetch_daily_close(symbol: str) -> Optional[pd.Series]:
    """Fetch daily close exclusively from Yahoo Finance via yfinance (full history)."""
    return _yahoo_daily(symbol)


def fetch_daily_close_incremental(symbol: str, since_date: pd.Timestamp) -> Optional[pd.Series]:
    """
    Fetch daily close since a specific date (incremental update).
    
    Args:
        symbol: Stock ticker symbol
        since_date: Fetch data from this date onwards (inclusive)
        
    Returns:
        Series with new close prices, or None if unavailable
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


