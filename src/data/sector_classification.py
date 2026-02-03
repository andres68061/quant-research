"""
Sector classification data management using Yahoo Finance.

This module provides functions to fetch, store, and manage sector/industry
classifications for stocks. Classifications are stored in Parquet format
and refreshed quarterly to minimize API calls.

Storage Format:
    data/sectors/sector_classifications.parquet
    Columns: symbol, sector, industry, industryKey, sectorKey, last_updated

Refresh Policy:
    - Fetch once when symbol is added
    - Refresh every 3 months (quarterly)
    - Label as 'Unknown' if data unavailable
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from config.settings import PROJECT_ROOT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SECTORS_DIR = PROJECT_ROOT / "data" / "sectors"
SECTORS_FILE = SECTORS_DIR / "sector_classifications.parquet"
REFRESH_DAYS = 90  # 3 months
UNKNOWN_LABEL = "Unknown"


def ensure_sectors_directory() -> None:
    """Create sectors directory if it doesn't exist."""
    SECTORS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Sectors directory: {SECTORS_DIR}")


def fetch_sector_info(symbol: str) -> Dict[str, str]:
    """
    Fetch sector, industry, and asset type classification for a single symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dict with keys: sector, industry, industryKey, sectorKey, quoteType
        Returns 'Unknown' for missing fields
        
    Example:
        >>> info = fetch_sector_info('AAPL')
        >>> print(info)
        {
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'industryKey': 'consumer-electronics',
            'sectorKey': 'technology',
            'quoteType': 'EQUITY'
        }
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        result = {
            'sector': info.get('sector', UNKNOWN_LABEL),
            'industry': info.get('industry', UNKNOWN_LABEL),
            'industryKey': info.get('industryKey', UNKNOWN_LABEL),
            'sectorKey': info.get('sectorKey', UNKNOWN_LABEL),
            'quoteType': info.get('quoteType', UNKNOWN_LABEL),
        }
        
        logger.info(f"✅ Fetched sector for {symbol}: {result['sector']} ({result['quoteType']})")
        return result
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to fetch sector for {symbol}: {e}")
        return {
            'sector': UNKNOWN_LABEL,
            'industry': UNKNOWN_LABEL,
            'industryKey': UNKNOWN_LABEL,
            'sectorKey': UNKNOWN_LABEL,
            'quoteType': UNKNOWN_LABEL,
        }


def fetch_sectors_batch(
    symbols: List[str],
    delay_seconds: float = 0.5
) -> pd.DataFrame:
    """
    Fetch sector classifications for multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        delay_seconds: Delay between requests to avoid rate limits
        
    Returns:
        DataFrame with columns: symbol, sector, industry, industryKey,
        sectorKey, quoteType, last_updated
        
    Example:
        >>> df = fetch_sectors_batch(['AAPL', 'MSFT', 'JPM'])
        >>> print(df)
           symbol       sector              industry  quoteType
        0   AAPL   Technology  Consumer Electronics     EQUITY
        1   MSFT   Technology              Software     EQUITY
        2    JPM   Financials                 Banks     EQUITY
    """
    import time
    
    records = []
    
    for i, symbol in enumerate(symbols):
        info = fetch_sector_info(symbol)
        
        record = {
            'symbol': symbol,
            'sector': info['sector'],
            'industry': info['industry'],
            'industryKey': info['industryKey'],
            'sectorKey': info['sectorKey'],
            'quoteType': info['quoteType'],
            'last_updated': datetime.now().isoformat(),
        }
        records.append(record)
        
        # Progress logging
        if (i + 1) % 10 == 0:
            logger.info(f"   Progress: {i + 1}/{len(symbols)} symbols")
        
        # Rate limiting
        if i < len(symbols) - 1:
            time.sleep(delay_seconds)
    
    df = pd.DataFrame(records)
    return df


def load_sector_classifications() -> Optional[pd.DataFrame]:
    """
    Load existing sector classifications from Parquet.
    
    Returns:
        DataFrame with sector classifications, or None if file doesn't exist
        
    Columns:
        - symbol: Stock ticker
        - sector: Yahoo Finance sector (e.g., 'Technology')
        - industry: Yahoo Finance industry (e.g., 'Software')
        - industryKey: Industry key code
        - sectorKey: Sector key code
        - quoteType: Asset type (EQUITY, ETF, INDEX, MUTUALFUND, etc.)
        - last_updated: ISO timestamp of last fetch
    """
    ensure_sectors_directory()
    
    if not SECTORS_FILE.exists():
        logger.info("No existing sector classifications found")
        return None
    
    try:
        df = pd.read_parquet(SECTORS_FILE)
        logger.info(f"✅ Loaded {len(df)} sector classifications")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading sector classifications: {e}")
        return None


def save_sector_classifications(df: pd.DataFrame) -> None:
    """
    Save sector classifications to Parquet.
    
    Args:
        df: DataFrame with sector classifications
        
    Required columns:
        - symbol
        - sector
        - industry
        - industryKey
        - sectorKey
        - quoteType
        - last_updated
    """
    ensure_sectors_directory()
    
    try:
        df.to_parquet(SECTORS_FILE, index=False)
        logger.info(f"💾 Saved {len(df)} sector classifications to {SECTORS_FILE}")
    except Exception as e:
        logger.error(f"❌ Error saving sector classifications: {e}")
        raise


def needs_refresh(last_updated: str, refresh_days: int = REFRESH_DAYS) -> bool:
    """
    Check if a sector classification needs refreshing.
    
    Args:
        last_updated: ISO timestamp string
        refresh_days: Number of days before refresh needed
        
    Returns:
        True if refresh needed, False otherwise
        
    Example:
        >>> needs_refresh('2024-01-01T00:00:00', refresh_days=90)
        True  # If current date is after 2024-04-01
    """
    try:
        last_date = datetime.fromisoformat(last_updated)
        days_since = (datetime.now() - last_date).days
        return days_since >= refresh_days
    except Exception:
        # If parsing fails, assume refresh needed
        return True


def get_symbols_needing_refresh(
    df: pd.DataFrame,
    refresh_days: int = REFRESH_DAYS
) -> List[str]:
    """
    Get list of symbols that need sector refresh.
    
    Args:
        df: DataFrame with sector classifications
        refresh_days: Number of days before refresh needed
        
    Returns:
        List of symbols needing refresh
    """
    if df is None or df.empty:
        return []
    
    needs_update = df['last_updated'].apply(
        lambda x: needs_refresh(x, refresh_days)
    )
    
    symbols = df.loc[needs_update, 'symbol'].tolist()
    logger.info(f"Found {len(symbols)} symbols needing refresh")
    
    return symbols


def add_or_update_sectors(
    symbols: List[str],
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Add new symbols or update existing sector classifications.
    
    This is the main function to use when:
    - Adding new symbols to the database
    - Running quarterly sector updates
    
    Args:
        symbols: List of ticker symbols to add/update
        force_refresh: If True, refresh even if not stale
        
    Returns:
        Updated DataFrame with all sector classifications
        
    Example:
        >>> # Add new symbols
        >>> df = add_or_update_sectors(['NVDA', 'TSLA'])
        
        >>> # Quarterly refresh
        >>> all_symbols = get_all_symbols()
        >>> df = add_or_update_sectors(all_symbols, force_refresh=False)
    """
    # Load existing classifications
    existing_df = load_sector_classifications()
    
    if existing_df is None or existing_df.empty:
        # No existing data - fetch all symbols
        logger.info(f"📥 Fetching sectors for {len(symbols)} new symbols...")
        new_df = fetch_sectors_batch(symbols)
        save_sector_classifications(new_df)
        return new_df
    
    # Identify symbols to fetch
    existing_symbols = set(existing_df['symbol'].tolist())
    new_symbols = [s for s in symbols if s not in existing_symbols]
    
    if force_refresh:
        # Refresh all provided symbols
        refresh_symbols = symbols
    else:
        # Only refresh stale symbols
        stale_symbols = get_symbols_needing_refresh(existing_df)
        refresh_symbols = [s for s in symbols if s in stale_symbols]
    
    # Fetch new and stale symbols
    to_fetch = list(set(new_symbols + refresh_symbols))
    
    if not to_fetch:
        logger.info("✅ All sectors up to date - no fetching needed")
        return existing_df
    
    logger.info(f"📥 Fetching sectors:")
    logger.info(f"   New symbols: {len(new_symbols)}")
    logger.info(f"   Stale symbols: {len(refresh_symbols)}")
    logger.info(f"   Total to fetch: {len(to_fetch)}")
    
    # Fetch updated data
    fetched_df = fetch_sectors_batch(to_fetch)
    
    # Merge with existing data
    # Remove old entries for refreshed symbols
    updated_df = existing_df[~existing_df['symbol'].isin(to_fetch)].copy()
    
    # Add new/refreshed entries
    updated_df = pd.concat([updated_df, fetched_df], ignore_index=True)
    
    # Sort by symbol
    updated_df = updated_df.sort_values('symbol').reset_index(drop=True)
    
    # Save
    save_sector_classifications(updated_df)
    
    logger.info(f"✅ Updated sectors: {len(updated_df)} total symbols")
    
    return updated_df


def get_sector_for_symbol(symbol: str) -> Tuple[str, str]:
    """
    Get sector and industry for a single symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Tuple of (sector, industry)
        Returns ('Unknown', 'Unknown') if not found
        
    Example:
        >>> sector, industry = get_sector_for_symbol('AAPL')
        >>> print(f"{sector} - {industry}")
        Technology - Consumer Electronics
    """
    df = load_sector_classifications()
    
    if df is None or df.empty:
        logger.warning(f"No sector data available for {symbol}")
        return (UNKNOWN_LABEL, UNKNOWN_LABEL)
    
    row = df[df['symbol'] == symbol]
    
    if row.empty:
        logger.warning(f"Symbol {symbol} not found in sector classifications")
        return (UNKNOWN_LABEL, UNKNOWN_LABEL)
    
    return (
        row.iloc[0]['sector'],
        row.iloc[0]['industry']
    )


def get_asset_type_for_symbol(symbol: str) -> str:
    """
    Get asset type (quoteType) for a single symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Asset type: 'EQUITY', 'ETF', 'INDEX', 'MUTUALFUND', or 'Unknown'
        
    Example:
        >>> asset_type = get_asset_type_for_symbol('SPY')
        >>> print(asset_type)
        ETF
    """
    df = load_sector_classifications()
    
    if df is None or df.empty:
        logger.warning(f"No sector data available for {symbol}")
        return UNKNOWN_LABEL
    
    # Check if quoteType column exists (backwards compatibility)
    if 'quoteType' not in df.columns:
        logger.warning("quoteType column not found - run fetch_sectors.py to update")
        return UNKNOWN_LABEL
    
    row = df[df['symbol'] == symbol]
    
    if row.empty:
        logger.warning(f"Symbol {symbol} not found in sector classifications")
        return UNKNOWN_LABEL
    
    return row.iloc[0]['quoteType']


def get_symbols_by_sector(sector: str) -> List[str]:
    """
    Get all symbols in a given sector.
    
    Args:
        sector: Sector name (e.g., 'Technology', 'Financials')
        
    Returns:
        List of symbols in that sector
        
    Example:
        >>> tech_stocks = get_symbols_by_sector('Technology')
        >>> print(f"Found {len(tech_stocks)} tech stocks")
    """
    df = load_sector_classifications()
    
    if df is None or df.empty:
        logger.warning("No sector data available")
        return []
    
    symbols = df[df['sector'] == sector]['symbol'].tolist()
    logger.info(f"Found {len(symbols)} symbols in {sector}")
    
    return symbols


def get_sector_summary() -> pd.DataFrame:
    """
    Get summary statistics of sector classifications.
    
    Returns:
        DataFrame with sector counts and percentages
        
    Example:
        >>> summary = get_sector_summary()
        >>> print(summary)
                    sector  count  percentage
        0       Technology    150       25.0%
        1       Financials    120       20.0%
        2  Consumer Cyclical  90       15.0%
        ...
    """
    df = load_sector_classifications()
    
    if df is None or df.empty:
        logger.warning("No sector data available")
        return pd.DataFrame()
    
    summary = df['sector'].value_counts().reset_index()
    summary.columns = ['sector', 'count']
    summary['percentage'] = (summary['count'] / len(df) * 100).round(2)
    
    return summary


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("SECTOR CLASSIFICATION MODULE TEST")
    print("=" * 80)
    
    # Test with a few symbols
    test_symbols = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
    
    print(f"\n1. Fetching sectors for test symbols: {test_symbols}")
    df = add_or_update_sectors(test_symbols)
    print(df)
    
    print(f"\n2. Getting sector for AAPL:")
    sector, industry = get_sector_for_symbol('AAPL')
    print(f"   Sector: {sector}")
    print(f"   Industry: {industry}")
    
    print(f"\n3. Sector summary:")
    summary = get_sector_summary()
    print(summary)
    
    print(f"\n4. Checking refresh status:")
    stale = get_symbols_needing_refresh(df, refresh_days=90)
    print(f"   Symbols needing refresh: {len(stale)}")
    
    print("\n✅ Test complete!")
