#!/usr/bin/env python3
"""
Commodities Data Fetcher

Fetches and stores commodity prices from multiple sources:
- Yahoo Finance (precious metals ETFs)
- Alpha Vantage (energy, industrial, agricultural commodities)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf

from config.settings import ALPHAVANTAGE_API_KEY

# Commodities metadata
COMMODITIES_CONFIG = {
    # Precious Metals (Yahoo Finance ETFs)
    "GLD": {
        "name": "Gold (GLD ETF)",
        "source": "yahoo",
        "unit": "USD",
        "category": "precious_metals",
    },
    "SLV": {
        "name": "Silver (SLV ETF)",
        "source": "yahoo",
        "unit": "USD",
        "category": "precious_metals",
    },
    "PPLT": {
        "name": "Platinum (PPLT ETF)",
        "source": "yahoo",
        "unit": "USD",
        "category": "precious_metals",
    },
    "PALL": {
        "name": "Palladium (PALL ETF)",
        "source": "yahoo",
        "unit": "USD",
        "category": "precious_metals",
    },
    # Energy (Alpha Vantage)
    "WTI": {
        "name": "Crude Oil (WTI)",
        "source": "alphavantage",
        "unit": "USD/barrel",
        "category": "energy",
    },
    "BRENT": {
        "name": "Crude Oil (Brent)",
        "source": "alphavantage",
        "unit": "USD/barrel",
        "category": "energy",
    },
    "NATURAL_GAS": {
        "name": "Natural Gas",
        "source": "alphavantage",
        "unit": "USD/MMBtu",
        "category": "energy",
    },
    # Industrial Metals (Alpha Vantage)
    "COPPER": {
        "name": "Copper",
        "source": "alphavantage",
        "unit": "USD/lb",
        "category": "industrial",
    },
    "ALUMINUM": {
        "name": "Aluminum",
        "source": "alphavantage",
        "unit": "USD/ton",
        "category": "industrial",
    },
    # Agricultural (Alpha Vantage)
    "WHEAT": {
        "name": "Wheat",
        "source": "alphavantage",
        "unit": "USD/bushel",
        "category": "agricultural",
    },
    "CORN": {
        "name": "Corn",
        "source": "alphavantage",
        "unit": "USD/bushel",
        "category": "agricultural",
    },
    "COFFEE": {
        "name": "Coffee",
        "source": "alphavantage",
        "unit": "USD/lb",
        "category": "agricultural",
    },
    "COTTON": {
        "name": "Cotton",
        "source": "alphavantage",
        "unit": "USD/lb",
        "category": "agricultural",
    },
    "SUGAR": {
        "name": "Sugar",
        "source": "alphavantage",
        "unit": "USD/lb",
        "category": "agricultural",
    },
}


class CommodityDataFetcher:
    """Fetches commodity data from multiple sources."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize fetcher.

        Args:
            data_dir: Directory to store parquet files. Defaults to ROOT/data/commodities/
        """
        if data_dir is None:
            # Default to project_root/data/commodities/
            self.data_dir = Path(__file__).parents[2] / "data" / "commodities"
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.prices_file = self.data_dir / "prices.parquet"

    def fetch_yahoo_commodity(self, symbol: str, interval: str = "1d") -> pd.Series:
        """
        Fetch commodity data from Yahoo Finance (ETFs).

        Args:
            symbol: Yahoo Finance ticker (e.g., 'GLD', 'SLV')
            interval: Data interval ('1d', '1wk', '1mo')

        Returns:
            pandas Series with datetime index and closing prices
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get maximum available history
            hist = ticker.history(period="max", interval=interval)
            
            if hist.empty:
                print(f"⚠️  No data returned for {symbol}")
                return pd.Series(dtype=float)
            
            # Extract closing prices
            series = hist["Close"]
            series.index = pd.to_datetime(series.index).tz_localize(None)
            series.index.name = "date"
            series.name = symbol
            
            print(f"✓ Fetched {symbol}: {len(series)} data points "
                  f"({series.index[0].date()} to {series.index[-1].date()})")
            
            return series
            
        except Exception as e:
            print(f"✗ Error fetching {symbol} from Yahoo Finance: {e}")
            return pd.Series(dtype=float)

    def fetch_alphavantage_commodity(
        self, symbol: str, interval: str = "daily"
    ) -> pd.Series:
        """
        Fetch commodity data from Alpha Vantage.

        Args:
            symbol: Alpha Vantage commodity symbol (e.g., 'WTI', 'COPPER')
            interval: Data interval ('daily', 'weekly', 'monthly')

        Returns:
            pandas Series with datetime index and prices
        """
        if not ALPHAVANTAGE_API_KEY:
            print(f"⚠️  Alpha Vantage API key not configured, skipping {symbol}")
            return pd.Series(dtype=float)

        try:
            # Rate limit: 1 request per second for free tier
            time.sleep(1.1)
            
            url = "https://www.alphavantage.co/query"
            params = {
                "function": symbol,
                "interval": interval,
                "apikey": ALPHAVANTAGE_API_KEY,
                "datatype": "json",
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Alpha Vantage returns data in 'data' key
            if "data" not in data:
                # Try to extract error message
                error_msg = data.get("Note") or data.get("Error Message") or data.get("Information") or str(data)
                if "rate limit" in error_msg.lower() or "per second" in error_msg.lower():
                    print(f"⚠️  Rate limit hit for {symbol}, please wait and retry later")
                else:
                    print(f"⚠️  No data for {symbol}: {error_msg[:100]}")
                return pd.Series(dtype=float)

            records = data["data"]
            if not records:
                print(f"⚠️  Empty data for {symbol}")
                return pd.Series(dtype=float)

            # Convert to DataFrame then Series
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            # Alpha Vantage uses 'value' column for commodity prices
            if "value" not in df.columns:
                print(f"⚠️  No 'value' column for {symbol}")
                return pd.Series(dtype=float)

            # Convert to float, handling '.' as NaN
            series = pd.to_numeric(df["value"], errors="coerce")
            series = series.dropna()
            series.name = symbol
            series.index.name = "date"

            print(f"✓ Fetched {symbol}: {len(series)} data points "
                  f"({series.index[0].date()} to {series.index[-1].date()})")

            return series

        except requests.RequestException as e:
            print(f"✗ Network error fetching {symbol}: {e}")
            return pd.Series(dtype=float)
        except Exception as e:
            print(f"✗ Error fetching {symbol} from Alpha Vantage: {e}")
            return pd.Series(dtype=float)

    def fetch_commodity(self, symbol: str) -> pd.Series:
        """
        Fetch commodity data (auto-detects source).

        Args:
            symbol: Commodity symbol

        Returns:
            pandas Series with datetime index and prices
        """
        if symbol not in COMMODITIES_CONFIG:
            print(f"✗ Unknown commodity: {symbol}")
            return pd.Series(dtype=float)

        config = COMMODITIES_CONFIG[symbol]
        source = config["source"]

        if source == "yahoo":
            return self.fetch_yahoo_commodity(symbol, interval="1d")
        elif source == "alphavantage":
            return self.fetch_alphavantage_commodity(symbol, interval="daily")
        else:
            print(f"✗ Unknown source '{source}' for {symbol}")
            return pd.Series(dtype=float)

    def fetch_all_commodities(self) -> pd.DataFrame:
        """
        Fetch all configured commodities.

        Returns:
            DataFrame with date index and commodity prices as columns
        """
        all_series = {}

        for symbol in COMMODITIES_CONFIG.keys():
            series = self.fetch_commodity(symbol)
            if not series.empty:
                all_series[symbol] = series

        if not all_series:
            print("⚠️  No commodities data fetched")
            return pd.DataFrame()

        # Combine into single DataFrame
        df = pd.concat(all_series, axis=1)
        df.index.name = "date"
        df = df.sort_index()

        return df

    def save_prices(self, df: pd.DataFrame) -> None:
        """
        Save commodity prices to parquet file.

        Args:
            df: DataFrame with date index and commodity prices
        """
        if df.empty:
            print("⚠️  No data to save")
            return

        df.to_parquet(self.prices_file, engine="pyarrow", compression="snappy")
        print(f"✓ Saved {len(df)} rows to {self.prices_file}")

    def load_prices(self) -> pd.DataFrame:
        """
        Load commodity prices from parquet file.

        Returns:
            DataFrame with date index and commodity prices
        """
        if not self.prices_file.exists():
            print(f"⚠️  No existing data at {self.prices_file}")
            return pd.DataFrame()

        df = pd.read_parquet(self.prices_file)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    def update_commodity(self, symbol: str, existing_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Update a single commodity (incremental fetch from last date).

        Args:
            symbol: Commodity symbol
            existing_df: Optional existing DataFrame to update (if None, loads from file)

        Returns:
            Updated DataFrame with all commodities
        """
        # Load existing data if not provided
        if existing_df is None:
            existing_df = self.load_prices()
        
        # Fetch new data
        new_series = self.fetch_commodity(symbol)

        if new_series.empty:
            print(f"⚠️  No new data for {symbol}")
            return existing_df

        # If we have existing data for this commodity, only keep new dates
        if symbol in existing_df.columns:
            last_date = existing_df[symbol].last_valid_index()
            if last_date is not None:
                new_series = new_series[new_series.index > last_date]
                if new_series.empty:
                    print(f"✓ {symbol} already up to date")
                    return existing_df

        # Merge with existing data
        if existing_df.empty:
            updated_df = pd.DataFrame({symbol: new_series})
        else:
            # Add new series as column or update existing
            updated_df = existing_df.copy()
            if symbol in updated_df.columns:
                # Combine old and new, dropping duplicates
                combined = pd.concat([updated_df[symbol], new_series])
                combined = combined[~combined.index.duplicated(keep="last")]
                updated_df[symbol] = combined
            else:
                updated_df[symbol] = new_series

        updated_df = updated_df.sort_index()
        return updated_df

    def update_all_commodities(self) -> pd.DataFrame:
        """
        Update all commodities (incremental fetch).

        Returns:
            Updated DataFrame with all commodities
        """
        updated_df = self.load_prices()

        for symbol in COMMODITIES_CONFIG.keys():
            updated_df = self.update_commodity(symbol)

        return updated_df


if __name__ == "__main__":
    # Quick test
    fetcher = CommodityDataFetcher()
    print(f"Data directory: {fetcher.data_dir}")
    print(f"Prices file: {fetcher.prices_file}")
    print("\nConfigured commodities:")
    for symbol, config in COMMODITIES_CONFIG.items():
        print(f"  {symbol:15s} - {config['name']:30s} [{config['source']}]")

