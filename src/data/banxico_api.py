"""
Banxico (Banco de México) API integration for fetching CETES 28 rates.

The Banxico SIE API provides access to Mexican economic indicators including:
- CETES 28 (28-day Treasury Certificates): Series SF43936
- USD/MXN Exchange Rate: Series SF43718

API Documentation: https://www.banxico.org.mx/SieAPIRest/service/v1/doc/catalogoSeries
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from project root
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

# Constants
# Note: The env var is BANXIC_API_KEY (typo in original, but keeping for compatibility)
BANXICO_API_KEY = os.getenv("BANXIC_API_KEY") or os.getenv("BANXICO_API_KEY")
BANXICO_BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series"

# Series IDs
CETES_28_SERIES = "SF43936"  # CETES 28 annualized rate
USDMXN_SERIES = "SF43718"    # USD/MXN exchange rate


def fetch_banxico_series(
    series_id: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch time series data from Banxico SIE API.
    
    Parameters
    ----------
    series_id : str
        Banxico series ID (e.g., 'SF43936' for CETES 28)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    api_key : str, optional
        Banxico API key. If None, reads from BANXIC_API_KEY env var
    
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and 'value' column
    
    Raises
    ------
    ValueError
        If API key is missing or API request fails
    
    Examples
    --------
    >>> df = fetch_banxico_series('SF43936', '2020-01-01', '2020-12-31')
    >>> df.head()
                value
    2020-01-02  7.25
    2020-01-03  7.25
    ...
    """
    if api_key is None:
        api_key = BANXICO_API_KEY
    
    if not api_key:
        raise ValueError(
            "Banxico API key not found. Set BANXIC_API_KEY environment variable."
        )
    
    # Build URL
    url = f"{BANXICO_BASE_URL}/{series_id}/datos/{start_date}/{end_date}"
    
    # Make request
    headers = {"Bmx-Token": api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Banxico API request failed: {e}")
    
    # Parse JSON
    data = response.json()
    
    if "bmx" not in data or "series" not in data["bmx"]:
        raise ValueError("Unexpected Banxico API response format")
    
    series_data = data["bmx"]["series"]
    
    if not series_data or len(series_data) == 0:
        raise ValueError(f"No data returned for series {series_id}")
    
    # Extract data points
    datos = series_data[0].get("datos", [])
    
    if len(datos) == 0:
        raise ValueError(f"No data points returned for series {series_id}")
    
    # Convert to DataFrame
    records = []
    for point in datos:
        date_str = point.get("fecha")
        value_str = point.get("dato")
        
        if date_str and value_str:
            try:
                date = pd.to_datetime(date_str, format="%d/%m/%Y")
                value = float(value_str.replace(",", ""))
                records.append({"date": date, "value": value})
            except (ValueError, AttributeError):
                continue
    
    if len(records) == 0:
        raise ValueError(f"No valid data points for series {series_id}")
    
    df = pd.DataFrame(records)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    
    return df


def get_cetes28_returns(
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    exact28: bool = False
) -> pd.Series:
    """
    Fetch CETES 28 daily returns from Banxico API.
    
    CETES 28 is published weekly (Thursdays). This function:
    1. Fetches weekly annualized rates from Banxico
    2. Forward-fills to daily frequency
    3. Converts annualized rates to daily returns
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    api_key : str, optional
        Banxico API key
    exact28 : bool, default False
        If True, uses 28-day compounding. If False, uses 360-day year.
    
    Returns
    -------
    pd.Series
        Daily returns with DatetimeIndex
    
    Examples
    --------
    >>> returns = get_cetes28_returns('2020-01-01', '2020-12-31')
    >>> returns.mean() * 252  # Annualized return
    0.0725
    
    Notes
    -----
    - CETES rates are annualized (e.g., 7.25% means 7.25% per year)
    - Banxico publishes rates weekly, so we forward-fill to daily
    - Daily return formula: (1 + annual_rate)^(1/360) - 1
    """
    # Fetch CETES 28 annualized rates
    df = fetch_banxico_series(CETES_28_SERIES, start_date, end_date, api_key)
    
    # Convert percentage to decimal
    df["value"] = df["value"] / 100
    
    # Create daily date range (timezone-aware to match price database)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D", tz="America/New_York")
    
    # Make df index timezone-aware if it isn't already
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    
    # Reindex and forward-fill (CETES is published weekly)
    df_daily = df.reindex(date_range, method="ffill")
    
    # Convert annualized rate to daily return
    if exact28:
        # Exact 28-day compounding
        daily_returns = (1 + df_daily["value"]) ** (28 / 360) - 1
        daily_returns = daily_returns / 28
    else:
        # Simple daily compounding (360-day year)
        daily_returns = (1 + df_daily["value"]) ** (1 / 360) - 1
    
    daily_returns.name = "cetes28_daily_return"
    
    return daily_returns


def get_usdmxn_rate(
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None
) -> pd.Series:
    """
    Fetch USD/MXN exchange rate from Banxico API.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    api_key : str, optional
        Banxico API key
    
    Returns
    -------
    pd.Series
        Daily USD/MXN exchange rate with DatetimeIndex
    
    Examples
    --------
    >>> usdmxn = get_usdmxn_rate('2020-01-01', '2020-12-31')
    >>> usdmxn.mean()
    21.5
    """
    df = fetch_banxico_series(USDMXN_SERIES, start_date, end_date, api_key)
    
    # Create daily date range (timezone-aware to match price database)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D", tz="America/New_York")
    
    # Make df index timezone-aware if it isn't already
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    
    # Reindex and forward-fill
    series = df["value"].reindex(date_range, method="ffill")
    series.name = "usdmxn"
    
    return series


def get_current_cetes28_rate(api_key: Optional[str] = None) -> Tuple[float, datetime]:
    """
    Get the most recent CETES 28 annualized rate.
    
    Parameters
    ----------
    api_key : str, optional
        Banxico API key
    
    Returns
    -------
    rate : float
        Current CETES 28 annualized rate (as decimal, e.g., 0.0815 for 8.15%)
    date : datetime
        Date of the rate
    
    Examples
    --------
    >>> rate, date = get_current_cetes28_rate()
    >>> print(f"CETES 28: {rate*100:.2f}% as of {date.date()}")
    CETES 28: 8.15% as of 2026-01-23
    """
    # Fetch last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    df = fetch_banxico_series(CETES_28_SERIES, start_date, end_date, api_key)
    
    if len(df) == 0:
        raise ValueError("No recent CETES 28 data available")
    
    # Get most recent value
    latest_rate = df["value"].iloc[-1] / 100  # Convert to decimal
    latest_date = df.index[-1]
    
    return latest_rate, latest_date


def cache_cetes28_data(
    start_date: str = "2019-01-01",
    output_path: Optional[Path] = None
) -> pd.Series:
    """
    Fetch and cache CETES 28 data to parquet file.
    
    Parameters
    ----------
    start_date : str, default '2019-01-01'
        Start date for historical data
    output_path : Path, optional
        Output path for parquet file. If None, uses data/cetes28_daily.parquet
    
    Returns
    -------
    pd.Series
        Daily CETES 28 returns
    
    Examples
    --------
    >>> returns = cache_cetes28_data()
    >>> print(f"Cached {len(returns)} days of CETES 28 data")
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch data
    returns = get_cetes28_returns(start_date, end_date)
    
    # Save to parquet
    if output_path is None:
        output_path = Path(__file__).parents[2] / "data" / "cetes28_daily.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = returns.to_frame()
    df.to_parquet(output_path)
    
    print(f"✅ Cached CETES 28 data: {len(returns)} days")
    print(f"   Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"   Saved to: {output_path}")
    
    return returns


if __name__ == "__main__":
    """Test Banxico API integration."""
    print("Testing Banxico API integration...")
    print()
    
    # Test 1: Get current CETES 28 rate
    try:
        rate, date = get_current_cetes28_rate()
        print(f"✅ Current CETES 28 rate: {rate*100:.2f}% (as of {date.date()})")
    except Exception as e:
        print(f"❌ Failed to get current rate: {e}")
    
    print()
    
    # Test 2: Fetch historical data
    try:
        returns = get_cetes28_returns("2024-01-01", "2024-12-31")
        print(f"✅ Fetched CETES 28 returns for 2024")
        print(f"   Days: {len(returns)}")
        print(f"   Mean daily return: {returns.mean():.6f}")
        print(f"   Annualized: {returns.mean() * 360 * 100:.2f}%")
    except Exception as e:
        print(f"❌ Failed to fetch historical data: {e}")
    
    print()
    
    # Test 3: Cache data
    try:
        cached = cache_cetes28_data()
        print(f"✅ Cached {len(cached)} days of data")
    except Exception as e:
        print(f"❌ Failed to cache data: {e}")
