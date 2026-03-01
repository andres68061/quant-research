from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from config.settings import FMP_API_KEY


def load_sp500_static(csv_path: Path) -> pd.DataFrame:
    """Load a static S&P 500 universe from CSV with columns: symbol,name,sector."""
    df = pd.read_csv(csv_path)
    required = {'symbol', 'name', 'sector'}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    df['start_date'] = pd.NaT
    df['end_date'] = pd.NaT
    return df[['symbol', 'name', 'sector', 'start_date', 'end_date']]


def fetch_sp500_from_wikipedia() -> pd.DataFrame:
    """Fetch current S&P 500 constituents.

    Primary: Wikipedia with browser-like headers to avoid 403.
    Fallback: Public CSV from DataHub if Wikipedia is blocked.

    Returns columns: symbol,name,sector,start_date,end_date
    """
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/126.0.0.0 Safari/537.36'
    }
    try:
        resp = requests.get(wiki_url, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        table = tables[0]
        col_map = {
            'Symbol': 'symbol',
            'Security': 'name',
            'GICS Sector': 'sector',
        }
        table = table.rename(columns=col_map)
        df = table[['symbol', 'name', 'sector']].copy()
    except Exception:
        # Fallback CSV (may be slightly outdated)
        fallback_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        fallback = pd.read_csv(fallback_url)
        col_map = {
            'Symbol': 'symbol',
            'Name': 'name',
            'Sector': 'sector',
        }
        fallback = fallback.rename(columns=col_map)
        df = fallback[['symbol', 'name', 'sector']].copy()

    # Normalize tickers for Yahoo (replace '.' with '-')
    df['symbol'] = df['symbol'].astype(str).str.replace('.', '-', regex=False).str.upper()
    df['start_date'] = pd.NaT
    df['end_date'] = pd.NaT
    return df[['symbol', 'name', 'sector', 'start_date', 'end_date']]


def fetch_sp500_from_fmp() -> pd.DataFrame:
    """Fetch current S&P 500 constituents from Financial Modeling Prep.

    Endpoint: /api/v3/sp500_constituent (or fallback /sp500_constituents)
    Returns columns: symbol,name,sector,start_date,end_date
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not set")

    base = "https://financialmodelingprep.com/api/v3"
    urls = [
        f"{base}/sp500_constituent?apikey={FMP_API_KEY}",
        f"{base}/sp500_constituents?apikey={FMP_API_KEY}",
    ]
    data = None
    for u in urls:
        try:
            r = requests.get(u, timeout=30)
            r.raise_for_status()
            j = r.json()
            if isinstance(j, list) and len(j) > 0:
                data = j
                break
        except Exception:
            continue
    if data is None:
        raise RuntimeError("Failed to fetch S&P 500 from FMP")

    df = pd.DataFrame(data)
    # Map common field names; sector might be missing in some responses
    # Try a few common keys; default to 'Unknown' if absent
    symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
    name_col = 'name' if 'name' in df.columns else ('Security' if 'Security' in df.columns else None)
    sector_col = 'sector' if 'sector' in df.columns else ('GICS Sector' if 'GICS Sector' in df.columns else None)

    out = pd.DataFrame()
    out['symbol'] = df[symbol_col].astype(str)
    if name_col:
        out['name'] = df[name_col].astype(str)
    else:
        out['name'] = ''
    if sector_col and sector_col in df.columns:
        out['sector'] = df[sector_col].astype(str)
    else:
        out['sector'] = 'Unknown'

    # Normalize tickers for Yahoo (replace '.' with '-')
    out['symbol'] = out['symbol'].str.replace('.', '-', regex=False).str.upper()
    out['start_date'] = pd.NaT
    out['end_date'] = pd.NaT
    return out[['symbol', 'name', 'sector', 'start_date', 'end_date']]


