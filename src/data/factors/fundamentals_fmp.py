import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from config.settings import FMP_API_KEY, PROJECT_ROOT

FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _get(url: str, params: Dict = None, sleep: float = 0.0) -> Optional[Dict]:
    try:
        r = requests.get(url, params=params or {}, timeout=30)
        r.raise_for_status()
        if sleep:
            time.sleep(sleep)
        return r.json()
    except Exception:
        return None


def fetch_ratios_for_symbol(symbol: str, period: str = 'quarter', limit: int = 64) -> pd.DataFrame:
    """Fetch key ratios from FMP for a symbol.

    Uses /ratios endpoint which aggregates many value/quality ratios.
    Returns DataFrame with at least columns: date, symbol, pe, pb, ps, roe, roa, grossMargin, debtToEquity, revenueGrowth
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not set")

    url = f"{FMP_BASE}/ratios/{symbol}"
    params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
    data = _get(url, params, sleep=0.2)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df['symbol'] = symbol
    # Ensure a standard date field (report date)
    if 'date' in df.columns:
        df['report_date'] = pd.to_datetime(df['date'])
    elif 'calendarYear' in df.columns:
        # Fallback: construct from calendarYear and period (Q1/Q2/...)
        df['report_date'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')
    else:
        df['report_date'] = pd.to_datetime('today')

    # Select and rename key fields if present
    col_map = {
        'priceEarningsRatio': 'pe',
        'priceToBookRatio': 'pb',
        'priceToSalesRatio': 'ps',
        'returnOnEquity': 'roe',
        'returnOnAssets': 'roa',
        'grossProfitMargin': 'gross_margin',
        'debtEquityRatio': 'debt_to_equity',
        'revenueGrowth': 'revenue_growth',
    }
    for src, dst in col_map.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors='coerce')
        else:
            df[dst] = pd.NA

    out = df[['symbol', 'report_date', 'pe', 'pb', 'ps', 'roe', 'roa', 'gross_margin', 'debt_to_equity', 'revenue_growth']].copy()
    return out.sort_values('report_date')


# -------------------- BULK + CACHING --------------------

def _cache_dir() -> Path:
    d = PROJECT_ROOT / 'data' / '.cache' / 'fmp'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path_ratios(period: str, year: int) -> Path:
    return _cache_dir() / f'ratios_bulk_{period}_{year}.json'


def fetch_bulk_ratios_year(year: int, period: str = 'quarter', refresh_cache: bool = False) -> Optional[pd.DataFrame]:
    """Fetch ratios-bulk for a given year and cache raw JSON.

    Returns a DataFrame with columns including symbol, date and various ratios, or None if unavailable.
    """
    if not FMP_API_KEY:
        return None
    cache_path = _cache_path_ratios(period, year)
    data = None
    if cache_path.exists() and not refresh_cache:
        try:
            data = pd.read_json(cache_path)
            # If saved as list, read_json returns DataFrame; else try json.load fallback below
            if isinstance(data, pd.DataFrame) and not data.empty:
                df = data
            else:
                data = None
        except Exception:
            data = None
        if data is not None:
            df = data
            # Ensure symbol/date presence
            if 'symbol' in df.columns and ('date' in df.columns or 'reportDate' in df.columns):
                return df
            # else fall through to re-fetch

    # Fetch from API
    url = f"{FMP_BASE}/ratios-bulk"
    params = {"period": period, "year": year, "apikey": FMP_API_KEY}
    j = _get(url, params=params, sleep=0.2)
    if not j or not isinstance(j, list) or len(j) == 0:
        return None
    df = pd.DataFrame(j)
    # Cache raw JSON via DataFrame to_json for simplicity
    try:
        df.to_json(cache_path, orient='records')
    except Exception:
        pass
    return df


def load_bulk_ratios_range(start_year: int, end_year: int, period: str = 'quarter', refresh_cache: bool = False) -> pd.DataFrame:
    """Load bulk ratios across a year range (inclusive), using cache when possible.

    Returns a DataFrame with at least columns: symbol, report_date, and selected ratios.
    """
    frames: List[pd.DataFrame] = []
    for y in range(start_year, end_year + 1):
        df = fetch_bulk_ratios_year(y, period=period, refresh_cache=refresh_cache)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, axis=0, ignore_index=True)

    # Normalize columns to expected names
    # Prefer 'date' for report date, some payloads may use other keys
    if 'date' in all_df.columns:
        all_df['report_date'] = pd.to_datetime(all_df['date'])
    elif 'reportDate' in all_df.columns:
        all_df['report_date'] = pd.to_datetime(all_df['reportDate'])
    else:
        all_df['report_date'] = pd.NaT

    all_df['symbol'] = all_df['symbol'].astype(str).str.replace('.', '-', regex=False).str.upper()

    # Map ratio fields where available
    col_map = {
        'priceEarningsRatio': 'pe',
        'priceToBookRatio': 'pb',
        'priceToSalesRatio': 'ps',
        'returnOnEquity': 'roe',
        'returnOnAssets': 'roa',
        'grossProfitMargin': 'gross_margin',
        'debtEquityRatio': 'debt_to_equity',
        'revenueGrowth': 'revenue_growth',
    }
    out = pd.DataFrame()
    out['symbol'] = all_df['symbol']
    out['report_date'] = all_df['report_date']
    for src, dst in col_map.items():
        out[dst] = pd.to_numeric(all_df.get(src), errors='coerce') if src in all_df.columns else pd.NA

    out = out.dropna(subset=['report_date']).sort_values(['symbol', 'report_date'])
    return out


def fetch_fundamentals_universe(symbols: List[str], period: str = 'quarter', limit: int = 64) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        df = fetch_ratios_for_symbol(sym, period=period, limit=limit)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, axis=0, ignore_index=True)
    return all_df


def apply_asof_lag(fund_df: pd.DataFrame, lag_days: int = 60) -> pd.DataFrame:
    """Add as-of date to enforce reporting lag and drop pre-as-of rows when dailyized."""
    if fund_df is None or fund_df.empty:
        return pd.DataFrame()
    df = fund_df.copy()
    if 'report_date' not in df.columns:
        # Nothing to lag; return empty to signal missing fundamentals
        return pd.DataFrame()
    df['asof_date'] = df['report_date'] + pd.to_timedelta(lag_days, unit='D')
    return df


def dailyize_fundamentals(fund_df: pd.DataFrame, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Convert lagged fundamentals to daily business-day DataFrame indexed by (date, symbol).

    For each symbol, values are forward-filled from asof_date onwards.
    """
    if fund_df.empty:
        return pd.DataFrame()
    fund_df = fund_df.copy()
    if start is None:
        start = fund_df['asof_date'].min()
    if end is None:
        end = pd.Timestamp.today().normalize()
    idx = pd.date_range(start=start, end=end, freq='B')

    out_frames = []
    for sym, g in fund_df.groupby('symbol'):
        g = g.sort_values('asof_date')
        # Build a time series by setting index to asof_date and forward-fill onto business days
        ts = g.set_index('asof_date')[['pe', 'pb', 'ps', 'roe', 'roa', 'gross_margin', 'debt_to_equity', 'revenue_growth']]
        ts = ts[~ts.index.duplicated(keep='last')]
        ts = ts.reindex(idx).ffill()
        ts['symbol'] = sym
        ts.index.name = 'date'
        out_frames.append(ts)
    daily = pd.concat(out_frames, axis=0)
    daily = daily.reset_index().set_index(['date', 'symbol']).sort_index()
    return daily


def compute_value_quality_factors(fund_daily: pd.DataFrame) -> pd.DataFrame:
    """Derive value and quality factors from dailyized fundamentals.

    - earnings_yield = 1 / pe
    - value_composite: z-score of [PB, PS, 1/PE]
    - quality_composite: z-score of [ROE, ROA, gross_margin] minus leverage z (debt_to_equity)
    """
    if fund_daily.empty:
        return pd.DataFrame()
    df = fund_daily.copy()
    df['earnings_yield'] = 1.0 / df['pe']

    def zscore(x: pd.Series) -> pd.Series:
        return (x - x.mean()) / (x.std(ddof=0) + 1e-9)

    # Cross-sectional z per day
    parts = []
    for _date, g in df.groupby(level=0):
        zz = pd.DataFrame(index=g.index)
        zz['z_pe_inv'] = zscore(g['earnings_yield'])
        zz['z_pb'] = -zscore(g['pb'])  # lower PB is more value → flip sign
        zz['z_ps'] = -zscore(g['ps'])  # lower PS is more value → flip sign
        zz['value_composite'] = zz[['z_pe_inv', 'z_pb', 'z_ps']].mean(axis=1)

        zz['z_roe'] = zscore(g['roe'])
        zz['z_roa'] = zscore(g['roa'])
        zz['z_gm'] = zscore(g['gross_margin'])
        zz['z_leverage'] = -zscore(g['debt_to_equity'])  # lower leverage preferred → flip sign
        zz['quality_composite'] = zz[['z_roe', 'z_roa', 'z_gm', 'z_leverage']].mean(axis=1)

        parts.append(zz[['value_composite', 'quality_composite']])

    factors_vq = pd.concat(parts, axis=0).sort_index()
    return factors_vq


