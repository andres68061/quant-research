from typing import Dict

import pandas as pd
from fredapi import Fred

from config.settings import FRED_API_KEY


def load_fred_series(series_map: Dict[str, str]) -> pd.DataFrame:
    """Load multiple FRED series by id and return daily ffilled DataFrame.

    series_map: {column_name: fred_series_id}
    """
    fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
    frames = []
    for col, fred_id in series_map.items():
        if fred is None:
            s = pd.Series(dtype=float, name=col)
        else:
            s = fred.get_series(fred_id)
            s = s.rename(col)
        s.index = pd.to_datetime(s.index)
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index()
    df = df.asfreq('B').ffill()
    df.index.name = 'date'
    return df


def load_default_macro() -> pd.DataFrame:
    """Load a default macro set and derive common features.

    - cpi_yoy: YoY from CPIAUCSL (monthly)
    - unrate: unemployment rate (monthly)
    - fed_funds: FEDFUNDS (monthly)
    - dgs10: 10-year Treasury yield (daily)
    - t10y2y: 10y-2y spread (daily)
    """
    fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
    frames = {}
    if fred is None:
        return pd.DataFrame()
    # CPI YoY
    cpi = fred.get_series('CPIAUCSL')
    cpi.index = pd.to_datetime(cpi.index)
    cpi_yoy = cpi.pct_change(12, fill_method=None).rename('cpi_yoy')
    frames['cpi_yoy'] = cpi_yoy
    # Unemployment rate
    unrate = fred.get_series('UNRATE')
    unrate.index = pd.to_datetime(unrate.index)
    frames['unrate'] = unrate.rename('unrate')
    # Fed funds
    ff = fred.get_series('FEDFUNDS')
    ff.index = pd.to_datetime(ff.index)
    frames['fed_funds'] = ff.rename('fed_funds')
    # 10-year yield
    dgs10 = fred.get_series('DGS10')
    dgs10.index = pd.to_datetime(dgs10.index)
    frames['dgs10'] = dgs10.rename('dgs10')
    # 10y-2y spread
    try:
        t10y2y = fred.get_series('T10Y2Y')
        t10y2y.index = pd.to_datetime(t10y2y.index)
        frames['t10y2y'] = t10y2y.rename('t10y2y')
    except Exception:
        pass

    df = pd.concat(frames.values(), axis=1).sort_index()
    df = df.asfreq('B').ffill()
    df.index.name = 'date'
    return df


def compute_macro_zscores(macro_df: pd.DataFrame, window_days: int = 252 * 5) -> pd.DataFrame:
    """Rolling z-score per series to standardize macro signals.

    If insufficient history, uses expanding stats.
    """
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()
    df = macro_df.copy()
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col].astype(float)
        roll = s.rolling(window_days, min_periods=max(30, window_days // 12))
        mu = roll.mean()
        sd = roll.std(ddof=0)
        z = (s - mu) / (sd.replace(0, pd.NA))
        out[f'macro_z_{col}'] = z
    return out


