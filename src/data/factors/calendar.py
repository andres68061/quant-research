
import pandas as pd


def generate_trading_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate a simple NYSE-like trading calendar (Mon-Fri, no holidays)."""
    idx = pd.date_range(start=start_date, end=end_date, freq='B')
    cal = pd.DataFrame({'date': idx.date, 'is_trading_day': True})
    return cal


def resample_to_daily(series: pd.Series) -> pd.Series:
    """Resample any DatetimeIndex series to daily business days with forward-fill."""
    daily = series.asfreq('B').ffill()
    return daily


