import numpy as np
import pandas as pd


def compute_returns(close: pd.Series) -> pd.Series:
    return close.pct_change(fill_method=None)


def momentum_excluding_recent(close: pd.Series, months: int) -> pd.Series:
    # 12-1 style: past m months excluding the most recent 21 trading days
    ret = close.pct_change(fill_method=None)
    recent = 21
    window = months * 21
    cum = (1 + ret).rolling(window).apply(np.prod, raw=True) - 1.0
    ex_recent = (1 + ret).rolling(recent).apply(np.prod, raw=True) - 1.0
    return cum.sub(ex_recent, fill_value=0.0)


def rolling_volatility(ret: pd.Series, window: int) -> pd.Series:
    return ret.rolling(window).std() * np.sqrt(252)


def rolling_beta(asset_ret: pd.Series, market_ret: pd.Series, window: int = 60) -> pd.Series:
    cov = asset_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    beta = cov / var
    return beta


def build_price_factors(close_panel: pd.DataFrame, market_symbol: str = 'SPY') -> pd.DataFrame:
    close_panel = close_panel.sort_index()
    ret_panel = close_panel.pct_change(fill_method=None)
    market_ret = ret_panel.get(market_symbol)
    factors = {}
    for sym in close_panel.columns:
        s_close = close_panel[sym]
        s_ret = ret_panel[sym]
        factors[(sym, 'mom_12_1')] = momentum_excluding_recent(s_close, 12)
        factors[(sym, 'mom_6_1')] = momentum_excluding_recent(s_close, 6)
        factors[(sym, 'mom_3_1')] = momentum_excluding_recent(s_close, 3)
        factors[(sym, 'vol_60d')] = rolling_volatility(s_ret, 60)
        if market_ret is not None:
            factors[(sym, 'beta_60d')] = rolling_beta(s_ret, market_ret, 60)
    df = pd.concat(factors, axis=1)
    df.index.name = 'date'
    # flatten columns to MultiIndex → columns: symbol,factor
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['symbol', 'factor'])
    # Adopt future stack behavior to silence deprecation warning
    out = df.stack('symbol', future_stack=True).reset_index().set_index(['date', 'symbol']).sort_index()
    return out


