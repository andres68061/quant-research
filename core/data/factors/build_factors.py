import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-12


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


def build_price_factors(close_panel: pd.DataFrame, market_symbol: str = "SPY") -> pd.DataFrame:
    close_panel = close_panel.sort_index()
    ret_panel = close_panel.pct_change(fill_method=None)
    market_ret = ret_panel.get(market_symbol)
    factors = {}
    for sym in close_panel.columns:
        s_close = close_panel[sym]
        s_ret = ret_panel[sym]
        factors[(sym, "mom_12_1")] = momentum_excluding_recent(s_close, 12)
        factors[(sym, "mom_6_1")] = momentum_excluding_recent(s_close, 6)
        factors[(sym, "mom_3_1")] = momentum_excluding_recent(s_close, 3)
        factors[(sym, "vol_60d")] = rolling_volatility(s_ret, 60)
        if market_ret is not None:
            factors[(sym, "beta_60d")] = rolling_beta(s_ret, market_ret, 60)
    df = pd.concat(factors, axis=1)
    df.index.name = "date"
    # flatten columns to MultiIndex → columns: symbol,factor
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "factor"])
    # Adopt future stack behavior to silence deprecation warning
    out = (
        df.stack("symbol", future_stack=True)
        .reset_index()
        .set_index(["date", "symbol"])
        .sort_index()
    )
    return out


def load_market_cap(path: Path) -> Optional[pd.DataFrame]:
    """
    Load historical market caps and return a Series of ``log_market_cap``.

    Expects a Parquet file with MultiIndex ``(date, ticker)`` and a ``market_cap`` column
    (as produced by ``scripts/fetch_shares_and_market_caps.py``).

    Returns ``None`` if the file is missing or empty.
    """
    if not path.exists():
        logger.warning("Market cap file not found at %s — skipping log_market_cap", path)
        return None
    mc = pd.read_parquet(path)
    if mc.empty:
        return None
    mc = mc.rename_axis(index={"ticker": "symbol"})
    mc["log_market_cap"] = np.log(mc["market_cap"].clip(lower=_EPS))
    return mc[["log_market_cap"]]


def merge_market_cap(
    factors: pd.DataFrame,
    market_cap_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Left-join ``log_market_cap`` onto a factor panel indexed by ``(date, symbol)``.

    If ``market_cap_path`` is ``None`` or the file is missing, returns factors unchanged.
    Timezone alignment is handled automatically (dates are stripped to tz-naive for the join
    if one side is tz-aware and the other is not).
    """
    if market_cap_path is None:
        return factors
    mc = load_market_cap(market_cap_path)
    if mc is None:
        return factors

    fi = factors.index.get_level_values("date")
    mi = mc.index.get_level_values("date")
    if fi.tz is not None and mi.tz is None:
        mc.index = mc.index.set_levels(mc.index.levels[0].tz_localize(fi.tz), level=0)
    elif fi.tz is None and mi.tz is not None:
        mc.index = mc.index.set_levels(mc.index.levels[0].tz_localize(None), level=0)
    elif fi.tz is not None and mi.tz is not None and str(fi.tz) != str(mi.tz):
        mc.index = mc.index.set_levels(mc.index.levels[0].tz_convert(fi.tz), level=0)

    merged = factors.join(mc, how="left")
    n_filled = merged["log_market_cap"].notna().sum()
    logger.info(
        "Merged log_market_cap: %d / %d rows filled (%.1f%%)",
        n_filled,
        len(merged),
        100 * n_filled / max(len(merged), 1),
    )
    return merged
