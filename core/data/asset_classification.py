"""
Asset type classification.

Categorises tickers as Stock, ETF, Commodity, Index, etc. using
sector-classification metadata with a pattern-matching fallback.
"""

from typing import Optional

import pandas as pd

COMMON_ETFS = frozenset(
    [
        "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "AGG", "BND",
        "GLD", "SLV", "USO", "TLT", "EEM", "VWO", "XLE", "XLF",
        "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
    ]
)

_QUOTE_TYPE_MAP = {
    "EQUITY": "Stock",
    "ETF": "ETF",
    "INDEX": "Index",
    "MUTUALFUND": "Fund",
    "CRYPTOCURRENCY": "Crypto",
}


def categorize_asset_type(
    symbol: str,
    df_sectors: Optional[pd.DataFrame] = None,
) -> str:
    """
    Categorize a symbol as Stock, ETF, Commodity, Index, etc.

    Uses quoteType from sector-classification data when available;
    falls back to pattern matching.

    Args:
        symbol: Ticker symbol
        df_sectors: DataFrame with sector classifications (optional).
                    Must contain 'symbol' and 'quoteType' columns.

    Returns:
        One of 'Stock', 'ETF', 'Commodity', 'Index', 'Fund', 'Crypto',
        or the raw quoteType string if unmapped.
    """
    if df_sectors is not None and "quoteType" in df_sectors.columns:
        row = df_sectors[df_sectors["symbol"] == symbol]
        if not row.empty:
            quote_type = row.iloc[0]["quoteType"]
            mapped = _QUOTE_TYPE_MAP.get(quote_type)
            if mapped:
                return mapped
            if quote_type != "Unknown":
                return quote_type

    if "=F" in symbol:
        return "Commodity"
    if symbol.startswith("^"):
        return "Index"
    if symbol in COMMON_ETFS:
        return "ETF"

    return "Stock"
