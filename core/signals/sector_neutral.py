"""Sector-neutral factor transforms for cross-sectional strategies.

FMP (and previously yfinance) only expose *today's* sector label. Applying that
label historically is mild lookahead — documented and accepted for research
until a PIT sector source exists.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def demean_factor_within_sector(
    factor: pd.Series,
    symbol_to_sector: pd.Series,
) -> pd.Series:
    """
    Subtract the cross-sectional sector mean of ``factor`` on each date.

    Args:
        factor: MultiIndex ``(date, symbol)`` Series of factor values.
        symbol_to_sector: Index = symbol, values = sector label.

    Returns:
        Same index as ``factor``; NaN where the original factor is NaN or the
        sector group has fewer than 2 non-NaN names on that date.
    """
    if not isinstance(factor.index, pd.MultiIndex):
        raise TypeError("factor must have MultiIndex (date, symbol)")

    work = factor.rename("value").to_frame()
    sector_labels = work.index.get_level_values("symbol").map(symbol_to_sector)
    work = work.assign(_sector=sector_labels)
    date_level = work.index.get_level_values("date")
    grouped = work.groupby([date_level, "_sector"], dropna=False)["value"]
    counts = grouped.transform("count")
    means = grouped.transform("mean")
    demeaned = factor - means
    demeaned = demeaned.where(counts >= 2)
    demeaned.name = factor.name
    return demeaned


def zscore_cross_section(factor: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score of ``factor`` within each date.

    Groups with fewer than 5 non-NaN names return all NaN for that date.
    """
    if not isinstance(factor.index, pd.MultiIndex):
        raise TypeError("factor must have MultiIndex (date, symbol)")

    def _z(group: pd.Series) -> pd.Series:
        valid = group.dropna()
        if len(valid) < 5:
            return pd.Series(float("nan"), index=group.index)
        std = float(valid.std(ddof=0))
        if std == 0.0 or pd.isna(std):
            return pd.Series(float("nan"), index=group.index)
        return (group - float(valid.mean())) / std

    return factor.groupby(level="date", group_keys=False).apply(_z)


def combine_value_quality(
    earnings_yield: pd.Series,
    roe: pd.Series,
    *,
    symbol_to_sector: Optional[pd.Series] = None,
    sector_neutral: bool = False,
) -> pd.Series:
    """
    Equal-weight composite of cross-sectional z(earnings yield) and z(ROE).

    When ``sector_neutral`` is True, each leg is demeaned within sector before
    z-scoring (requires ``symbol_to_sector``).
    """
    ey = earnings_yield
    quality = roe
    if sector_neutral:
        if symbol_to_sector is None:
            raise ValueError("symbol_to_sector required when sector_neutral=True")
        ey = demean_factor_within_sector(ey, symbol_to_sector)
        quality = demean_factor_within_sector(quality, symbol_to_sector)
    composite = (zscore_cross_section(ey) + zscore_cross_section(quality)) / 2.0
    composite.name = "value_quality"
    return composite
