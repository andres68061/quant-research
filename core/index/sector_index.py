"""Point-in-time sector performance indices from the price panel.

Groups the universe by sector and compounds each group's daily return into an
index level, with membership resolved **daily**:

- a stock contributes only on days it actually traded (entry/exit and IPOs are
  handled by data availability, not by a static list),
- optionally, only while it was an S&P 500 member on that day (point-in-time
  membership filter — removes the "small stock joins the panel early" drift),
- cap-weighting uses the **prior day's** market cap, so a day's weight cannot
  embed that same day's return.

Biases handled and biases that remain (all disclosed via the caveat registry):
handled — survivorship (dead names contribute until their last traded day),
weight lookahead, entry/exit. NOT handled — sector labels are today's
(reclassifications leak backward), and a delisted stock exits at its last traded
price without the terminal loss a real holder of the final day would have taken.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.research.caveats import SURFACE_SECTOR_PERFORMANCE

logger = logging.getLogger(__name__)

MIN_MEMBERS_DEFAULT = 5

# Caveats live in the single registry (core/research/caveats.py), never inline —
# see the research-disclosure skill for why.
SECTOR_INDEX_SURFACE = SURFACE_SECTOR_PERFORMANCE


def compute_sector_indices(
    prices: pd.DataFrame,
    symbol_to_sector: pd.Series,
    market_cap: Optional[pd.DataFrame] = None,
    weighting: str = "cap",
    membership_filter: Optional[Callable[[pd.Timestamp], set[str]]] = None,
    start: Optional[pd.Timestamp] = None,
    min_members: int = MIN_MEMBERS_DEFAULT,
) -> dict[str, pd.DataFrame]:
    """
    Build daily sector index levels with day-by-day membership.

    Args:
        prices: Wide adjusted-close panel (date × symbol, tz-aware).
        symbol_to_sector: Symbol → sector label. Symbols missing from it (or
            labeled "Unknown") are excluded.
        market_cap: Wide date × symbol market caps; required for ``weighting="cap"``.
        weighting: ``"cap"`` (prior-day cap weights) or ``"equal"``.
        membership_filter: Optional ``date -> set of eligible symbols`` (e.g. the
            S&P 500 point-in-time filter). Applied per day on top of data
            availability.
        start: First date of the output; index levels start at 1.0 there.
        min_members: Days on which a sector has fewer members hold level flat
            (return treated as 0 rather than trusting a 2-stock "sector").

    Returns:
        Dict with:
        - ``levels``: date × sector index levels (start = 1.0),
        - ``members``: date × sector contributing-member counts.

    Raises:
        ValueError: On an unknown weighting or missing market caps for cap weighting.
    """
    if weighting not in ("cap", "equal"):
        raise ValueError(f"Unknown weighting {weighting!r}; use 'cap' or 'equal'")
    if weighting == "cap" and market_cap is None:
        raise ValueError("weighting='cap' requires a market_cap panel")

    working_prices = prices.loc[start:] if start is not None else prices
    if len(working_prices) < 2:
        raise ValueError("Not enough price history in the requested window")

    daily_returns = working_prices.pct_change(fill_method=None)

    labeled = symbol_to_sector.dropna()
    labeled = labeled[labeled != "Unknown"]
    symbols = [s for s in working_prices.columns if s in labeled.index]
    sectors = sorted(labeled.loc[symbols].unique())

    prior_cap: Optional[pd.DataFrame] = None
    if weighting == "cap":
        prior_cap = (
            market_cap.reindex(index=working_prices.index, columns=working_prices.columns)
            .ffill(limit=5)
            .shift(1)  # yesterday's cap weights today's return
        )

    level_frames: dict[str, pd.Series] = {}
    member_frames: dict[str, pd.Series] = {}

    for sector in sectors:
        sector_symbols = [s for s in symbols if labeled.loc[s] == sector]
        sector_returns = daily_returns[sector_symbols]

        eligible = sector_returns.notna()
        if membership_filter is not None:
            membership_mask = pd.DataFrame(
                False, index=sector_returns.index, columns=sector_symbols
            )
            for day in sector_returns.index:
                members_today = membership_filter(day)
                allowed = [s for s in sector_symbols if s in members_today]
                if allowed:
                    membership_mask.loc[day, allowed] = True
            eligible &= membership_mask

        if weighting == "cap":
            weights = prior_cap[sector_symbols].where(eligible)
            weight_sum = weights.sum(axis=1)
            normalized = weights.div(weight_sum.replace(0.0, np.nan), axis=0)
            sector_return = (sector_returns * normalized).sum(axis=1, min_count=1)
        else:
            masked = sector_returns.where(eligible)
            sector_return = masked.mean(axis=1)

        member_count = eligible.sum(axis=1)
        sector_return = sector_return.where(member_count >= min_members, 0.0).fillna(0.0)
        # First row's pct_change is NaN by construction; level starts at 1.0.
        sector_return.iloc[0] = 0.0

        level_frames[sector] = (1.0 + sector_return).cumprod()
        member_frames[sector] = member_count

    return {
        "levels": pd.DataFrame(level_frames),
        "members": pd.DataFrame(member_frames),
    }
