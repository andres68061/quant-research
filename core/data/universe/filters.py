"""Universe eligibility filters — what belongs in a cross-section, and what does not.

The expanded universe (ADR 0013) is survivorship-aware, which is the point. It is
also full of things that trade but are not operating companies:

- **Shell companies / SPACs** — 1,779 of ~9,000 symbols (20%), measured
  2026-08-12. A pre-merger SPAC is a trust account with a ticker: it sits within
  pennies of $10.00 (measured annualized vol under 5% for many), then either
  merges into an unrelated business or liquidates. Ranking it on momentum,
  value, or profitability is meaningless — the "stock" has no operations, and
  its near-zero volatility makes it look like an extreme low-vol name.
- **Closed-end funds and trusts** — hold securities rather than run a business;
  their "fundamentals" describe a portfolio, not a company.

Leaving these in does not just add noise: low-vol, flat-price, near-zero-turnover
instruments cluster at the extremes of exactly the factors most sensitive to
them, so they land in the traded tiers rather than the middle.

These filters are **not** applied at download time (the raw layer stays complete)
and **not** silently at load time (a filter you cannot see is a filter you cannot
audit). Backtests opt in, and the choice is disclosed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SECTORS_PATH = Path("data/sectors/sector_classifications.parquet")
DEFAULT_MEMBERSHIP_PATH = Path("data/universe/index_membership.parquet")

# Industry labels that identify non-operating listed vehicles.
NON_OPERATING_INDUSTRIES: frozenset[str] = frozenset(
    {
        "Shell Companies",
        "Asset Management - Income",
        "Asset Management - Bonds",
        "Asset Management - Global",
    }
)


def load_non_operating_symbols(
    sectors_path: Path = DEFAULT_SECTORS_PATH,
    industries: frozenset[str] = NON_OPERATING_INDUSTRIES,
) -> set[str]:
    """
    Symbols whose industry marks them as non-operating vehicles.

    Args:
        sectors_path: Sector classification parquet.
        industries: Industry labels to treat as non-operating.

    Returns:
        Set of symbols to exclude. Empty when the file is missing (the caller
        gets an unfiltered universe rather than a silent crash, and the count is
        logged so the absence is visible).
    """
    if not Path(sectors_path).exists():
        logger.warning("No sector file at %s; cannot filter non-operating symbols", sectors_path)
        return set()

    classifications = pd.read_parquet(sectors_path)
    excluded = classifications[classifications["industry"].isin(industries)]
    logger.info(
        "Non-operating filter: %d symbols excluded (%s)",
        len(excluded),
        excluded["industry"].value_counts().to_dict(),
    )
    return set(excluded["symbol"])


def load_membership_filter(
    index_name: str = "sp500",
    membership_path: Path = DEFAULT_MEMBERSHIP_PATH,
) -> Callable[[pd.Timestamp], set[str]]:
    """
    Build a point-in-time index-membership filter from the labeling table.

    This is the ADR-0013 replacement for "membership = which price panel you
    loaded": the canonical panel is now the whole market, so restricting to an
    index is an explicit, dated lookup.

    Args:
        index_name: Index label in the membership table.
        membership_path: Membership intervals parquet.

    Returns:
        ``date -> set of member symbols`` on that date.

    Raises:
        FileNotFoundError: If the membership table is missing — silently
            returning "everything is a member" would be a survivorship bug.
    """
    path = Path(membership_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No membership table at {path}; run scripts/build/build_index_membership.py"
        )

    intervals = pd.read_parquet(path)
    intervals = intervals[intervals["index_name"] == index_name]
    starts = pd.to_datetime(intervals["valid_from"]).to_numpy()
    ends = pd.to_datetime(intervals["valid_to"]).to_numpy()
    symbols = intervals["symbol"].to_numpy()

    def members_on(date: pd.Timestamp) -> set[str]:
        as_of = pd.Timestamp(date).tz_localize(None) if date.tzinfo else pd.Timestamp(date)
        active = (starts <= as_of.to_datetime64()) & (
            pd.isna(ends) | (ends >= as_of.to_datetime64())
        )
        return set(symbols[active])

    return members_on


def build_universe_filter(
    panel_symbols: pd.Index,
    exclude_non_operating: bool = True,
    index_name: Optional[str] = None,
    sectors_path: Path = DEFAULT_SECTORS_PATH,
    membership_path: Path = DEFAULT_MEMBERSHIP_PATH,
) -> Callable[[pd.Timestamp], set[str]]:
    """
    Compose the eligibility rules into one ``date -> eligible symbols`` callable.

    ``panel_symbols`` is required so that "the whole universe minus the excluded
    set" is expressible as a concrete set on every date — a filter callable has
    to return members, not a complement.

    Args:
        panel_symbols: Columns of the price panel being traded.
        exclude_non_operating: Drop shells/SPACs/fund vehicles.
        index_name: Restrict to an index's members on each date (e.g. ``"sp500"``);
            None means the whole panel.
        sectors_path: Sector classification parquet.
        membership_path: Membership intervals parquet.

    Returns:
        Callable usable as ``universe_filter`` in the factor runner.

    Raises:
        ValueError: If no filter is requested at all (pass ``None`` instead).

    Example:
        >>> eligible = build_universe_filter(prices.columns, index_name="sp500")
        ... # doctest: +SKIP
    """
    excluded = load_non_operating_symbols(sectors_path) if exclude_non_operating else set()
    membership = load_membership_filter(index_name, membership_path) if index_name else None

    if membership is None and not excluded:
        raise ValueError("No filter requested; pass universe_filter=None instead")

    tradable = set(panel_symbols) - excluded

    def eligible_on(date: pd.Timestamp) -> set[str]:
        if membership is None:
            return tradable
        return membership(date) & tradable

    return eligible_on
