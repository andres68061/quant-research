"""Which symbols the API loads into memory, and the honest reason for the choice.

After the ADR-0013 cutover the canonical panel holds ~8,900 symbols. Loading all
of it plus the factor panels costs roughly **7 GB** of process memory (measured:
0.75 GB prices + ~5.2 GB fundamentals factors + ~1 GB price factors), which an
interactive API cannot carry.

The wrong fix is "load fewer rows" chosen arbitrarily — a silent, undisclosed
universe change makes every result on the page describe a population nobody
declared. The right fix is an **explicit, disclosed policy**: the API states
which universe it loaded and why, and the UI can show it.

Policies:

``research`` (default)
    Operating companies with enough history to rank: excludes shells/SPACs and
    fund vehicles, and requires a minimum number of observed trading days. This
    is the tradeable cross-section — the names a factor strategy could actually
    hold — and is what most research questions mean by "the universe".
``sp500``
    Symbols that were S&P 500 members at any point. Smallest and fastest;
    reproduces pre-cutover results.
``full``
    Everything in the panel. Correct but memory-hungry; for batch jobs and
    machines with headroom, not the default API process.

Scripts are unaffected — they read the parquet files directly and should
continue to work on the full universe.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

Policy = Literal["research", "sp500", "full"]

DEFAULT_POLICY: Policy = "research"
ENV_VAR = "API_UNIVERSE"

# A symbol needs enough history for a 252-day factor to exist at all, plus a
# margin so the first rankable date is not also the last.
MIN_TRADING_DAYS = 300


def resolve_policy(explicit: Optional[str] = None) -> Policy:
    """
    Resolve the active policy from an argument, then the environment, then default.

    Args:
        explicit: Caller-supplied policy name, if any.

    Returns:
        A valid policy name; unknown values fall back to the default with a
        warning rather than raising, so a typo in an env var cannot take the API
        down.
    """
    candidate = (explicit or os.getenv(ENV_VAR) or DEFAULT_POLICY).lower()
    if candidate not in ("research", "sp500", "full"):
        logger.warning("Unknown %s=%r; falling back to %r", ENV_VAR, candidate, DEFAULT_POLICY)
        return DEFAULT_POLICY
    return candidate  # type: ignore[return-value]


def select_api_symbols(
    prices: pd.DataFrame,
    policy: Optional[str] = None,
    sectors_path: Path = Path("data/sectors/sector_classifications.parquet"),
    membership_path: Path = Path("data/universe/index_membership.parquet"),
) -> tuple[list[str], dict[str, object]]:
    """
    Choose the symbols the API should hold in memory under the active policy.

    Args:
        prices: The full canonical price panel.
        policy: Override for the resolved policy.
        sectors_path: Sector classifications, for the non-operating exclusion.
        membership_path: Index membership intervals, for the ``sp500`` policy.

    Returns:
        ``(symbols, disclosure)`` where ``disclosure`` records the policy, the
        counts at each filtering step, and the reason — so the API can serve it
        and the UI can show which universe produced a number.
    """
    active = resolve_policy(policy)
    all_symbols = [str(c) for c in prices.columns]
    # Index/benchmark tickers are always kept: beta needs the market series.
    benchmarks = [s for s in all_symbols if s.startswith("^")]

    disclosure: dict[str, object] = {
        "policy": active,
        "panel_symbols": len(all_symbols),
        "steps": [],
    }

    if active == "full":
        disclosure["loaded_symbols"] = len(all_symbols)
        disclosure["reason"] = "No filtering; the whole canonical panel is in memory."
        return all_symbols, disclosure

    if active == "sp500":
        from core.data.universe_filters import load_membership_filter

        try:
            members_on = load_membership_filter("sp500", membership_path)
        except FileNotFoundError:
            logger.warning("No membership table; falling back to the research policy")
            return select_api_symbols(prices, "research", sectors_path, membership_path)

        ever_member: set[str] = set()
        for date in prices.index[::63]:  # quarterly sampling is enough for "ever"
            ever_member |= members_on(date)
        selected = sorted((set(all_symbols) & ever_member) | set(benchmarks))
        disclosure["steps"].append({"step": "S&P 500 members (ever)", "kept": len(selected)})
        disclosure["loaded_symbols"] = len(selected)
        disclosure["reason"] = (
            "Symbols that were S&P 500 members at some point. Reproduces pre-cutover "
            "results; excludes the small-cap tail entirely."
        )
        return selected, disclosure

    # research policy
    from core.data.universe_filters import load_non_operating_symbols

    excluded = load_non_operating_symbols(sectors_path)
    after_operating = [s for s in all_symbols if s not in excluded]
    disclosure["steps"].append(
        {
            "step": "exclude shells/SPACs/fund vehicles",
            "removed": len(all_symbols) - len(after_operating),
            "kept": len(after_operating),
        }
    )

    observed = prices[after_operating].notna().sum()
    long_enough = observed[observed >= MIN_TRADING_DAYS].index.tolist()
    selected = sorted(set(long_enough) | set(benchmarks))
    disclosure["steps"].append(
        {
            "step": f"require >= {MIN_TRADING_DAYS} trading days of history",
            "removed": len(after_operating) - len(long_enough),
            "kept": len(selected),
        }
    )
    disclosure["loaded_symbols"] = len(selected)
    disclosure["reason"] = (
        "Operating companies with enough history to rank. Excludes non-operating "
        "vehicles (a pre-merger SPAC has no factor exposure) and symbols too "
        "short-lived for a 252-day factor. Set API_UNIVERSE=full to load everything."
    )
    return selected, disclosure
