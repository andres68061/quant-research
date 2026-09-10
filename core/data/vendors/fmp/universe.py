"""Build a survivorship-bias-free US equity universe from FMP.

Two halves, and the second is the one that matters:

1. **Live names** from ``company-screener`` — every US-listed common stock above a
   market-cap floor, with sector/industry/exchange attached.
2. **Dead names** from ``delisted-companies`` — companies that were tradable in the
   past and no longer are.

A universe built from part 1 alone is the textbook survivorship trap: it silently
asks "how would this strategy have done, restricted to companies that survived to
today?". The screener cannot see them, so the delisted feed is not optional.

Membership is *coarse*: the screener reports today's market cap, so the cap floor
is a present-day filter, not a point-in-time one. Point-in-time eligibility still
has to come from the market-cap panel at backtest time — this module decides what
to **download**, not what is tradable on a given date.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from core.data.vendors.fmp.client import fmp_get
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

US_EXCHANGES: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX")

# The screener caps a single response; request generously and page by exchange.
_SCREENER_LIMIT = 10_000
_DELISTED_PAGE_SIZE = 100
_DELISTED_MAX_PAGES = 200

UNIVERSE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "company_name",
    "exchange",
    "sector",
    "industry",
    "market_cap",
    "is_delisted",
    "ipo_date",
    "delisted_date",
)


def fetch_live_universe(
    min_market_cap: float,
    exchanges: tuple[str, ...] = US_EXCHANGES,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch currently listed US common stocks above a market-cap floor.

    ETFs and funds are excluded at the API level; the screener's ``isEtf`` and
    ``isFund`` flags are more reliable than filtering on name patterns.

    Args:
        min_market_cap: Market-cap floor in USD, applied by the vendor.
        exchanges: Exchanges to include, queried one at a time so no single
            response approaches the row cap.
        api_key: Optional key override.

    Returns:
        DataFrame with :data:`UNIVERSE_COLUMNS`; ``is_delisted`` is False and
        ``delisted_date`` is NaT throughout.
    """
    frames: list[pd.DataFrame] = []
    for exchange in exchanges:
        rows = fmp_get(
            "company-screener",
            {
                "exchange": exchange,
                "isEtf": "false",
                "isFund": "false",
                "marketCapMoreThan": int(min_market_cap),
                "limit": _SCREENER_LIMIT,
            },
            api_key=api_key,
        )
        if not isinstance(rows, list):
            raise DataSchemaError(f"Unexpected screener payload for {exchange}: {type(rows)}")
        logger.info(
            "Screener %s: %d symbols above $%.0fM", exchange, len(rows), min_market_cap / 1e6
        )
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=list(UNIVERSE_COLUMNS))

    screened = pd.concat(frames, ignore_index=True)
    universe = pd.DataFrame(
        {
            "symbol": screened["symbol"],
            "company_name": screened.get("companyName"),
            "exchange": screened.get("exchangeShortName", screened.get("exchange")),
            "sector": screened.get("sector"),
            "industry": screened.get("industry"),
            "market_cap": pd.to_numeric(screened.get("marketCap"), errors="coerce"),
            "is_delisted": False,
            "ipo_date": pd.NaT,
            "delisted_date": pd.NaT,
        }
    )
    return universe


def fetch_delisted_universe(
    exchanges: tuple[str, ...] = US_EXCHANGES,
    api_key: Optional[str] = None,
    max_pages: int = _DELISTED_MAX_PAGES,
) -> pd.DataFrame:
    """
    Page through the delisted-companies feed and keep the US-listed names.

    Args:
        exchanges: Exchanges to keep; the feed is global and dominated by
            Canadian and other non-US venues.
        api_key: Optional key override.
        max_pages: Safety stop. Paging ends early on the first empty page.

    Returns:
        DataFrame with :data:`UNIVERSE_COLUMNS`; ``is_delisted`` is True and
        ``market_cap`` is NaN (the vendor reports none for dead names).
    """
    collected: list[dict[str, Any]] = []
    for page in range(max_pages):
        rows = fmp_get(
            "delisted-companies",
            {"page": page, "limit": _DELISTED_PAGE_SIZE},
            api_key=api_key,
        )
        if not isinstance(rows, list):
            raise DataSchemaError(f"Unexpected delisted payload on page {page}: {type(rows)}")
        if not rows:
            break
        collected.extend(rows)
        if page > 0 and page % 25 == 0:
            logger.info("Delisted feed: %d rows after %d pages", len(collected), page + 1)

    if not collected:
        return pd.DataFrame(columns=list(UNIVERSE_COLUMNS))

    delisted = pd.DataFrame(collected)
    logger.info("Delisted feed: %d rows total across all exchanges", len(delisted))
    delisted = delisted[delisted["exchange"].isin(exchanges)]
    logger.info("Delisted feed: %d rows on %s", len(delisted), ", ".join(exchanges))

    return pd.DataFrame(
        {
            "symbol": delisted["symbol"],
            "company_name": delisted.get("companyName"),
            "exchange": delisted["exchange"],
            "sector": pd.NA,
            "industry": pd.NA,
            "market_cap": float("nan"),
            "is_delisted": True,
            "ipo_date": pd.to_datetime(delisted.get("ipoDate"), errors="coerce"),
            "delisted_date": pd.to_datetime(delisted.get("delistedDate"), errors="coerce"),
        }
    )


def build_universe(
    min_market_cap: float,
    exchanges: tuple[str, ...] = US_EXCHANGES,
    extra_symbols: Optional[list[str]] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Combine live, delisted, and explicitly required symbols into one universe.

    Args:
        min_market_cap: Market-cap floor for the live screener leg.
        exchanges: US exchanges to include.
        extra_symbols: Symbols to keep regardless of the screener — pass the
            existing panel's columns so an expansion never silently drops a name
            that current research depends on.
        api_key: Optional key override.

    Returns:
        DataFrame with :data:`UNIVERSE_COLUMNS`, one row per symbol. When a symbol
        appears in both legs the live row wins (it is currently tradable).

    Example:
        >>> universe = build_universe(300e6, extra_symbols=["AAPL"])  # doctest: +SKIP
    """
    live = fetch_live_universe(min_market_cap, exchanges, api_key)
    delisted = fetch_delisted_universe(exchanges, api_key)

    combined = pd.concat([live, delisted], ignore_index=True)

    if extra_symbols:
        already = set(combined["symbol"])
        missing = sorted(set(extra_symbols) - already)
        if missing:
            logger.info("Adding %d symbols not returned by either feed", len(missing))
            combined = pd.concat(
                [
                    combined,
                    pd.DataFrame(
                        {
                            "symbol": missing,
                            "company_name": pd.NA,
                            "exchange": pd.NA,
                            "sector": pd.NA,
                            "industry": pd.NA,
                            "market_cap": float("nan"),
                            "is_delisted": pd.NA,
                            "ipo_date": pd.NaT,
                            "delisted_date": pd.NaT,
                        }
                    ),
                ],
                ignore_index=True,
            )

    # Live rows sort first, so keep="first" prefers the tradable record.
    combined = combined.sort_values("is_delisted", na_position="last", kind="stable")
    combined = combined.drop_duplicates("symbol", keep="first")
    return combined[list(UNIVERSE_COLUMNS)].sort_values("symbol").reset_index(drop=True)
