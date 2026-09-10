"""Partition keys for FMP endpoints that are not keyed by ticker.

Most of the vendor surface is per-symbol, and the symbol universe comes from the
security master. A minority is keyed by something else — a CIK, a legislator's
name, an economic series name — and those key sets are themselves things the
vendor tells us, in wave-1 downloads. Deriving them here rather than hard-coding
them keeps the catalog honest: an endpoint gets the keys the data supports, and
an endpoint whose key source has not been downloaded yet plans no tasks and says
so, instead of silently appearing complete.

The scoping decision worth stating: per-CIK endpoints run over the CIKs of *our*
universe, taken from company profiles, not over every SEC registrant. FMP lists
491,000 CIKs; five per-CIK endpoints across all of them would be 2.45 million
requests, about 68 hours, almost all of it describing entities that will never
enter a backtest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Economic series accepted by the `economic-indicators` endpoint, confirmed by
# probing each one; `inflation` and the certificates-of-deposit series are
# documented but return no rows on this subscription.
ECONOMIC_INDICATORS: tuple[str, ...] = (
    "GDP",
    "realGDP",
    "nominalPotentialGDP",
    "realGDPPerCapita",
    "federalFunds",
    "CPI",
    "inflationRate",
    "retailSales",
    "consumerSentiment",
    "durableGoods",
    "unemploymentRate",
    "totalNonfarmPayroll",
    "initialClaims",
    "industrialProductionTotalIndex",
    "newPrivatelyOwnedHousingUnitsStartedTotalUnits",
    "totalVehicleSales",
    "retailMoneyFunds",
    "smoothedUSRecessionProbabilities",
    "30YearFixedRateMortgageAverage",
    "15YearFixedRateMortgageAverage",
    "commercialBankInterestRateOnCreditCardPlansAllAccounts",
)

# A guard against pulling the whole SEC registrant list into a run by accident.
MAX_CIK_KEYS = 20_000


def _read_global(raw_root: Path, dataset: str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Read a wave-1 global table, or an empty frame when it has not landed."""
    path = raw_root / dataset / "_all.parquet"
    if not path.is_file():
        logger.info("%s not downloaded yet; keys derived from it will be empty", dataset)
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:  # noqa: BLE001 - a partial download is not fatal here
        logger.warning("could not read %s: %s", path, exc)
        return pd.DataFrame()


def universe_ciks(raw_root: Path, limit: int = MAX_CIK_KEYS) -> list[str]:
    """
    CIKs of the companies in our own universe, from downloaded company profiles.

    Args:
        raw_root: Vendor raw layer root.
        limit: Safety cap on how many keys to return.

    Returns:
        Sorted unique zero-padded CIK strings. Empty when no profiles exist yet.
    """
    directory = raw_root / "profiles"
    if not directory.is_dir():
        return []
    ciks: set[str] = set()
    for path in directory.glob("*.parquet"):
        if path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_parquet(path, columns=["cik"])
        except Exception:  # noqa: BLE001 - a file without a cik column is not an error
            continue
        for value in frame["cik"].dropna().unique():
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none"}:
                ciks.add(text.zfill(10))
    ordered = sorted(ciks)
    if len(ordered) > limit:
        logger.warning("found %d CIKs; capping at %d", len(ordered), limit)
        return ordered[:limit]
    return ordered


def legislator_names(raw_root: Path) -> list[str]:
    """
    Surnames of legislators seen in recent Senate and House disclosures.

    The by-name trade endpoints take a person, and the only list of people the
    vendor gives us is whoever appears in the latest disclosures.

    Args:
        raw_root: Vendor raw layer root.

    Returns:
        Sorted unique surnames.
    """
    names: set[str] = set()
    for dataset in ("senate_latest", "house_latest"):
        frame = _read_global(raw_root, dataset)
        if frame.empty or "lastName" not in frame.columns:
            continue
        for value in frame["lastName"].dropna().unique():
            text = str(value).strip()
            if text:
                names.add(text)
    return sorted(names)


def issuer_names(raw_root: Path) -> list[str]:
    """
    Company names usable as search keys for the fundraising and M&A endpoints.

    Args:
        raw_root: Vendor raw layer root.

    Returns:
        Sorted unique company names.
    """
    names: set[str] = set()
    for dataset, column in (
        ("crowdfunding_offerings_latest", "companyName"),
        ("fundraising_latest", "companyName"),
        ("mergers_acquisitions_latest", "companyName"),
    ):
        frame = _read_global(raw_root, dataset)
        if frame.empty or column not in frame.columns:
            continue
        for value in frame[column].dropna().unique():
            text = str(value).strip()
            if text:
                names.add(text)
    return sorted(names)


def insider_reporting_names(raw_root: Path, limit: int = 5000) -> list[str]:
    """
    People and entities that have filed Form 4s for symbols we already hold.

    Args:
        raw_root: Vendor raw layer root.
        limit: Safety cap; the long tail of one-off filers is unbounded.

    Returns:
        Sorted unique reporting names.
    """
    directory = raw_root / "insider_trading"
    if not directory.is_dir():
        return []
    names: set[str] = set()
    for path in directory.glob("*.parquet"):
        if path.stat().st_size == 0 or len(names) >= limit:
            continue
        try:
            frame = pd.read_parquet(path, columns=["reportingName"])
        except Exception:  # noqa: BLE001 - column absent in older files
            continue
        for value in frame["reportingName"].dropna().unique():
            text = str(value).strip()
            if text:
                names.add(text)
    return sorted(names)[:limit]
