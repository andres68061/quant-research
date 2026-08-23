#!/usr/bin/env python3
"""
Build sector/industry labels for the whole universe, not just the S&P names.

Before the ADR-0013 cutover the sector file held 929 symbols — fine for a
774-name panel, useless for an 8,908-symbol one. Sector-neutral factors and the
sector performance page silently dropped every unlabeled name.

Three sources, in priority order:

1. **Universe table** (`sector`/`industry` from the screener) — covers live names
   at no API cost.
2. **Existing sector file** — preserves any hand-checked labels already present.
3. **`profile` endpoint** — one call per still-unlabeled symbol, which is how
   delisted names get labels (the screener cannot see them).

Output keeps the existing schema so every consumer keeps working:
``symbol, sector, industry, industryKey, sectorKey, last_updated, quoteType``.

**Known limitation this does NOT fix:** labels are current-as-of-fetch. FMP
exposes no historical classification, so a company reclassified in 2020 carries
that label back through all history (registry flaw `sector-labels-current-only`).

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build_sector_classifications.py
    /opt/anaconda3/envs/quant/bin/python scripts/build_sector_classifications.py --no-fetch
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.client import fmp_get
from core.data.fmp.storage import write_atomic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_sector_classifications")

UNIVERSE_FILE = ROOT / "data" / "raw" / "fmp" / "universe" / "us_equity_universe.parquet"
SECTORS_OUT = ROOT / "data" / "sectors" / "sector_classifications.parquet"
PROFILE_CACHE = ROOT / "data" / "raw" / "fmp" / "profiles"

OUTPUT_COLUMNS = [
    "symbol",
    "sector",
    "industry",
    "industryKey",
    "sectorKey",
    "last_updated",
    "quoteType",
]
_PROGRESS_EVERY = 250


def fetch_profile_sectors(symbols: list[str]) -> pd.DataFrame:
    """
    Fetch sector/industry via the profile endpoint, caching one file per symbol.

    Resumable like every other fetcher: a symbol with a cached profile is not
    refetched, so an interrupted run costs nothing on restart.

    Args:
        symbols: Symbols still lacking a label.

    Returns:
        DataFrame with ``symbol``, ``sector``, ``industry``.
    """
    PROFILE_CACHE.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    started = time.monotonic()

    for i, symbol in enumerate(symbols, 1):
        cache_path = PROFILE_CACHE / f"{symbol.replace('/', '-')}.parquet"
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            if not cached.empty:
                records.append(
                    {
                        "symbol": symbol,
                        "sector": cached.iloc[0].get("sector"),
                        "industry": cached.iloc[0].get("industry"),
                    }
                )
            continue

        try:
            rows = fmp_get("profile", {"symbol": symbol})
        except Exception:
            logger.warning("profile failed for %s", symbol)
            continue

        frame = pd.DataFrame(rows if isinstance(rows, list) else [rows])
        write_atomic(frame, cache_path)
        if not frame.empty:
            records.append(
                {
                    "symbol": symbol,
                    "sector": frame.iloc[0].get("sector"),
                    "industry": frame.iloc[0].get("industry"),
                }
            )

        if i % _PROGRESS_EVERY == 0:
            rate = i / max(time.monotonic() - started, 1e-9)
            logger.info(
                "profiles [%d/%d] (~%.0f min left)", i, len(symbols), (len(symbols) - i) / rate / 60
            )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-universe sector labels")
    parser.add_argument("--no-fetch", action="store_true", help="Use cached/screener data only")
    args = parser.parse_args()

    universe = pd.read_parquet(UNIVERSE_FILE)
    labels = universe[["symbol", "sector", "industry"]].copy()

    if SECTORS_OUT.exists():
        existing = pd.read_parquet(SECTORS_OUT)[["symbol", "sector", "industry"]]
        existing = existing[existing["sector"].notna() & (existing["sector"] != "Unknown")]
        labels = labels.set_index("symbol")
        existing = existing.set_index("symbol")
        # Existing hand-checked labels win where the screener has nothing.
        labels["sector"] = labels["sector"].fillna(existing["sector"])
        labels["industry"] = labels["industry"].fillna(existing["industry"])
        labels = labels.reset_index()
        # Symbols only in the old file (e.g. indexes/ETFs we track) are kept.
        extra = existing[~existing.index.isin(labels["symbol"])].reset_index()
        labels = pd.concat([labels, extra], ignore_index=True)

    unlabeled = labels[labels["sector"].isna()]["symbol"].tolist()
    logger.info("%d symbols labeled, %d unlabeled", len(labels) - len(unlabeled), len(unlabeled))

    if unlabeled and not args.no_fetch:
        logger.info("Fetching %d profiles (~%.0f min)", len(unlabeled), len(unlabeled) / 500)
        fetched = fetch_profile_sectors(unlabeled)
        if not fetched.empty:
            fetched = fetched.set_index("symbol")
            labels = labels.set_index("symbol")
            labels["sector"] = labels["sector"].fillna(fetched["sector"])
            labels["industry"] = labels["industry"].fillna(fetched["industry"])
            labels = labels.reset_index()

    labels["sector"] = labels["sector"].fillna("Unknown")
    labels["industry"] = labels["industry"].fillna("Unknown")
    labels["sectorKey"] = labels["sector"].str.lower().str.replace(" ", "-", regex=False)
    labels["industryKey"] = labels["industry"].str.lower().str.replace(" ", "-", regex=False)
    labels["last_updated"] = pd.Timestamp.now().normalize()
    labels["quoteType"] = "EQUITY"
    labels = labels.drop_duplicates("symbol", keep="first")[OUTPUT_COLUMNS]

    SECTORS_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(labels, SECTORS_OUT)
    known = labels[labels["sector"] != "Unknown"]
    logger.info(
        "Wrote %s: %d symbols (%d labeled, %d Unknown)",
        SECTORS_OUT.name,
        len(labels),
        len(known),
        len(labels) - len(known),
    )
    logger.info("Distribution: %s", known["sector"].value_counts().head(12).to_dict())


if __name__ == "__main__":
    main()
