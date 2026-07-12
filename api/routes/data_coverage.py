"""Data coverage and quality endpoint: dataset inventory, universe coverage, quarantine."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter

from api.dependencies import get_prices, get_quarantined_symbols
from api.schemas.data_coverage import (
    DataCoverageResponse,
    DatasetInfo,
    QuarantineEntry,
    SP500CsvInfo,
    YearCoverage,
)
from config.settings import PROJECT_ROOT
from core.backtest.portfolio import sp500_universe_filter
from core.data.quality import QUARANTINE_PATH, load_quarantine_list
from core.data.sp500_constituents import resolve_sp500_historical_csv
from core.exceptions import ConfigError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-coverage", tags=["data-coverage"])

# name -> (relative path, source vendor, layer, description)
_DATASET_CATALOG: dict[str, tuple[str, str, str, str]] = {
    "prices": (
        "data/factors/prices.parquet",
        "FMP (dividend-adjusted EOD)",
        "derived",
        "Wide adjusted-close panel; the platform's canonical stock price layer",
    ),
    "raw_fmp_prices": (
        "data/raw/fmp/prices",
        "FMP (dividend-adjusted EOD)",
        "raw",
        "Immutable per-symbol raw price files (source of truth for prices)",
    ),
    "factors_price": (
        "data/factors/factors_price.parquet",
        "derived from prices",
        "derived",
        "MultiIndex (date, symbol) factor panel: momentum, reversal, vol, beta",
    ),
    "factors_all": (
        "data/factors/factors_all.parquet",
        "derived from prices + FMP market caps",
        "derived",
        "Factor panel plus market-cap columns",
    ),
    "raw_fmp_fundamentals": (
        "data/raw/fmp/fundamentals/income_statement",
        "FMP (quarterly statements)",
        "raw",
        "Immutable quarterly income statements with filingDate/acceptedDate",
    ),
    "fundamentals_pit": (
        "data/factors/fundamentals.parquet",
        "derived from FMP statements",
        "derived",
        "Point-in-time fundamentals panel: metrics visible only after their filing date",
    ),
    "factors_fundamental": (
        "data/factors/factors_fundamental.parquet",
        "derived from fundamentals + market caps",
        "derived",
        "Valuation/quality factors: book_to_market, earnings_yield, roe, asset_growth",
    ),
    "macro_raw": (
        "data/raw/macro_fred.parquet",
        "FRED (latest values; not ALFRED vintages)",
        "raw",
        "Long-format raw macro series, native frequency, no lag applied",
    ),
    "macro": (
        "data/factors/macro.parquet",
        "derived from FRED raw (fixed pub lags, not true vintages)",
        "derived",
        "Business-day macro panel with fixed publication lag applied",
    ),
    "fama_french_5": (
        "data/factors/fama_french_5.parquet",
        "Kenneth French data library",
        "raw",
        "FF5 daily factor returns",
    ),
    "market_caps": (
        "data/market_caps/historical_market_caps.parquet",
        "FMP (historical-market-capitalization)",
        "derived",
        "Historical market capitalizations (true shares × price from FMP)",
    ),
    "sectors": (
        "data/sectors/sector_classifications.parquet",
        "FMP (/profile) — today's sector/industry (not point-in-time)",
        "raw",
        "Sector and industry classifications",
    ),
    "vix": (
        "data/factors/vix.parquet",
        "FMP (historical-price-eod/full ^VIX)",
        "raw",
        "VIX close history",
    ),
    "commodities": (
        "data/commodities/prices.parquet",
        "FMP (ETFs + futures)",
        "raw",
        "Commodity futures price history",
    ),
}


def _describe_parquet(path: Path) -> tuple[int, int, Optional[str], Optional[str]]:
    """Return (rows, columns, first_date, last_date) for a parquet file."""
    frame = pd.read_parquet(path)
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        index = index.get_level_values(0)
    if isinstance(index, pd.DatetimeIndex) and len(index) > 0:
        first_date = str(index.min().date())
        last_date = str(index.max().date())
    return len(frame), frame.shape[1], first_date, last_date


def _sp500_csv_info() -> Optional[SP500CsvInfo]:
    """Resolve the live membership CSV and report file age + last snapshot date."""
    try:
        path = resolve_sp500_historical_csv()
    except ConfigError as exc:
        logger.warning("S&P CSV not found for coverage: %s", exc)
        return None

    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - mtime).total_seconds() / 86400.0
    last_membership_date: Optional[str] = None
    n_snapshots: Optional[int] = None
    try:
        frame = pd.read_csv(path, usecols=["date"])
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if len(dates) > 0:
            last_membership_date = str(dates.max().date())
            n_snapshots = int(len(dates))
    except Exception as exc:  # noqa: BLE001 — best-effort metadata only
        logger.warning("Could not peek S&P CSV dates: %s", exc)

    rel = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    return SP500CsvInfo(
        filename=path.name,
        path=rel,
        file_mtime_utc=mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        age_days=round(age_days, 1),
        last_membership_date=last_membership_date,
        n_snapshots=n_snapshots,
    )


def _dataset_infos(sp500: Optional[SP500CsvInfo] = None) -> list[DatasetInfo]:
    infos: list[DatasetInfo] = []
    for name, (rel_path, source, layer, description) in _DATASET_CATALOG.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        if path.is_dir():
            files = list(path.glob("*.parquet"))
            size_mb = sum(f.stat().st_size for f in files) / 1e6
            infos.append(
                DatasetInfo(
                    name=name,
                    source=source,
                    path=rel_path,
                    layer=layer,
                    rows=len(files),
                    columns=0,
                    size_mb=round(size_mb, 1),
                    description=f"{description} ({len(files)} symbol files)",
                )
            )
            continue
        size_mb = path.stat().st_size / 1e6
        if path.suffix == ".parquet":
            rows, columns, first_date, last_date = _describe_parquet(path)
        else:
            rows, columns, first_date, last_date = 0, 0, None, None
        infos.append(
            DatasetInfo(
                name=name,
                source=source,
                path=rel_path,
                layer=layer,
                rows=rows,
                columns=columns,
                first_date=first_date,
                last_date=last_date,
                size_mb=round(size_mb, 1),
                description=description,
            )
        )

    if sp500 is not None:
        infos.append(
            DatasetInfo(
                name="sp500_constituents",
                source="fja05680/sp500 Updated CSV (FMP = cross-check only)",
                path=sp500.path,
                layer="raw",
                rows=sp500.n_snapshots or 0,
                columns=2,
                first_date=None,
                last_date=sp500.last_membership_date,
                size_mb=round((PROJECT_ROOT / sp500.path).stat().st_size / 1e6, 1),
                description=(
                    f"Point-in-time S&P 500 membership; file age {sp500.age_days:.0f}d "
                    f"(mtime {sp500.file_mtime_utc[:10]})"
                ),
            )
        )
    return infos


def _coverage_by_year(prices: pd.DataFrame) -> list[YearCoverage]:
    """Symbols with >=1 price per year vs point-in-time S&P 500 membership mid-year."""
    membership = sp500_universe_filter()
    stock_columns = [c for c in prices.columns if not c.startswith("^")]
    has_data = prices[stock_columns].notna()
    yearly_symbols = has_data.groupby(has_data.index.year).any().sum(axis=1)

    coverage: list[YearCoverage] = []
    for year, n_symbols in yearly_symbols.items():
        members = membership(pd.Timestamp(f"{year}-06-30", tz=prices.index.tz))
        n_members = len(members)
        covered = len(members & set(stock_columns)) if members else 0
        coverage.append(
            YearCoverage(
                year=int(year),
                symbols_with_data=int(n_symbols),
                sp500_members=n_members,
                coverage_pct=round(100 * covered / n_members, 1) if n_members else 0.0,
            )
        )
    return coverage


@router.get("", response_model=DataCoverageResponse)
def data_coverage() -> DataCoverageResponse:
    """Full inventory: datasets, per-year S&P coverage, and the quarantine list."""
    prices = get_prices()
    coverage = _coverage_by_year(prices) if prices is not None else []
    sp500 = _sp500_csv_info()

    quarantine = load_quarantine_list(PROJECT_ROOT / QUARANTINE_PATH)
    if "review_note" not in quarantine.columns:
        quarantine["review_note"] = ""
    entries = [
        QuarantineEntry(
            symbol=row.symbol,
            check=row.check,
            value=round(float(row.value), 4),
            detail=row.detail,
            status=row.status,
            review_note=str(row.review_note or ""),
        )
        for row in quarantine.itertuples()
    ]
    flagged = (
        set(quarantine.loc[quarantine["status"] == "flagged", "symbol"])
        if not quarantine.empty
        else set()
    )

    age_note = ""
    if sp500 is not None:
        age_note = (
            f" Live CSV: {sp500.filename} (mtime age {sp500.age_days:.0f}d, "
            f"last membership row {sp500.last_membership_date})."
        )

    return DataCoverageResponse(
        datasets=_dataset_infos(sp500),
        coverage_by_year=coverage,
        quarantine=entries,
        quarantined_symbol_count=len(get_quarantined_symbols()),
        flagged_symbol_count=len(flagged),
        total_symbols_loaded=prices.shape[1] if prices is not None else 0,
        sp500_csv=sp500,
        survivorship_note=(
            "FMP has no EOD history for many pre-~2015 delisted S&P members "
            "(~67% coverage in 2005 → ~98% in 2025). Missing names are dropped "
            "from the cross-section. Prefer 2015+ windows; earlier results are "
            "survivor-tilted. Closing the gap needs Norgate/CRSP/Tiingo. "
            "S&P membership: trust the Updated CSV (docs/SP500_MEMBERSHIP.md); "
            "FMP is cross-check only (notation-normalized Jaccard). "
            "Macro uses fixed pub lags, not ALFRED vintages "
            "(docs/MACRO_VINTAGES.md)."
            + age_note
        ),
    )
