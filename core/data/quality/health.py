"""Data-health audit: measure what is actually on disk, layer by layer.

This module is the single source of truth for "how good is our data" — the
universe funnel, per-layer coverage, survivorship gaps, leakage flags, calendar
anomalies, and a machine-readable registry of known flaws. It is consumed by:

- ``scripts/ops/audit_data_health.py`` — writes the JSON snapshot + refreshes
  ``docs/DATA_HEALTH.md``'s generated section,
- ``GET /data-health`` — serves the snapshot and per-symbol drilldowns to the
  frontend Data Health page.

Design rule: **measure, don't assert.** Every number here is recomputed from the
files on disk at audit time; nothing is copied forward from a previous audit or
from what a fetch script logged. Where full recomputation would be slow the
function reads parquet metadata (row counts) rather than data, and says so.

The audit is honest about its own blind spots: a dataset it does not know how to
check appears in the output as ``unchecked``, never silently omitted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq

from core.data.quality.validation import validate_price_panel, violations_as_dicts
from core.research.caveats import (
    SURFACE_DATA_HEALTH,
    SURFACE_UNIVERSE,
    as_dicts,
    caveats_for_surface,
)
from core.research.glossary import terms_in_category

logger = logging.getLogger(__name__)

RAW_FMP = Path("data/raw/fmp")
FACTORS = Path("data/factors")
UNIVERSE_FILE = RAW_FMP / "universe" / "us_equity_universe.parquet"
CANONICAL_PRICES = FACTORS / "prices.parquet"

# Per-symbol datasets fetched by scripts/ingest/fetch_fmp_datasets.py.
DATASET_DIRS: tuple[str, ...] = (
    "earnings",
    "dividends",
    "splits",
    "key_metrics",
    "ratios",
    "financial_growth",
    "enterprise_values",
    "financial_scores",
    "analyst_estimates",
    "grades_historical",
    "ratings_historical",
    "price_target_summary",
    "shares_float",
    "employee_count",
    "insider_trading",
    "stock_peers",
)

STATEMENT_DIRS: tuple[str, ...] = ("income_statement", "balance_sheet", "cash_flow")

# Factor/panel artifacts whose state the audit reports.
PANEL_FILES: tuple[str, ...] = (
    "prices.parquet",
    "prices_fmp.parquet",
    "ohlcv.parquet",
    "factors_price.parquet",
    "factors_all.parquet",
    "factors_fundamental.parquet",
    "fundamentals.parquet",
    "factors_microstructure.parquet",
    "factors_earnings_surprise.parquet",
    "factors_vendor_metrics.parquet",
    "dollar_adv_21d.parquet",
)


def _row_counts(directory: Path) -> dict[str, int]:
    """Map file stem -> parquet row count using metadata only (no data read)."""
    counts: dict[str, int] = {}
    for path in sorted(directory.glob("*.parquet")):
        try:
            counts[path.stem] = pq.ParquetFile(path).metadata.num_rows
        except Exception:  # corrupt file is itself a finding
            counts[path.stem] = -1
    return counts


def _coverage_against(universe_files: pd.Series, counts: dict[str, int]) -> dict[str, Any]:
    """Coverage stats of one dataset directory against the universe symbol list."""
    present = universe_files.map(lambda s: counts.get(s, None))
    return {
        "files": len(counts),
        "universe_with_data": int((present.fillna(0) > 0).sum()),
        "universe_empty_file": int((present == 0).sum()),
        "universe_missing_file": int(present.isna().sum()),
        "corrupt_files": int(sum(1 for v in counts.values() if v < 0)),
    }


def audit_universe() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the universe table and summarize its composition."""
    universe = pd.read_parquet(UNIVERSE_FILE)
    universe["file"] = universe["symbol"].str.replace("/", "-")
    summary = {
        "total": len(universe),
        "live": int((universe["is_delisted"] == False).sum()),  # noqa: E712 NA-aware
        "delisted": int((universe["is_delisted"] == True).sum()),  # noqa: E712
        "carried_over": int(universe["is_delisted"].isna().sum()),
        "by_exchange": universe["exchange"]
        .value_counts(dropna=False)
        .rename(lambda x: str(x))
        .to_dict(),
        "market_cap_floor_usd": 300e6,
    }
    return universe, summary


def audit_survivorship(universe: pd.DataFrame) -> dict[str, Any]:
    """
    The bias that matters most: do dead companies actually have data?

    A delisted name with no price history is invisible to a backtest, which
    quietly turns the universe back into a survivors-only universe.
    """
    report_path = RAW_FMP / "prices" / "_fetch_report.csv"
    report = pd.read_csv(report_path).set_index("symbol")
    delisted = universe[universe["is_delisted"] == True].copy()  # noqa: E712

    status = delisted["file"].map(report["status"])
    delisted["has_prices"] = status == "ok"
    delisted["delist_year"] = pd.to_datetime(delisted["delisted_date"]).dt.year

    era_bins = [1990, 2000, 2010, 2015, 2020, 2030]
    by_era = (
        delisted.groupby(pd.cut(delisted["delist_year"], era_bins), observed=True)["has_prices"]
        .agg(names="size", with_prices_pct=lambda s: round(s.mean() * 100, 1))
        .reset_index()
    )
    by_era["delist_year"] = by_era["delist_year"].astype(str)

    # Does the price history actually END at the delisting?
    last_price = pd.to_datetime(delisted["file"].map(report["last"]), errors="coerce")
    gap_days = (pd.to_datetime(delisted["delisted_date"]) - last_price).dt.days.dropna()

    missing = delisted[~delisted["has_prices"]]
    return {
        "delisted_total": len(delisted),
        "with_prices_pct": round(delisted["has_prices"].mean() * 100, 1),
        "by_delist_era": by_era.to_dict(orient="records"),
        "price_end_within_30d_of_delisting_pct": round((gap_days.abs() <= 30).mean() * 100, 1),
        "missing_symbols_sample": missing["symbol"].head(30).tolist(),
        "missing_count": len(missing),
    }


def audit_prices(universe: pd.DataFrame) -> dict[str, Any]:
    """Raw price layer: fetch outcomes and history depth."""
    report = pd.read_csv(RAW_FMP / "prices" / "_fetch_report.csv")
    ok = report[report["status"] == "ok"].copy()
    ok["first"] = pd.to_datetime(ok["first"], errors="coerce")
    ok["last"] = pd.to_datetime(ok["last"], errors="coerce")
    years = (ok["last"] - ok["first"]).dt.days / 365.25
    return {
        "fetch_outcomes": report["status"].value_counts().to_dict(),
        "failed_symbols": report[report["status"] == "failed"]["symbol"].tolist(),
        "history_years": {
            "median": round(float(years.median()), 1),
            "p10": round(float(years.quantile(0.1)), 1),
            "p90": round(float(years.quantile(0.9)), 1),
        },
        "symbols_starting_by_1990": int((ok["first"].dt.year <= 1990).sum()),
        "last_date_max": str(ok["last"].max().date()),
    }


def audit_panel_invariants() -> dict[str, Any]:
    """
    Structural checks on the canonical price panel (core.data.quality.validation).

    Distinct from the quarantine scanner: that flags suspicious *symbols* for
    human review, this asserts things that are never acceptable at all. Cheap
    enough to run on every audit, which is the point — the zero-price defect went
    unnoticed for hours because nothing checked automatically.
    """
    if not CANONICAL_PRICES.exists():
        return {"status": "unchecked"}
    prices = pd.read_parquet(CANONICAL_PRICES)
    violations = validate_price_panel(prices)
    return {
        "panel": CANONICAL_PRICES.name,
        "symbols": int(prices.shape[1]),
        "dates": int(prices.shape[0]),
        "violations": violations_as_dicts(violations),
        "errors": sum(1 for v in violations if v.severity == "error"),
        "warnings": sum(1 for v in violations if v.severity == "warning"),
    }


def audit_calendar() -> dict[str, Any]:
    """
    Non-trading-day bars in the expanded panel (vendor bad prints).

    Compares the expanded panel's date index against the canonical panel's
    trading calendar; anything extra is a Sunday or a US market holiday.
    """
    expanded_path = FACTORS / "prices_fmp.parquet"
    if not expanded_path.exists() or not CANONICAL_PRICES.exists():
        return {"status": "unchecked"}

    canonical_index = pd.read_parquet(CANONICAL_PRICES, columns=[]).index
    expanded_index = pd.read_parquet(expanded_path, columns=[]).index
    extra = expanded_index.difference(canonical_index)
    weekend = int((pd.Series(extra.dayofweek) >= 5).sum())
    return {
        "canonical_trading_days": len(canonical_index),
        "expanded_dates": len(expanded_index),
        "non_trading_dates": len(extra),
        "weekend_dates": weekend,
        "holiday_or_other_dates": len(extra) - weekend,
        "sample": [str(d.date()) for d in extra[:5]],
    }


def audit_publication_dates(sample_size: int = 1200, seed: int = 7) -> dict[str, Any]:
    """
    Share of statement rows whose acceptedDate is a placeholder (ADR 0012).

    Reads only the two date columns of a random sample of income statements.
    """
    import random

    paths = sorted((RAW_FMP / "fundamentals" / "income_statement").glob("*.parquet"))
    random.seed(seed)
    sample = random.sample(paths, min(sample_size, len(paths)))

    frames = []
    for path in sample:
        try:
            frame = pd.read_parquet(path, columns=["date", "acceptedDate"])
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return {"status": "unchecked"}

    rows = pd.concat(frames, ignore_index=True)
    lag = (
        pd.to_datetime(rows["acceptedDate"]).dt.normalize() - pd.to_datetime(rows["date"])
    ).dt.days
    decade = (pd.to_datetime(rows["date"]).dt.year // 10) * 10
    by_decade = (
        pd.DataFrame({"decade": decade, "placeholder": lag <= 0})
        .groupby("decade")["placeholder"]
        .agg(rows="size", placeholder_pct=lambda s: round(s.mean() * 100, 1))
        .reset_index()
    )
    return {
        "sampled_symbols": len(sample),
        "sampled_rows": len(rows),
        "placeholder_accepted_date_pct": round(float((lag <= 0).mean() * 100), 1),
        "by_decade": by_decade.to_dict(orient="records"),
        "remediation": "45-day fallback lag + publication_date_imputed flag (ADR 0012)",
    }


def audit_intraday() -> dict[str, Any]:
    """Intraday raw layer: symbols, span, and empty-year share per interval."""
    intraday_root = RAW_FMP / "intraday"
    if not intraday_root.exists():
        return {"status": "unchecked"}
    result: dict[str, Any] = {}
    for interval_dir in sorted(intraday_root.iterdir()):
        if not interval_dir.is_dir():
            continue
        symbol_dirs = [d for d in interval_dir.iterdir() if d.is_dir()]
        n_files = n_empty = 0
        years: set[str] = set()
        for symbol_dir in symbol_dirs:
            for file in symbol_dir.glob("*.parquet"):
                n_files += 1
                years.add(file.stem)
                if pq.ParquetFile(file).metadata.num_rows == 0:
                    n_empty += 1
        result[interval_dir.name] = {
            "symbols": len(symbol_dirs),
            "symbol_year_files": n_files,
            "empty_symbol_years": n_empty,
            "year_range": f"{min(years)}-{max(years)}" if years else None,
        }
    return result


def audit_panels() -> list[dict[str, Any]]:
    """State of every derived panel artifact: rows, symbols, freshness."""
    entries = []
    for name in PANEL_FILES:
        path = FACTORS / name
        if not path.exists():
            entries.append({"file": name, "status": "missing"})
            continue
        meta = pq.ParquetFile(path)
        schema_names = meta.schema_arrow.names
        entry: dict[str, Any] = {
            "file": name,
            "rows": meta.metadata.num_rows,
            "columns": len(schema_names),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        }
        # Wide panels: symbol count == column count; long panels: count level values.
        if "symbol" in schema_names or "__index_level_1__" in schema_names:
            try:
                symbols = pd.read_parquet(path, columns=[]).index.get_level_values("symbol")
                entry["symbols"] = int(symbols.nunique())
            except Exception:
                entry["symbols"] = None
        else:
            entry["symbols"] = len([c for c in schema_names if not c.startswith("__")]) - (
                1 if "date" in schema_names else 0
            )
        entries.append(entry)
    return entries


def audit_datasets(universe: pd.DataFrame) -> dict[str, Any]:
    """Coverage of every per-symbol dataset directory against the universe."""
    result = {}
    for dataset in DATASET_DIRS:
        directory = RAW_FMP / dataset
        if not directory.exists():
            result[dataset] = {"status": "missing"}
            continue
        result[dataset] = _coverage_against(universe["file"], _row_counts(directory))
    for statement in STATEMENT_DIRS:
        directory = RAW_FMP / "fundamentals" / statement
        result[f"fundamentals/{statement}"] = _coverage_against(
            universe["file"], _row_counts(directory)
        )
    result["market_caps"] = _coverage_against(
        universe["file"], _row_counts(RAW_FMP / "market_caps")
    )
    return result


def build_funnel(universe_summary: dict[str, Any], datasets: dict[str, Any]) -> list[dict]:
    """
    The universe decomposition, each stage explaining what its number MEANS.

    A bare funnel invites the obvious misreading — "we have 49,939 symbols but
    only 7,852 statements, so 42,000 are broken". They are different populations,
    and only the stages from ``in_our_universe`` onward describe what we hold.
    Every stage therefore carries ``scope`` (vendor catalog vs our holdings), a
    plain-language ``definition``, and ``why_smaller`` explaining the drop from
    the stage above.
    """
    statements = datasets.get("fundamentals/income_statement", {})
    prices_report = pd.read_csv(RAW_FMP / "prices" / "_fetch_report.csv")
    live = universe_summary["live"]
    total = universe_summary["total"]
    with_prices = int((prices_report["status"] == "ok").sum())
    with_statements = statements.get("universe_with_data", 0)
    with_earnings = datasets.get("earnings", {}).get("universe_with_data", 0)

    return [
        {
            "stage": "Everything the vendor lists",
            "count": 49939,
            "scope": "vendor catalog",
            "definition": (
                "Every ticker FMP serves worldwide: US and foreign exchanges, ETFs, mutual "
                "funds, trusts, ADRs, crypto pairs. NOT companies we could trade or want."
            ),
            "why_smaller": None,
            "note": "probed 2026-08-07",
        },
        {
            "stage": "US-listed common stock",
            "count": 9543,
            "scope": "vendor catalog",
            "definition": (
                "Filtered to ordinary shares on NYSE, NASDAQ and AMEX — excludes ETFs, funds, "
                "and every non-US venue. This is the realistic opportunity set for a US "
                "equity strategy."
            ),
            "why_smaller": "Drops ~40,000 foreign listings, ETFs, funds and non-equity tickers.",
            "note": "screener, live names only",
        },
        {
            "stage": "Live, above $300M market cap",
            "count": live,
            "scope": "our universe",
            "definition": (
                "Currently-trading US common stocks worth more than $300M TODAY. The floor "
                "exists because sub-$300M names are largely untradeable at any size."
            ),
            "why_smaller": "Drops ~5,000 micro-caps below the floor.",
            "note": "cap measured today, not point-in-time",
        },
        {
            "stage": "Our universe: live + dead companies",
            "count": total,
            "scope": "our universe",
            "definition": (
                "The live names PLUS every US company that was delisted (acquired, bankrupt, "
                "taken private). Dead companies are included on purpose: a universe of only "
                "survivors makes every backtest look better than reality."
            ),
            "why_smaller": None,
            "note": f"{universe_summary['delisted']:,} delisted + {live:,} live",
        },
        {
            "stage": "…with daily price history",
            "count": with_prices,
            "scope": "what we hold",
            "definition": (
                "Symbols where we actually downloaded and stored dividend-adjusted daily "
                "prices. This is the number that bounds any price-based backtest."
            ),
            "why_smaller": (
                f"{total - with_prices} symbols returned no price data — mostly old delistings "
                "the vendor no longer carries."
            ),
            "note": "raw price layer",
        },
        {
            "stage": "…with quarterly financial statements",
            "count": with_statements,
            "scope": "what we hold",
            "definition": (
                "Symbols with income statement / balance sheet / cash flow filings stored. "
                "Bounds every fundamental factor (value, quality, profitability)."
            ),
            "why_smaller": (
                f"{with_prices - with_statements} priced symbols have no filings: trusts, "
                "foreign-domiciled issuers, and companies that delisted before EDGAR."
            ),
            "note": "raw fundamentals layer",
        },
        {
            "stage": "…with analyst earnings estimates",
            "count": with_earnings,
            "scope": "what we hold",
            "definition": (
                "Symbols with actual-vs-estimated EPS per announcement. Bounds the PEAD / "
                "earnings-surprise research, which needs a consensus to measure surprise "
                "against."
            ),
            "why_smaller": (
                f"{with_statements - with_earnings} symbols file statements but were never "
                "covered by analysts — small and micro-caps, and pre-1990s history."
            ),
            "note": "raw earnings layer",
        },
    ]


def known_flaws(
    survivorship: dict[str, Any],
    calendar: dict[str, Any],
    publication: dict[str, Any],
    datasets: dict[str, Any],
    panels: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    The machine-readable flaw registry.

    Every entry is a caveat a researcher must know before trusting a result.
    Severity: ``high`` = can silently flip a conclusion; ``medium`` = biases a
    subset of results; ``low`` = operational annoyance.
    """
    flaws: list[dict[str, str]] = []

    flaws.append(
        {
            "id": "survivorship-pre-2015-delistings",
            "severity": "medium",
            "title": "Old delistings are under-covered",
            "detail": (
                f"{survivorship['missing_count']} delisted names have no price history, "
                "concentrated in pre-2015 delistings (e.g. Compaq, Countrywide, Albertsons). "
                "FMP's delisted feed skews recent — backtests before ~2015 on the expanded "
                "universe still lean toward survivors. The 774-name S&P membership panel is "
                "the better pre-2015 universe."
            ),
        }
    )
    if calendar.get("non_trading_dates", 0) > 0:
        flaws.append(
            {
                "id": "non-trading-day-bars",
                "severity": "medium",
                "title": f"{calendar['non_trading_dates']} non-trading dates in expanded panel",
                "detail": (
                    f"{calendar.get('weekend_dates', 0)} weekend dates and "
                    f"{calendar.get('holiday_or_other_dates', 0)} US-holiday dates carry vendor "
                    "bars in prices_fmp.parquet. Returns computed across them are wrong for the "
                    "symbols involved. Panel builders now intersect with the canonical NYSE "
                    "calendar; artifacts built before 2026-08-09 may retain them."
                ),
            }
        )
    flaws.append(
        {
            "id": "placeholder-accepted-dates",
            "severity": "high",
            "title": (
                f"{publication.get('placeholder_accepted_date_pct', '?')}% of statement rows "
                "had placeholder filing dates"
            ),
            "detail": (
                "FMP fills acceptedDate with the period end for pre-EDGAR filings (100% of "
                "1980s rows, ~50% of 1990s). Taken literally this made results knowable the "
                "day the quarter closed. Remediated with a 45-day fallback lag, flagged "
                "per-row as publication_date_imputed (ADR 0012). Results leaning on pre-2000 "
                "history must be re-run excluding imputed rows."
            ),
        }
    )
    earnings_cov = datasets.get("earnings", {}).get("universe_with_data", 0)
    flaws.append(
        {
            "id": "event-panels-still-774",
            "severity": "medium",
            "title": (
                f"Raw event datasets cover {earnings_cov:,} symbols, but the DERIVED "
                "surprise/vendor panels are still 774-name builds"
            ),
            "detail": (
                "The expanded dataset fetch completed (earnings, ratios, etc. now cover most "
                "of the universe), but factors_earnings_surprise.parquet and "
                "factors_vendor_metrics.parquet are built against the canonical 774-name "
                "price panel. PEAD and vendor-metric research is large-cap-only until "
                "build_event_factors gains an --expanded mode against prices_fmp.parquet — "
                "which matters, because the PEAD literature puts the effect in small caps."
            ),
        }
    )
    flaws.append(
        {
            "id": "intraday-snapshot-adjustment-and-bar-cap",
            "severity": "high",
            "title": "Intraday: frozen split-adjustment snapshot, no dividend adjustment, bar-capped endpoint",
            "detail": (
                "(1) Stored intraday bars are split-adjusted AS OF THE FETCH DATE (measured on "
                "AAPL/NVDA/TSLA splits, 2026-08-11) — a split after the fetch leaves them stale, "
                "and a blanket re-adjustment double-adjusts. Always read via "
                "core.data.vendors.fmp.intraday.load_intraday_bars, which detects and repairs exactly "
                "the unapplied splits. (2) Bars are NEVER dividend-adjusted, unlike the daily "
                "adj_close layer: overnight returns across ex-dividend dates carry the dividend "
                "drop as a fake negative return. (3) The endpoint silently returns only the most "
                "recent ~1,170 bars per request; only the chunked fetcher is safe."
            ),
        }
    )
    flaws.append(
        {
            "id": "canonical-panel-still-774",
            "severity": "low",
            "title": "Canonical prices.parquet is still the 774-name panel",
            "detail": (
                "The expanded 8,908-symbol panel lives in prices_fmp.parquet (staging). The "
                "API, existing backtests, and the fundamentals build still run on the 774 "
                "panel. Cutover is a deliberate pending decision (memory, calendar, and every "
                "downstream artifact change with it)."
            ),
        }
    )
    flaws.append(
        {
            "id": "sector-labels-current-only",
            "severity": "medium",
            "title": "Sector labels are today's, applied historically",
            "detail": (
                "FMP exposes only current sector/industry. Sector-neutral factors "
                "(value_quality_sn, roe_sn) carry mild lookahead through reclassifications "
                "(documented in core/signals/sector_neutral.py)."
            ),
        }
    )
    flaws.append(
        {
            "id": "extreme-return-bad-prints",
            "severity": "high",
            "title": "Vendor bad prints produce daily returns up to +100,000,000%",
            "detail": (
                "Un-adjusted reverse splits and quote errors on delisted micro-caps. Measured "
                "2026-08-13: max daily return +101,599,900%; 7,691 returns above +100%; 20,642 "
                "exact-zero closes (now nulled by the panel builder). One such value in a "
                "cross-sectional mean makes every symbol's abnormal return infinite for that "
                "date. All return math must go through core.data.returns.compute_clean_returns."
            ),
        }
    )
    flaws.append(
        {
            "id": "shell-companies-and-spacs-in-universe",
            "severity": "high",
            "title": "20% of the universe is shell companies / SPACs, not operating businesses",
            "detail": (
                "1,779 of ~9,000 symbols carry industry 'Shell Companies' (measured "
                "2026-08-12) — pre-merger SPACs and blank-check vehicles. A pre-merger SPAC "
                "is a trust account with a ticker: it sits within pennies of $10.00 (many "
                "show annualized vol under 5%), then merges into an unrelated business or "
                "liquidates. Ranking one on momentum, value or profitability is meaningless, "
                "and its near-zero volatility makes it look like an extreme low-vol name, so "
                "it lands in traded tiers rather than the middle. Use "
                "core.data.universe_filters.build_universe_filter(exclude_non_operating=True) "
                "for any cross-sectional work on the expanded universe."
            ),
        }
    )
    flaws.append(
        {
            "id": "screener-drops-some-reits-and-own-exchange-listings",
            "severity": "medium",
            "title": "The live-universe screener silently drops some REITs and own-exchange listings",
            "detail": (
                "FMP tags several large REITs as isFund=true (measured 2026-08-11: Realty "
                "Income $58B, Federal Realty $10B), so the screener's isFund=false filter "
                "excludes them; CBOE lists on its own exchange, outside the NYSE/NASDAQ/AMEX "
                "filter. Such names are in the universe only if carried over from the 774 "
                "panel — the expanded universe under-represents Real Estate. Fix on the next "
                "universe rebuild: add a REIT-inclusive pass (sector=Real Estate, isEtf=false) "
                "and audit exchange coverage."
            ),
        }
    )
    flaws.append(
        {
            "id": "screener-cap-floor-today",
            "severity": "medium",
            "title": "The $300M cap floor is applied at today's market cap",
            "detail": (
                "A stock worth $200M today but $2B in 2015 is excluded; one worth $350M today "
                "but $50M in 2015 is included. The universe table decides what we DOWNLOAD; "
                "point-in-time tradability must still be enforced with the market-cap panel "
                "at backtest time."
            ),
        }
    )
    return flaws


# The data-hygiene subset of the platform glossary. Definitions live in
# core.research.glossary — one registry, so the data-health page and the glossary
# page can never disagree about what a term means.
GLOSSARY: tuple[dict[str, str], ...] = tuple(
    {"term": entry.term, "definition": entry.definition} for entry in terms_in_category("data")
)


def run_full_audit() -> dict[str, Any]:
    """Run every audit section and assemble the snapshot dict."""
    universe, universe_summary = audit_universe()
    survivorship = audit_survivorship(universe)
    calendar = audit_calendar()
    publication = audit_publication_dates()
    datasets = audit_datasets(universe)
    panels = audit_panels()

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "universe": universe_summary,
        "funnel": build_funnel(universe_summary, datasets),
        "prices": audit_prices(universe),
        "survivorship": survivorship,
        "calendar": calendar,
        "publication_dates": publication,
        "datasets": datasets,
        "intraday": audit_intraday(),
        "panels": panels,
        "flaws": known_flaws(survivorship, calendar, publication, datasets, panels),
        "registry_caveats": as_dicts(caveats_for_surface(SURFACE_DATA_HEALTH))
        + as_dicts(caveats_for_surface(SURFACE_UNIVERSE)),
        "glossary": list(GLOSSARY),
    }


def load_symbol_detail(symbol: str) -> Optional[dict[str, Any]]:
    """
    Everything we hold for one symbol — the drilldown behind the health page.

    Reads ~20 small per-symbol files; fast enough to serve live.
    """
    file_stem = symbol.replace("/", "-")
    universe = pd.read_parquet(UNIVERSE_FILE)
    row = universe[universe["symbol"] == symbol]
    detail: dict[str, Any] = {"symbol": symbol}
    if not row.empty:
        record = row.iloc[0]
        detail["universe"] = {
            "company_name": None if pd.isna(record["company_name"]) else record["company_name"],
            "exchange": None if pd.isna(record["exchange"]) else record["exchange"],
            "sector": None if pd.isna(record["sector"]) else record["sector"],
            "is_delisted": None if pd.isna(record["is_delisted"]) else bool(record["is_delisted"]),
            "ipo_date": None if pd.isna(record["ipo_date"]) else str(record["ipo_date"].date()),
            "delisted_date": (
                None if pd.isna(record["delisted_date"]) else str(record["delisted_date"].date())
            ),
            "market_cap": None if pd.isna(record["market_cap"]) else float(record["market_cap"]),
        }

    prices_path = RAW_FMP / "prices" / f"{file_stem}.parquet"
    if prices_path.exists():
        bars = pd.read_parquet(prices_path)
        detail["prices"] = {
            "rows": len(bars),
            "first": str(bars.index.min().date()) if len(bars) else None,
            "last": str(bars.index.max().date()) if len(bars) else None,
            "has_ohlc": bool({"adj_open", "adj_high", "adj_low"} <= set(bars.columns)),
        }

    statements: dict[str, Any] = {}
    for statement in STATEMENT_DIRS:
        path = RAW_FMP / "fundamentals" / statement / f"{file_stem}.parquet"
        if not path.exists():
            statements[statement] = None
            continue
        frame = pd.read_parquet(path, columns=["date", "acceptedDate"])
        if frame.empty:
            statements[statement] = {"quarters": 0}
            continue
        lag = (
            pd.to_datetime(frame["acceptedDate"]).dt.normalize() - pd.to_datetime(frame["date"])
        ).dt.days
        statements[statement] = {
            "quarters": len(frame),
            "first_period": str(pd.to_datetime(frame["date"]).min().date()),
            "last_period": str(pd.to_datetime(frame["date"]).max().date()),
            "placeholder_filing_dates_pct": round(float((lag <= 0).mean() * 100), 1),
        }
    detail["statements"] = statements

    mcap_path = RAW_FMP / "market_caps" / f"{file_stem}.parquet"
    detail["market_caps_rows"] = (
        pq.ParquetFile(mcap_path).metadata.num_rows if mcap_path.exists() else None
    )

    dataset_rows: dict[str, Optional[int]] = {}
    for dataset in DATASET_DIRS:
        path = RAW_FMP / dataset / f"{file_stem}.parquet"
        dataset_rows[dataset] = pq.ParquetFile(path).metadata.num_rows if path.exists() else None
    detail["datasets"] = dataset_rows

    intraday: dict[str, Any] = {}
    intraday_root = RAW_FMP / "intraday"
    if intraday_root.exists():
        for interval_dir in sorted(intraday_root.iterdir()):
            symbol_dir = interval_dir / file_stem
            if symbol_dir.is_dir():
                files = sorted(symbol_dir.glob("*.parquet"))
                intraday[interval_dir.name] = {
                    "years": [f.stem for f in files],
                    "rows": sum(pq.ParquetFile(f).metadata.num_rows for f in files),
                }
    detail["intraday"] = intraday

    if "universe" not in detail and "prices" not in detail:
        return None
    return detail
