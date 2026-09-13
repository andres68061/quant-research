#!/usr/bin/env python3
"""
Probe which FMP endpoints the configured API key is entitled to.

The vendor's plan/tier table does not reliably describe what a key returns, so
entitlements are established empirically: one representative request per endpoint
family, recording the HTTP status. HTTP 402 means "not on your plan".

Every probe uses the smallest possible response (``limit=1``/short date ranges) so
a full sweep costs ~50 calls.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/probe_fmp_entitlements.py
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/probe_fmp_entitlements.py --restricted-only
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/probe_fmp_entitlements.py --csv out.csv

Paste the restricted list into docs/data/DATA_INVENTORY.md §6 after running.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.vendors.fmp.client import FMP_BASE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("probe_fmp_entitlements")

_RECENT = pd.Timestamp.now().normalize()
_A_WEEK_AGO = (_RECENT - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
_TODAY = _RECENT.strftime("%Y-%m-%d")
_LAST_QUARTER_END = (_RECENT - pd.offsets.QuarterEnd(2)).strftime("%Y-%m-%d")

# (family, endpoint path, params). One representative endpoint per family.
PROBES: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("statements", "income-statement", {"symbol": "AAPL", "limit": 1}),
    ("statements", "balance-sheet-statement", {"symbol": "AAPL", "limit": 1}),
    ("statements", "cash-flow-statement", {"symbol": "AAPL", "limit": 1}),
    ("statements", "income-statement-as-reported", {"symbol": "AAPL", "limit": 1}),
    ("metrics", "key-metrics", {"symbol": "AAPL", "limit": 1}),
    ("metrics", "ratios", {"symbol": "AAPL", "limit": 1}),
    ("metrics", "key-metrics-ttm", {"symbol": "AAPL"}),
    ("metrics", "ratios-ttm", {"symbol": "AAPL"}),
    ("metrics", "financial-growth", {"symbol": "AAPL", "limit": 1}),
    ("metrics", "enterprise-values", {"symbol": "AAPL", "limit": 1}),
    ("metrics", "financial-scores", {"symbol": "AAPL"}),
    ("earnings", "earnings", {"symbol": "AAPL", "limit": 1}),
    ("earnings", "earnings-calendar", {"from": _A_WEEK_AGO, "to": _TODAY}),
    ("corporate-actions", "dividends", {"symbol": "AAPL", "limit": 1}),
    ("corporate-actions", "splits", {"symbol": "AAPL", "limit": 1}),
    ("analyst", "analyst-estimates", {"symbol": "AAPL", "period": "annual", "limit": 1}),
    ("analyst", "price-target-summary", {"symbol": "AAPL"}),
    ("analyst", "price-target-consensus", {"symbol": "AAPL"}),
    ("analyst", "grades-consensus", {"symbol": "AAPL"}),
    ("analyst", "grades-historical", {"symbol": "AAPL", "limit": 1}),
    ("analyst", "ratings-snapshot", {"symbol": "AAPL"}),
    ("analyst", "ratings-historical", {"symbol": "AAPL", "limit": 1}),
    ("company", "profile", {"symbol": "AAPL"}),
    ("company", "shares-float", {"symbol": "AAPL"}),
    ("company", "employee-count", {"symbol": "AAPL", "limit": 1}),
    ("company", "market-capitalization", {"symbol": "AAPL"}),
    ("company", "historical-market-capitalization", {"symbol": "AAPL", "limit": 1}),
    ("company", "stock-peers", {"symbol": "AAPL"}),
    ("company", "delisted-companies", {"page": 0, "limit": 1}),
    ("company", "all-industry-classification", {}),
    ("prices", "historical-price-eod/full", {"symbol": "AAPL", "from": _A_WEEK_AGO, "to": _TODAY}),
    (
        "prices",
        "historical-price-eod/dividend-adjusted",
        {"symbol": "AAPL", "from": _A_WEEK_AGO, "to": _TODAY},
    ),
    ("intraday", "historical-chart/1hour", {"symbol": "AAPL", "from": _TODAY, "to": _TODAY}),
    ("intraday", "historical-chart/5min", {"symbol": "AAPL", "from": _TODAY, "to": _TODAY}),
    ("intraday", "historical-chart/1min", {"symbol": "AAPL", "from": _TODAY, "to": _TODAY}),
    ("indexes", "sp500-constituent", {}),
    ("indexes", "historical-sp500-constituent", {}),
    ("indexes", "nasdaq-constituent", {}),
    ("indexes", "dowjones-constituent", {}),
    ("directory", "stock-list", {}),
    ("directory", "financial-statement-symbol-list", {}),
    ("directory", "available-exchanges", {}),
    ("directory", "company-screener", {"marketCapMoreThan": 1_000_000_000, "limit": 1}),
    ("economics", "treasury-rates", {"from": _A_WEEK_AGO, "to": _TODAY}),
    ("economics", "economic-indicators", {"name": "GDP", "from": _A_WEEK_AGO, "to": _TODAY}),
    ("economics", "economic-calendar", {"from": _A_WEEK_AGO, "to": _TODAY}),
    ("ownership", "insider-trading/search", {"symbol": "AAPL", "limit": 1}),
    (
        "ownership",
        "institutional-ownership/symbol-positions-summary",
        {"symbol": "AAPL", "year": _RECENT.year - 1, "quarter": 4},
    ),
    ("etf", "etf/holdings", {"symbol": "SPY"}),
    ("etf", "etf/sector-weightings", {"symbol": "SPY"}),
    ("alternative", "senate-trades", {"symbol": "AAPL"}),
    ("alternative", "commitment-of-traders-report", {"symbol": "KC"}),
    ("alternative", "esg-disclosures", {"symbol": "AAPL"}),
    ("alternative", "discounted-cash-flow", {"symbol": "AAPL"}),
    ("news", "news/stock", {"symbols": "AAPL", "limit": 1}),
    ("news", "earning-call-transcript", {"symbol": "AAPL", "year": _RECENT.year - 1, "quarter": 1}),
    (
        "technicals",
        "technical-indicators/rsi",
        {
            "symbol": "AAPL",
            "periodLength": 14,
            "timeframe": "1day",
            "from": _A_WEEK_AGO,
            "to": _TODAY,
        },
    ),
    ("bulk", "eod-bulk", {"date": _A_WEEK_AGO}),
    ("bulk", "profile-bulk", {"part": "0"}),
    ("bulk", "ratios-ttm-bulk", {}),
    ("bulk", "scores-bulk", {}),
)

_THROTTLE_SECONDS = 0.15
_TIMEOUT_SECONDS = 45


def probe_all(api_key: str) -> pd.DataFrame:
    """
    Request every endpoint in :data:`PROBES` once and record the outcome.

    Args:
        api_key: FMP API key.

    Returns:
        DataFrame with columns ``family``, ``endpoint``, ``status_code``,
        ``entitled`` (bool), ``rows`` (response length when JSON is a list),
        ``detail`` (truncated error body for non-200s).
    """
    records: list[dict[str, object]] = []
    for family, endpoint, params in PROBES:
        query = dict(params)
        query["apikey"] = api_key
        status: object
        rows: object = None
        detail = ""
        try:
            response = requests.get(
                f"{FMP_BASE_URL}/{endpoint}", params=query, timeout=_TIMEOUT_SECONDS
            )
            status = response.status_code
            if response.status_code == 200:
                payload = response.json()
                rows = len(payload) if isinstance(payload, list) else 1
            else:
                # Never log response.url: it embeds the API key.
                detail = response.text[:120].replace("\n", " ")
        except requests.RequestException as exc:
            status = "ERROR"
            detail = str(exc)[:120]

        records.append(
            {
                "family": family,
                "endpoint": endpoint,
                "status_code": status,
                "entitled": status == 200,
                "rows": rows,
                "detail": detail,
            }
        )
        time.sleep(_THROTTLE_SECONDS)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe FMP endpoint entitlements")
    parser.add_argument("--restricted-only", action="store_true", help="Only print non-200s")
    parser.add_argument("--csv", type=str, default=None, help="Write the full result to this path")
    args = parser.parse_args()

    from config.settings import FMP_API_KEY

    if not FMP_API_KEY:
        raise SystemExit("FMP_API_KEY is not set; add it to .env")

    results = probe_all(FMP_API_KEY)
    restricted = results[~results["entitled"]]

    display = restricted if args.restricted_only else results
    for _, row in display.iterrows():
        logger.info(
            "%-16s %-52s %s %s",
            row["family"],
            row["endpoint"],
            row["status_code"],
            row["detail"][:70],
        )

    logger.info(
        "Entitled %d/%d. Restricted families: %s",
        int(results["entitled"].sum()),
        len(results),
        sorted(restricted["family"].unique().tolist()) or "none",
    )

    if args.csv:
        results.to_csv(args.csv, index=False)
        logger.info("Wrote %s", args.csv)


if __name__ == "__main__":
    main()
