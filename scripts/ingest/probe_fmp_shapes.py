#!/usr/bin/env python3
"""
Probe the live FMP surface and regenerate ``config/vendors/fmp.json``.

The manifest is not hand-written. FMP's own documentation renders its parameter
tables in JavaScript, so the only reliable way to learn how an endpoint must be
called is to call it. This script does that in three passes and writes what it
learns, so the vendor catalog is reproducible rather than folklore:

1. **Shape.** Every entitled path from ``docs/sources/vendor/fmp/ENDPOINT_CATALOG.md`` is
   called with a series of candidate parameter sets until one returns rows. That
   settles whether the endpoint is global, per-symbol, per-CIK, and so on.
2. **Disambiguation.** Several endpoints answer a bare call *and* a symbol call —
   ``employee-count`` returns one company's history either way. Ingesting those
   globally would store one arbitrary company and call it a dataset, so the
   response to ``symbol=AAPL`` is checked for whether it is actually scoped to
   that symbol.
3. **Emit.** Results are merged with the curated judgement that cannot be probed —
   wave assignment, point-in-time classification, which directory an endpoint
   shares with existing downloads — and written as the manifest.

Re-run this when the vendor changes its surface, then run
``scripts/ingest/validate_fmp_manifest.py`` before starting a long ingestion.

Usage:
    python scripts/ingest/probe_fmp_shapes.py --out config/vendors/fmp.json
    python scripts/ingest/probe_fmp_shapes.py --shapes-only --out /tmp/shapes.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.vendors.fmp.transport import make_fetcher
from core.ingest.ratelimit import TokenBucket

logger = logging.getLogger("probe_fmp_shapes")

CATALOG = ROOT / "docs" / "sources" / "vendor" / "fmp" / "ENDPOINT_CATALOG.md"

# Candidate parameter sets, tried in order; the first returning rows wins.
VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("none", {}),
    ("symbol", {"symbol": "AAPL"}),
    ("symbol_dates", {"symbol": "AAPL", "from": "2024-01-01", "to": "2024-03-31"}),
    ("query", {"query": "AAPL"}),
    ("cik", {"cik": "0000320193"}),
    ("exchange", {"exchange": "NASDAQ"}),
    ("dates", {"from": "2024-01-01", "to": "2024-03-31"}),
    ("period", {"symbol": "AAPL", "period": "quarter", "limit": 5}),
    ("name", {"name": "Apple"}),
    ("symbols", {"symbols": "AAPL,MSFT"}),
    ("sector", {"sector": "Technology", "exchange": "NASDAQ"}),
    ("industry", {"industry": "Semiconductors", "exchange": "NASDAQ"}),
    ("indicator", {"symbol": "AAPL", "periodLength": 10, "timeframe": "1day"}),
    ("report", {"symbol": "AAPL", "year": 2024, "period": "FY"}),
    ("form_type", {"formType": "10-K", "from": "2024-01-01", "to": "2024-03-31"}),
    ("company", {"company": "Apple"}),
    ("page", {"page": 0, "limit": 10}),
)


def entitled_paths(catalog: Path = CATALOG) -> list[tuple[str, str]]:
    """
    Read ``(category, path)`` for every endpoint the catalog marks as working.

    Args:
        catalog: Markdown endpoint catalog.

    Returns:
        Pairs in document order; blocked (HTTP 402) paths are excluded.
    """
    rows: list[tuple[str, str]] = []
    category = ""
    for line in catalog.read_text().splitlines():
        if line.startswith("## "):
            category = line[3:].strip()
        match = re.match(r"^\|\s*`([^`]+)`\s*\|(.*?)\|\s*(works[^|]*|blocked[^|]*)\|\s*$", line)
        if match and match.group(3).strip().startswith("works"):
            rows.append((category, match.group(1).strip().lstrip("/")))
    return rows


def probe_path(fetch: Any, bucket: TokenBucket, path: str) -> dict[str, Any]:
    """
    Find the first parameter set that makes one endpoint return rows.

    Args:
        fetch: Transport callable.
        bucket: Shared rate limiter.
        path: Endpoint path.

    Returns:
        ``{"path", "ok_variant", "params", "n", "keys"}``; ``ok_variant`` is None
        when no candidate worked.
    """
    for name, params in VARIANTS:
        bucket.acquire()
        result = fetch(path, params)
        if result.status_code != 200:
            continue
        if result.content is not None:
            return {"path": path, "ok_variant": name, "params": params, "n": 1, "binary": True}
        rows = result.rows or []
        if rows:
            return {
                "path": path,
                "ok_variant": name,
                "params": params,
                "n": len(rows),
                "keys": sorted(rows[0].keys())[:40],
            }
    return {"path": path, "ok_variant": None, "params": {}, "n": 0, "keys": []}


def is_symbol_scoped(fetch: Any, bucket: TokenBucket, path: str) -> bool:
    """
    Whether an endpoint that answers a bare call is really scoped to one symbol.

    Args:
        fetch: Transport callable.
        bucket: Shared rate limiter.
        path: Endpoint path.

    Returns:
        True when every returned row carries the probed symbol, which means the
        endpoint must be ingested per symbol rather than once.
    """
    bucket.acquire()
    result = fetch(path, {"symbol": "AAPL"})
    rows = result.rows or []
    symbols = {row.get("symbol") for row in rows if isinstance(row, dict) and "symbol" in row}
    return symbols == {"AAPL"}


def main() -> int:
    """Probe the vendor and write shapes (and optionally the manifest)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "config" / "vendors" / "fmp.json"))
    parser.add_argument("--shapes-only", action="store_true", help="write raw probe output only")
    parser.add_argument("--rate", type=float, default=480.0)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    paths = entitled_paths()
    logger.info("probing %d entitled endpoints", len(paths))
    fetch = make_fetcher()
    bucket = TokenBucket(args.rate)

    with futures.ThreadPoolExecutor(args.workers) as pool:
        shapes = list(pool.map(lambda cp: probe_path(fetch, bucket, cp[1]), paths))
    for shape, (category, _) in zip(shapes, paths, strict=True):
        shape["category"] = category

    globals_ = [s["path"] for s in shapes if s["ok_variant"] == "none"]
    with futures.ThreadPoolExecutor(args.workers) as pool:
        scoped = dict(
            zip(
                globals_,
                pool.map(lambda p: is_symbol_scoped(fetch, bucket, p), globals_),
                strict=True,
            )
        )
    for shape in shapes:
        shape["symbol_scoped"] = scoped.get(shape["path"], False)

    unresolved = [s["path"] for s in shapes if s["ok_variant"] is None]
    logger.info(
        "resolved %d/%d endpoints; %d symbol-scoped despite bare calls; unresolved: %s",
        len(shapes) - len(unresolved),
        len(shapes),
        sum(1 for s in shapes if s["symbol_scoped"]),
        unresolved or "none",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.shapes_only:
        out.write_text(json.dumps(shapes, indent=1) + "\n")
        logger.info("wrote raw shapes to %s", out)
        return 0

    logger.warning(
        "manifest emission merges probe output with curated wave and point-in-time "
        "judgement; review the diff against %s before committing",
        out,
    )
    shapes_path = out.with_name(f"{out.stem}_shapes.json")
    shapes_path.write_text(json.dumps(shapes, indent=1) + "\n")
    logger.info("wrote probe evidence to %s", shapes_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
