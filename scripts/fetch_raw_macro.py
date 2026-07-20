#!/usr/bin/env python3
"""Fetch the raw FRED macro panel and write it to ``data/raw/macro_fred.parquet``.

The raw layer keeps each FRED series at its native frequency, with no
publication lag, no business-day forward-fill, and no standardisation.
Downstream artefacts (``data/factors/macro.parquet``, ``macro_z.parquet``,
notebooks) derive from this file.

Usage:
    python scripts/fetch_raw_macro.py
    python scripts/fetch_raw_macro.py --out data/raw/macro_fred.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.factors.macro import (  # noqa: E402  -- after sys.path edit
    DEFAULT_FRED_SERIES_MAP,
    RAW_MACRO_PARQUET,
    load_raw_macro_default,
)
from core.utils.io import write_parquet  # noqa: E402

logger = logging.getLogger("fetch_raw_macro")


def fetch_and_write(out_path: Path) -> Path:
    """Fetch all canonical FRED series and persist to ``out_path``."""
    raw_long = load_raw_macro_default()
    if raw_long.empty:
        raise RuntimeError(
            "Raw FRED fetch returned no rows. Check FRED_API_KEY and network access."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(raw_long, out_path)
    return out_path


def _summarise(out_path: Path) -> None:
    import pandas as pd

    raw_long = pd.read_parquet(out_path)
    logger.info("raw_macro_written", extra={"path": str(out_path), "rows": int(len(raw_long))})

    summary = (
        raw_long.groupby("series_id")["reference_date"].agg(["min", "max", "count"]).reset_index()
    )
    print(f"Raw FRED panel written to {out_path}")
    print(f"Total rows: {len(raw_long):,}")
    print(summary.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch raw FRED macro panel.")
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_MACRO_PARQUET,
        help=f"Output parquet path (default: {RAW_MACRO_PARQUET}).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    logger.info("fetch_raw_macro_start", extra={"series": list(DEFAULT_FRED_SERIES_MAP)})

    fetch_and_write(args.out)
    _summarise(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
