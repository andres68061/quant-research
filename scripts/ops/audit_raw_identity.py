#!/usr/bin/env python3
"""
Verify that every raw file holds the data it was fetched for.

The raw layer binds a payload to a company by the partition key the request was
made with. That binding is what stops one company's earnings, sector, or
fundamentals being attributed to another, so it should be checked mechanically
rather than assumed.

Three distinct things are checked:

1. **Stamp agreement.** Files written by the framework carry a ``_request_key``
   column. Where present, it must equal the filename. A disagreement means a file
   was renamed, copied, or written by something other than the runner.
2. **Payload agreement.** For endpoints whose rows describe the requested entity,
   any ``symbol`` column must contain the requested symbol. A file for ``AAPL``
   whose rows all say ``MSFT`` is the vendor returning the wrong company.
3. **Filename collisions.** Two different tickers must never map to one
   filesystem-safe key. ``BRK/B`` and ``BRK-B`` both reduce to ``BRK-B``, which
   would silently make one overwrite the other.

Endpoints declared ``subject: related`` in the manifest are exempt from check 2
by design: ``stock-peers`` for ``LUV`` returns ``CRS``, ``JBHT`` and ``JOBY`` —
the peers of LUV, not LUV — and only the request key identifies the subject.

Usage:
    python scripts/ops/audit_raw_identity.py
    python scripts/ops/audit_raw_identity.py --sample 200 --dataset earnings
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ingest.catalog import build_specs, load_manifest
from core.ingest.plan import GLOBAL_KEY, safe_key
from core.ingest.runner import REQUEST_KEY_COLUMN
from core.ingest.spec import Subject

logger = logging.getLogger("audit_raw_identity")

RAW = ROOT / "data" / "raw" / "fmp"
SECURITY_MASTER = ROOT / "data" / "universe" / "security_master.parquet"


def audit_key_collisions() -> list[tuple[str, list[str]]]:
    """
    Find universe symbols that would share one filename.

    Returns:
        ``(safe_key, [symbols])`` for every key claimed by more than one symbol.
    """
    if not SECURITY_MASTER.is_file():
        return []
    symbols = pd.read_parquet(SECURITY_MASTER)["symbol"].dropna().astype(str).unique()
    buckets: dict[str, list[str]] = defaultdict(list)
    for symbol in symbols:
        buckets[safe_key(symbol)].append(symbol)
    return [(key, names) for key, names in buckets.items() if len(names) > 1]


def audit_dataset(directory: Path, subject: Subject, sample: int) -> dict[str, int]:
    """
    Check one dataset directory's files against their filenames.

    Args:
        directory: Dataset directory under the raw layer.
        subject: Whether rows describe the requested entity.
        sample: Maximum files to open.

    Returns:
        Counters: files opened, stamp mismatches, payload mismatches, unstamped.
    """
    files = [f for f in directory.glob("*.parquet") if f.stat().st_size > 0]
    if not files:
        return {}
    chosen = random.sample(files, min(sample, len(files)))
    counts = {"checked": 0, "stamp_mismatch": 0, "payload_mismatch": 0, "unstamped": 0}
    for path in chosen:
        try:
            frame = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - an unreadable file is a separate concern
            continue
        if frame.empty:
            continue
        counts["checked"] += 1
        expected = path.stem

        if REQUEST_KEY_COLUMN in frame.columns:
            stamped = {str(v) for v in frame[REQUEST_KEY_COLUMN].dropna().unique()}
            if stamped != {expected}:
                counts["stamp_mismatch"] += 1
                logger.error("%s: stamped %s but filename says %s", path, stamped, expected)
        else:
            counts["unstamped"] += 1

        # Global endpoints are keyed "_all", which is never a ticker, so a symbol
        # column in them says nothing about identity.
        if subject is Subject.REQUEST_KEY and expected != GLOBAL_KEY and "symbol" in frame.columns:
            found = {str(v) for v in frame["symbol"].dropna().unique()}
            if found and expected not in found:
                counts["payload_mismatch"] += 1
                logger.error(
                    "%s: requested %s but payload names %s", path, expected, sorted(found)[:3]
                )
    return counts


def main() -> int:
    """Run the audit. Returns non-zero when any mismatch or collision is found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=60, help="files per dataset")
    parser.add_argument("--dataset", default=None, help="audit one dataset only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    random.seed(args.seed)

    subjects = {spec.name: spec.subject for spec in build_specs(load_manifest("fmp"))}

    collisions = audit_key_collisions()
    if collisions:
        logger.error("%d filesystem-key collisions in the universe:", len(collisions))
        for key, names in collisions[:10]:
            logger.error("  %s <- %s", key, names)
    else:
        logger.info("no filesystem-key collisions in the universe")

    directories = (
        [RAW / args.dataset] if args.dataset else sorted(d for d in RAW.iterdir() if d.is_dir())
    )
    # Directories written by older bespoke scripts are not keyed by request and
    # cannot be audited by this rule; name them rather than judge them.
    unmanaged = [d.name for d in directories if d.name not in subjects]
    directories = [d for d in directories if d.name in subjects]
    if unmanaged:
        logger.info(
            "%d directories predate the framework and are not request-keyed: %s",
            len(unmanaged),
            ", ".join(sorted(unmanaged)),
        )
    totals = {"checked": 0, "stamp_mismatch": 0, "payload_mismatch": 0, "unstamped": 0}
    print(f"\n{'dataset':<40}{'checked':>9}{'stamp bad':>11}{'payload bad':>13}{'unstamped':>11}")
    for directory in directories:
        subject = subjects.get(directory.name, Subject.REQUEST_KEY)
        counts = audit_dataset(directory, subject, args.sample)
        if not counts or not counts["checked"]:
            continue
        for key in totals:
            totals[key] += counts[key]
        tag = " (related)" if subject is Subject.RELATED else ""
        print(
            f"{directory.name + tag:<40}{counts['checked']:>9}"
            f"{counts['stamp_mismatch']:>11}{counts['payload_mismatch']:>13}"
            f"{counts['unstamped']:>11}"
        )

    print(
        f"\n{'TOTAL':<40}{totals['checked']:>9}{totals['stamp_mismatch']:>11}"
        f"{totals['payload_mismatch']:>13}{totals['unstamped']:>11}"
    )
    if totals["unstamped"]:
        print(
            f"\n{totals['unstamped']:,} files predate request-key stamping. Their identity "
            "still rests on the filename; they are re-stamped whenever refetched."
        )
    failed = bool(collisions) or totals["stamp_mismatch"] or totals["payload_mismatch"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
