#!/usr/bin/env python3
"""
Execute the ADR-0013 cutover: canonical price panel 774 -> full US universe.

What this changes, in one step, with an archive and a verifiable report:

    data/factors/prices.parquet   774 S&P names  ->  ~8,900 US symbols
    data/factors/archive/prices_sp500_774_<date>.parquet   (the old panel, verbatim)

Everything downstream keys off ``prices.parquet`` — its columns are the universe
and its index is the trading calendar — so this is the single swap that widens
the whole platform. Derived panels must then be rebuilt (``--rebuild``).

Safety properties:

- The old panel is archived BEFORE the swap and the archive is verified readable.
- The new panel must pass structural checks (tz-aware monotonic unique index,
  calendar match, no column loss vs the archive) or the swap is refused.
- ``--dry-run`` reports what would change without touching anything.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build/cutover_canonical_panel.py --dry-run
    /opt/anaconda3/envs/quant/bin/python scripts/build/cutover_canonical_panel.py
    /opt/anaconda3/envs/quant/bin/python scripts/build/cutover_canonical_panel.py --rebuild
"""

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cutover_canonical_panel")

PYTHON = "/opt/anaconda3/envs/quant/bin/python"
FACTORS = ROOT / "data" / "factors"
CANONICAL = FACTORS / "prices.parquet"
EXPANDED = FACTORS / "prices_fmp.parquet"
ARCHIVE_DIR = FACTORS / "archive"

# Rebuilt in dependency order after the swap.
REBUILD_STEPS: tuple[tuple[str, list[str]], ...] = (
    ("index membership labels", [PYTHON, "scripts/build/build_index_membership.py"]),
    # backfill_all.py is NOT usable here: it refetches prices and is scoped to the
    # S&P universe, both wrong post-cutover. build_price_factors.py is the batched
    # rebuild from whatever the canonical panel now holds.
    ("price factors", [PYTHON, "scripts/build/build_price_factors.py"]),
    ("OHLCV + microstructure", [PYTHON, "scripts/build/build_ohlcv_panel.py"]),
    ("PIT fundamentals + factors", [PYTHON, "scripts/build/build_fundamentals_panel.py"]),
    ("event + vendor factors", [PYTHON, "scripts/build/build_event_factors.py"]),
    ("dollar ADV", [PYTHON, "scripts/build/build_dollar_adv.py"]),
    ("data health audit", [PYTHON, "scripts/ops/audit_data_health.py"]),
)


def validate_new_panel(new_panel: pd.DataFrame, old_panel: pd.DataFrame) -> list[str]:
    """
    Structural checks the replacement panel must pass before it is installed.

    Args:
        new_panel: Candidate canonical panel.
        old_panel: Panel being replaced.

    Returns:
        List of failure messages; empty means the swap is safe.
    """
    failures: list[str] = []
    index = new_panel.index

    if index.tz is None:
        failures.append("new panel index is tz-naive (must be America/New_York)")
    if not index.is_monotonic_increasing:
        failures.append("new panel index is not monotonic increasing")
    if not index.is_unique:
        failures.append("new panel index has duplicate dates")
    if new_panel.shape[1] <= old_panel.shape[1]:
        failures.append(
            f"new panel has {new_panel.shape[1]} symbols, not more than the old "
            f"{old_panel.shape[1]} — this is supposed to be an expansion"
        )

    # A cutover must not silently drop names current research depends on.
    dropped = set(old_panel.columns) - set(new_panel.columns)
    if dropped:
        failures.append(f"{len(dropped)} symbols would be dropped, e.g. {sorted(dropped)[:10]}")

    # The expanded panel was calendar-filtered at build time; verify it held.
    extra_dates = index.difference(old_panel.index)
    weekend = int((pd.Series(extra_dates.dayofweek) >= 5).sum()) if len(extra_dates) else 0
    if weekend:
        failures.append(f"new panel carries {weekend} weekend dates (bad prints not filtered)")

    # The daily updater keeps the canonical panel fresher than a staged rebuild.
    # Silently cutting over to a stale panel loses recent history for every
    # symbol — the one failure mode that looks like nothing happened.
    lost_dates = old_panel.index.difference(index)
    if len(lost_dates):
        failures.append(
            f"new panel is missing {len(lost_dates)} date(s) the canonical panel has "
            f"(latest {lost_dates.max().date()}); refresh it before cutting over"
        )

    return failures


def run_rebuild() -> bool:
    """Rebuild every derived artifact against the new canonical panel."""
    for label, command in REBUILD_STEPS:
        logger.info("=" * 66)
        logger.info("REBUILD: %s", label)
        logger.info("$ %s", " ".join(command))
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            logger.error(
                "Rebuild step %r failed (exit %d); fix and rerun", label, result.returncode
            )
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut the canonical panel over (ADR 0013)")
    parser.add_argument("--dry-run", action="store_true", help="Report only; change nothing")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild derived panels after swap")
    parser.add_argument("--rebuild-only", action="store_true", help="Skip the swap; rebuild only")
    parser.add_argument(
        "--refresh-first",
        action="store_true",
        help="Fetch the trailing window for the full universe and rebuild the expanded "
        "panel before swapping (needed when the daily updater has run since the backfill)",
    )
    args = parser.parse_args()

    if args.refresh_first:
        universe_file = ROOT / "data" / "raw" / "fmp" / "universe" / "us_equity_universe.parquet"
        for label, command in (
            (
                "refresh raw prices (trailing window, full universe)",
                [
                    PYTHON,
                    "scripts/ingest/fetch_fmp_prices.py",
                    "--universe-file",
                    str(universe_file),
                    "--start",
                    (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                    "--refresh-recent",
                ],
            ),
            ("rebuild expanded panel", [PYTHON, "scripts/ingest/fetch_fmp_prices.py", "--build-panel"]),
        ):
            logger.info("PRE-CUTOVER: %s", label)
            if subprocess.run(command, cwd=ROOT).returncode != 0:
                raise SystemExit(f"Pre-cutover step failed: {label}")

    if args.rebuild_only:
        raise SystemExit(0 if run_rebuild() else 1)

    if not EXPANDED.exists():
        raise SystemExit(f"No expanded panel at {EXPANDED}; run fetch_fmp_prices.py --build-panel")

    old_panel = pd.read_parquet(CANONICAL)
    new_panel = pd.read_parquet(EXPANDED)

    logger.info(
        "Current canonical: %d symbols x %d dates (%s -> %s)",
        old_panel.shape[1],
        old_panel.shape[0],
        old_panel.index.min().date(),
        old_panel.index.max().date(),
    )
    logger.info(
        "Replacement:       %d symbols x %d dates (%s -> %s)",
        new_panel.shape[1],
        new_panel.shape[0],
        new_panel.index.min().date(),
        new_panel.index.max().date(),
    )
    added = len(set(new_panel.columns) - set(old_panel.columns))
    logger.info("Symbols added: %d", added)

    failures = validate_new_panel(new_panel, old_panel)
    if failures:
        for failure in failures:
            logger.error("VALIDATION: %s", failure)
        raise SystemExit("Cutover refused — the replacement panel failed validation")
    logger.info("Validation passed")

    if args.dry_run:
        logger.info("--dry-run: no files changed")
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_DIR / f"prices_sp500_{old_panel.shape[1]}_{date.today():%Y%m%d}.parquet"
    shutil.copy2(CANONICAL, archive_path)
    archived = pd.read_parquet(archive_path)
    if archived.shape != old_panel.shape:
        raise SystemExit(f"Archive verification failed: {archived.shape} != {old_panel.shape}")
    logger.info("Archived old panel -> %s (verified %s)", archive_path.name, archived.shape)

    shutil.copy2(EXPANDED, CANONICAL)
    installed = pd.read_parquet(CANONICAL)
    logger.info("CUTOVER DONE: canonical panel is now %s", installed.shape)
    logger.info("Reproduce pre-cutover results against %s", archive_path)

    if args.rebuild and not run_rebuild():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
