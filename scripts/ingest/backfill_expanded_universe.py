#!/usr/bin/env python3
"""
Run the full expanded-universe backfill end to end. Designed to be left overnight.

Every step delegates to a script that is already resumable (it skips symbols whose
raw file exists and writes atomically), so this orchestrator inherits that: if the
machine sleeps, the network drops, or the process is killed, re-running the same
command picks up where it stopped. Nothing is recomputed that was already fetched.

Steps, in dependency order:

    1. universe      build the symbol table (screener + delisted)
    2. prices        dividend-adjusted OHLCV per symbol
    3. fundamentals  income / balance / cash-flow statements
    4. market_caps   daily market cap history, then the market cap panel
    5. panels        derive OHLCV + microstructure + PIT fundamentals + factors

Steps 2-4 are network-bound and dominate the runtime; step 5 is CPU-bound and
takes minutes. Run ``--estimate`` first to see the call budget.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/backfill_expanded_universe.py --estimate
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/backfill_expanded_universe.py
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/backfill_expanded_universe.py --steps prices,panels
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/backfill_expanded_universe.py --skip universe

Leave it running with output captured:
    ... scripts/ingest/backfill_expanded_universe.py > runtime/logs/expanded_backfill.log 2>&1 &
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.vendors.fmp.prices import generate_date_chunks
from core.data.vendors.fmp.storage import load_fetch_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_expanded_universe")

PYTHON = "/opt/anaconda3/envs/quant/bin/python"
UNIVERSE_FILE = ROOT / "data" / "raw" / "fmp" / "universe" / "us_equity_universe.parquet"

STEP_ORDER = ("universe", "prices", "fundamentals", "market_caps", "panels")
_CALLS_PER_MINUTE = 500.0


def step_commands(universe_file: Path) -> dict[str, list[list[str]]]:
    """Map each step name to the commands it runs, in order."""
    universe_args = ["--universe-file", str(universe_file)]
    return {
        "universe": [[PYTHON, "scripts/build/build_fmp_universe.py"]],
        "prices": [[PYTHON, "scripts/ingest/fetch_fmp_prices.py", *universe_args]],
        "fundamentals": [[PYTHON, "scripts/ingest/fetch_fmp_fundamentals.py", *universe_args]],
        "market_caps": [[PYTHON, "scripts/ingest/fetch_fmp_market_caps.py", *universe_args]],
        "panels": [
            [PYTHON, "scripts/ingest/fetch_fmp_prices.py", "--build-panel"],
            [PYTHON, "scripts/build/build_ohlcv_panel.py"],
            [PYTHON, "scripts/build/build_fundamentals_panel.py"],
            [PYTHON, "scripts/build/build_event_factors.py"],
            [PYTHON, "scripts/build/build_dollar_adv.py"],
        ],
    }


def estimate_budget(universe_file: Path) -> None:
    """Log the call count and wall-clock estimate for the network-bound steps."""
    if not universe_file.exists():
        logger.warning("No universe table at %s — run the 'universe' step first", universe_file)
        return

    windows = load_fetch_windows(universe_file)
    n_symbols = len(windows)
    price_chunks = sum(len(generate_date_chunks(start, end)) for start, end in windows.values())
    # Market caps use the same 5-year chunking as prices.
    budget = {
        "prices": price_chunks,
        "fundamentals": n_symbols * 3,
        "market_caps": price_chunks,
    }
    total = sum(budget.values())
    logger.info("Universe: %s symbols", f"{n_symbols:,}")
    for step, calls in budget.items():
        logger.info(
            "  %-13s ~%9s calls  (~%.1f h)", step, f"{calls:,}", calls / _CALLS_PER_MINUTE / 60
        )
    logger.info(
        "  %-13s ~%9s calls  (~%.1f h total)",
        "TOTAL",
        f"{total:,}",
        total / _CALLS_PER_MINUTE / 60,
    )


def run_step(name: str, commands: list[list[str]]) -> bool:
    """
    Run one step's commands, returning False on the first failure.

    Args:
        name: Step name, for logging.
        commands: Commands to run in order.

    Returns:
        True when every command exited 0.
    """
    logger.info("=" * 70)
    logger.info("STEP %s", name.upper())
    logger.info("=" * 70)
    started = time.monotonic()

    for command in commands:
        logger.info("$ %s", " ".join(command))
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            logger.error(
                "STEP %s FAILED (exit %d) after %.1f min — rerun the same command to resume",
                name,
                result.returncode,
                (time.monotonic() - started) / 60,
            )
            return False

    logger.info("STEP %s done in %.1f min", name, (time.monotonic() - started) / 60)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the expanded-universe backfill")
    parser.add_argument("--universe-file", type=Path, default=UNIVERSE_FILE)
    parser.add_argument("--steps", type=str, default=None, help=f"Subset of {','.join(STEP_ORDER)}")
    parser.add_argument("--skip", type=str, default=None, help="Steps to skip")
    parser.add_argument("--estimate", action="store_true", help="Print the call budget and exit")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failed step instead of stopping",
    )
    args = parser.parse_args()

    if args.estimate:
        estimate_budget(args.universe_file)
        return

    steps = [s.strip() for s in args.steps.split(",")] if args.steps else list(STEP_ORDER)
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",")}
        steps = [s for s in steps if s not in skip]
    if unknown := set(steps) - set(STEP_ORDER):
        raise SystemExit(f"Unknown steps: {sorted(unknown)}; known: {list(STEP_ORDER)}")

    commands = step_commands(args.universe_file)
    logger.info("Running steps: %s", ", ".join(steps))
    estimate_budget(args.universe_file)

    overall_started = time.monotonic()
    failed: list[str] = []
    for step in steps:
        if not run_step(step, commands[step]):
            failed.append(step)
            if not args.continue_on_error:
                raise SystemExit(f"Stopped at failed step: {step}")

    elapsed_hours = (time.monotonic() - overall_started) / 3600
    if failed:
        logger.warning("Finished in %.2f h with failed steps: %s", elapsed_hours, failed)
    else:
        logger.info("All steps finished in %.2f h", elapsed_hours)
        logger.info("Next: review coverage, then update docs/data/DATA_INVENTORY.md §1")


if __name__ == "__main__":
    main()
