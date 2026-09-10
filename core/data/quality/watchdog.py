"""Continuous health checks that run unattended and escalate when they fail.

The gap this closes: the platform already had a *quarantine scanner* (judgement
calls about suspicious symbols), *structural validation*
(:mod:`core.data.quality.validation`, impossible values), and *guarded writes*
(:mod:`core.data.store.artifacts`, a build cannot clobber a panel). All three are
point-in-time — they fire when something runs.

Nothing watched the artifacts *between* runs. A cron job that silently stopped
firing, a panel that stopped advancing, an overnight rebuild killed by the OS
partway through — all of these leave the platform serving stale or truncated data
that looks completely normal, and are only noticed when a research result comes
back strange.

This module answers one question on a schedule: **is the data on disk right now
something I would let a backtest read?** It is deliberately cheap — parquet
metadata and file mtimes, no data scans — so it can run every few hours.

Severity contract matches :mod:`core.data.quality.validation`:
- ``error``   — do not trust results computed against this data until fixed.
- ``warning`` — degraded; worth looking at, not a stop-work.
- ``ok``      — check ran and passed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
FACTORS_DIR = ROOT / "data" / "factors"
QUALITY_DIR = ROOT / "data" / "quality"
LOGS_DIR = ROOT / "runtime" / "logs"

STATUS_FILE = QUALITY_DIR / "watchdog_status.json"
BASELINE_FILE = QUALITY_DIR / "artifact_baseline.json"

Status = Literal["ok", "warning", "error"]

# Panels a backtest can reach. Missing or collapsed => research is unsafe.
CRITICAL_PANELS: tuple[str, ...] = (
    "prices.parquet",
    "factors_all.parquet",
    "factors_fundamental.parquet",
    "factors_microstructure.parquet",
    "factors_earnings_surprise.parquet",
    "factors_vendor_metrics.parquet",
    "dollar_adv_21d.parquet",
)

# A panel losing more than this fraction of its rows between watchdog runs is
# treated as damage, not as a legitimate rebuild.
COLLAPSE_TOLERANCE = 0.10

# The canonical panel should advance on every trading day. Allow a long weekend
# plus a holiday before calling it stale.
MAX_PANEL_STALENESS_DAYS = 5

# Scheduled jobs, how often their log should be touched, and the line each
# writes when a run finishes well. A failure marker only counts if it appears
# AFTER the last success marker: otherwise a job that failed once stayed red
# until 8 KB of clean output had pushed the old traceback out of the tail,
# and a fixed job could not clear its own alarm.
SCHEDULED_JOBS: tuple[tuple[str, int, str], ...] = (
    ("update.log", 2, "Incremental update completed successfully"),
    ("commodities_update.log", 2, "DONE"),
    ("market_caps_update.log", 2, "COMPLETE"),
    ("watchdog.log", 2, "Watchdog verdict"),
)

# Log lines that mean a job ended badly even though it produced output.
FAILURE_MARKERS: tuple[str, ...] = ("Traceback", "CRITICAL", "ERROR", "Killed")


def failure_markers_after_last_success(tail: str, success_marker: str) -> list[str]:
    """Failure markers that occur after the final success line in ``tail``.

    Python block-buffers stdout when a job's output is redirected to a file,
    so the tail can interleave several runs; judging only what follows the
    last success line is what makes "the most recent run failed" mean that.
    """
    cut = tail.rfind(success_marker)
    recent = tail[cut + len(success_marker) :] if cut >= 0 else tail
    return [marker for marker in FAILURE_MARKERS if marker in recent]


# Invariants that are permanently violated *by design*, with the reason. These
# report as "ok" so the banner stays meaningful: a monitor that is always yellow
# is a monitor nobody reads, and these conditions are already disclosed in the
# caveat registry rather than being news.
ACKNOWLEDGED_INVARIANTS: dict[str, str] = {
    "extreme_return": (
        "expected — the raw panel keeps vendor values verbatim (ADR 0012 fidelity "
        "rule); core.data.factors.returns rejects them at compute time"
    ),
}


@dataclass
class Check:
    """One watchdog verdict."""

    name: str
    status: Status
    detail: str
    facts: dict[str, Any] = field(default_factory=dict)


def _panel_rows(path: Path) -> int | None:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001 - an unreadable panel is itself the finding
        return None


def _load_baseline() -> dict[str, int]:
    if not BASELINE_FILE.exists():
        return {}
    try:
        return json.loads(BASELINE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Unreadable baseline at %s; treating as absent", BASELINE_FILE)
        return {}


def check_panels_present() -> list[Check]:
    """Every panel a backtest can reach exists and is readable."""
    checks: list[Check] = []
    for name in CRITICAL_PANELS:
        path = FACTORS_DIR / name
        if not path.exists():
            checks.append(
                Check(f"panel_present:{name}", "error", "Panel is missing from data/factors/")
            )
            continue
        rows = _panel_rows(path)
        if rows is None:
            checks.append(
                Check(
                    f"panel_present:{name}",
                    "error",
                    "Panel exists but its parquet metadata cannot be read (truncated write?)",
                )
            )
        elif rows == 0:
            checks.append(Check(f"panel_present:{name}", "error", "Panel is empty"))
        else:
            checks.append(Check(f"panel_present:{name}", "ok", f"{rows:,} rows", {"rows": rows}))
    return checks


def check_panels_not_collapsed(baseline: dict[str, int]) -> list[Check]:
    """
    No panel has lost a material fraction of its rows since the last clean run.

    This is the between-runs counterpart to the write guard: it catches damage
    done by something that did not go through :mod:`core.data.store.artifacts` at all —
    a manual overwrite, an interrupted script, a disk problem.
    """
    checks: list[Check] = []
    for name in CRITICAL_PANELS:
        previous = baseline.get(name)
        path = FACTORS_DIR / name
        if previous is None or not path.exists():
            continue
        rows = _panel_rows(path)
        if rows is None:
            continue
        retained = rows / previous if previous else 1.0
        if retained < 1.0 - COLLAPSE_TOLERANCE:
            checks.append(
                Check(
                    f"panel_size:{name}",
                    "error",
                    f"Lost {1 - retained:.0%} of rows since last clean run "
                    f"({previous:,} -> {rows:,}). A partial build or manual write "
                    "reached a production panel.",
                    {"previous_rows": previous, "rows": rows},
                )
            )
        else:
            checks.append(
                Check(
                    f"panel_size:{name}",
                    "ok",
                    f"{rows:,} rows ({rows - previous:+,} since baseline)",
                    {"previous_rows": previous, "rows": rows},
                )
            )
    return checks


def check_panel_freshness(now: datetime | None = None) -> list[Check]:
    """The canonical price panel is still advancing with the market."""
    path = FACTORS_DIR / "prices.parquet"
    if not path.exists():
        return [Check("panel_freshness", "error", "Canonical price panel is missing")]

    reference = now or datetime.now(timezone.utc)
    try:
        index = pd.read_parquet(path, columns=[]).index
    except Exception as exc:  # noqa: BLE001
        return [Check("panel_freshness", "error", f"Cannot read the price panel index: {exc}")]

    if len(index) == 0:
        return [Check("panel_freshness", "error", "Canonical price panel has no dates")]

    last = index.max()
    last_utc = last.tz_convert("UTC") if last.tzinfo else last.tz_localize("UTC")
    age_days = (reference - last_utc).days
    facts = {"last_date": str(last.date()), "age_days": age_days}

    if age_days > MAX_PANEL_STALENESS_DAYS:
        return [
            Check(
                "panel_freshness",
                "error",
                f"Last price is {last.date()} ({age_days} days old). The daily "
                "update is not landing; every backtest is running on stale data.",
                facts,
            )
        ]
    return [Check("panel_freshness", "ok", f"Last price {last.date()} ({age_days}d old)", facts)]


def check_series_freshness(now: datetime | None = None) -> list[Check]:
    """Every monitored commodity and macro series is still arriving.

    The equity panel check above would not have caught the commodity panel
    freezing for two months while its nightly job logged "up to date"; this
    one judges each series against its own cadence and publication lag.
    """
    # Imported here so the fast watchdog path does not pay for the research
    # package on every 4-hourly run when it only needs file metadata.
    from core.data.factors.macro import RAW_MACRO_PARQUET
    from core.research.monitor import pivot_raw_macro, staleness_board

    reference = now or datetime.now(timezone.utc)
    as_of = pd.Timestamp(reference).tz_convert("UTC").tz_localize(None)
    panels: dict[str, pd.DataFrame] = {}
    commodities = ROOT / "data" / "commodities" / "prices.parquet"
    try:
        if commodities.exists():
            panels["fmp"] = pd.read_parquet(commodities)
        if RAW_MACRO_PARQUET.exists():
            panels["fred"] = pivot_raw_macro(pd.read_parquet(RAW_MACRO_PARQUET))
    except Exception as exc:  # noqa: BLE001
        return [Check("series_freshness", "error", f"Cannot read monitored panels: {exc}")]

    board = staleness_board(panels, as_of)
    stale = [r for r in board if r["status"] == "stale"]
    late = [r for r in board if r["status"] == "late"]
    empty = [r for r in board if r["status"] == "empty"]
    facts = {
        "n_series": len(board),
        "stale": [f"{r['series_id']} (last {r['last_date']})" for r in stale],
        "late": [f"{r['series_id']} (last {r['last_date']})" for r in late],
        "empty": [r["series_id"] for r in empty],
    }
    if stale or empty:
        names = ", ".join(str(r["series_id"]) for r in (stale + empty)[:8])
        return [
            Check(
                "series_freshness",
                "error",
                f"{len(stale)} stale and {len(empty)} missing monitored series ({names}). "
                "Their update job is not landing; the Data Monitor page shows which.",
                facts,
            )
        ]
    if late:
        names = ", ".join(str(r["series_id"]) for r in late[:8])
        return [
            Check(
                "series_freshness",
                "warning",
                f"{len(late)} monitored series late ({names})",
                facts,
            )
        ]
    return [Check("series_freshness", "ok", f"All {len(board)} monitored series fresh", facts)]


def check_scheduled_jobs(now: datetime | None = None) -> list[Check]:
    """
    Each cron job wrote to its log recently, and did not end in a traceback.

    A job that stops firing produces no error anywhere — its log simply stops
    changing. That silence is the failure mode this check exists for.
    """
    reference = now or datetime.now(timezone.utc)
    checks: list[Check] = []
    for log_name, max_age_days, success_marker in SCHEDULED_JOBS:
        path = LOGS_DIR / log_name
        if not path.exists():
            checks.append(
                Check(f"job:{log_name}", "warning", "No log yet; the job may never have run")
            )
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age = reference - mtime
        if age > timedelta(days=max_age_days):
            checks.append(
                Check(
                    f"job:{log_name}",
                    "error",
                    f"Silent for {age.days} days (expected every {max_age_days}). "
                    "The scheduled job has stopped running.",
                    {"last_run": mtime.isoformat(), "age_days": age.days},
                )
            )
            continue

        tail = _read_tail(path)
        hits = failure_markers_after_last_success(tail, success_marker)
        if hits:
            checks.append(
                Check(
                    f"job:{log_name}",
                    "error",
                    f"Ran {age.days}d ago but its log ends with {', '.join(hits)}",
                    {"last_run": mtime.isoformat(), "markers": hits},
                )
            )
        else:
            checks.append(
                Check(
                    f"job:{log_name}",
                    "ok",
                    f"Ran {age.days}d ago, clean",
                    {"last_run": mtime.isoformat()},
                )
            )
    return checks


def _read_tail(path: Path, n_bytes: int = 8192) -> str:
    """Last few KB of a log; failures show up at the end, and logs get large."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - n_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def check_structural_invariants() -> list[Check]:
    """Run the panel invariants from :mod:`core.data.quality.validation` on the price panel."""
    from core.data.quality.validation import validate_price_panel

    path = FACTORS_DIR / "prices.parquet"
    if not path.exists():
        return []
    try:
        prices = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return [Check("invariants", "error", f"Cannot read the price panel: {exc}")]

    violations = validate_price_panel(prices)
    if not violations:
        return [Check("invariants", "ok", "No structural violations")]

    checks: list[Check] = []
    for violation in violations:
        acknowledged = ACKNOWLEDGED_INVARIANTS.get(violation.check)
        if acknowledged:
            checks.append(
                Check(
                    f"invariant:{violation.check}",
                    "ok",
                    f"{violation.count:,} occurrences, {acknowledged}",
                    {"acknowledged": True, "examples": list(violation.examples)},
                )
            )
            continue
        checks.append(
            Check(
                f"invariant:{violation.check}",
                "error" if violation.severity == "error" else "warning",
                f"{violation.count:,} occurrences — {violation.detail}",
                {"examples": list(violation.examples)},
            )
        )
    return checks


def run_watchdog(deep: bool = False, now: datetime | None = None) -> dict[str, Any]:
    """
    Run every check and assemble a status snapshot.

    Args:
        deep: Also read full panel data for structural invariants. Slower
            (~10s) so the frequent schedule leaves it off and the daily one
            turns it on.
        now: Reference time, for tests.

    Returns:
        A snapshot with an overall verdict, the individual checks, and the
        artifact row counts to use as the next baseline.
    """
    baseline = _load_baseline()
    checks: list[Check] = []
    checks += check_panels_present()
    checks += check_panels_not_collapsed(baseline)
    checks += check_panel_freshness(now=now)
    checks += check_series_freshness(now=now)
    checks += check_scheduled_jobs(now=now)
    if deep:
        checks += check_structural_invariants()

    errors = [c for c in checks if c.status == "error"]
    warnings = [c for c in checks if c.status == "warning"]
    overall: Status = "error" if errors else ("warning" if warnings else "ok")

    return {
        "generated_at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC"),
        "status": overall,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "deep": deep,
        "checks": [asdict(c) for c in checks],
        "summary": _summarize(overall, errors, warnings),
    }


def _summarize(overall: Status, errors: list[Check], warnings: list[Check]) -> str:
    """One sentence a human can act on without opening anything else."""
    if overall == "ok":
        return "All data checks passed."
    parts: list[str] = []
    if errors:
        parts.append(f"{len(errors)} error(s): " + "; ".join(c.detail for c in errors[:3]))
    if warnings:
        parts.append(f"{len(warnings)} warning(s): " + "; ".join(c.detail for c in warnings[:2]))
    return " | ".join(parts)


def current_row_counts() -> dict[str, int]:
    """Row counts for the critical panels, for use as the next clean baseline."""
    counts: dict[str, int] = {}
    for name in CRITICAL_PANELS:
        rows = _panel_rows(FACTORS_DIR / name)
        if rows:
            counts[name] = rows
    return counts


def write_status(snapshot: dict[str, Any]) -> None:
    """Persist the snapshot for the API and the frontend banner."""
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(snapshot, indent=2))


def write_baseline(counts: dict[str, int]) -> None:
    """Record row counts as the reference for future collapse detection."""
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(json.dumps(counts, indent=2))


def load_status() -> dict[str, Any] | None:
    """Read the last snapshot, or None if the watchdog has never run."""
    if not STATUS_FILE.exists():
        return None
    try:
        return json.loads(STATUS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
