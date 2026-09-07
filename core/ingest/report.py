"""Human-readable run reports rendered from the ingestion journal.

The journal answers questions precisely but in SQL. This module renders the
answers a person actually wants after a long backfill: what landed, what the
vendor had nothing for, what failed and with which status, and which endpoints
failed so broadly that the result should not be trusted.

The distinction the report insists on is between *empty* and *failed*. An
endpoint that returns no rows for 8,000 of 9,011 symbols is not broken — most
companies have no executive-compensation filings — whereas one that 402s for
every symbol is unentitled and its directory should not be mistaken for data.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _latest_run(connection: sqlite3.Connection) -> Optional[str]:
    row = connection.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return row[0] if row else None


def render_report(journal_path: Path, run_id: Optional[str] = None) -> str:
    """
    Build a text report for one run.

    Args:
        journal_path: SQLite journal written by the run.
        run_id: Run to report on; defaults to the most recent.

    Returns:
        Formatted multi-section report.
    """
    if not journal_path.is_file():
        return f"no journal at {journal_path}"
    connection = sqlite3.connect(str(journal_path))
    try:
        run_id = run_id or _latest_run(connection)
        if run_id is None:
            return "journal contains no runs"

        head = connection.execute(
            "SELECT vendor, started_at, finished_at, n_tasks, argv FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        vendor, started, finished, n_tasks, argv = head
        elapsed = (finished or time.time()) - started

        lines = [
            f"Ingestion report — {vendor} — run {run_id}",
            "=" * 72,
            f"  started   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))}",
            f"  elapsed   {elapsed / 60:.1f} min" + ("" if finished else "  (STILL RUNNING)"),
            f"  planned   {n_tasks or 0:,} tasks",
            f"  argv      {argv}",
            "",
            "Outcomes",
            "-" * 72,
        ]
        totals = connection.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(n_rows),0), COALESCE(SUM(n_bytes),0)"
            " FROM tasks WHERE run_id=? GROUP BY status ORDER BY COUNT(*) DESC",
            (run_id,),
        ).fetchall()
        for status, count, rows, nbytes in totals:
            lines.append(
                f"  {status:10} {count:>9,} tasks  {rows:>14,} rows  {nbytes / 1e6:>10.1f} MB"
            )

        lines += [
            "",
            "Per endpoint",
            "-" * 72,
            f"  {'endpoint':<42}{'ok':>7}{'empty':>8}{'fail':>7}{'rows':>12}",
        ]
        per = connection.execute(
            "SELECT spec_name,"
            " SUM(status='ok'), SUM(status='empty'), SUM(status='failed'),"
            " COALESCE(SUM(n_rows),0)"
            " FROM tasks WHERE run_id=? GROUP BY spec_name ORDER BY SUM(status='failed') DESC,"
            " spec_name",
            (run_id,),
        ).fetchall()
        for name, ok, empty, failed, rows in per:
            lines.append(f"  {name:<42}{ok:>7,}{empty:>8,}{failed:>7,}{rows:>12,}")

        failing = [(n, ok, e, f) for n, ok, e, f, _ in per if f and f > (ok + e)]
        if failing:
            lines += [
                "",
                "Endpoints failing more often than succeeding — do not trust these",
                "-" * 72,
            ]
            for name, ok, empty, failed in failing:
                lines.append(f"  {name:<42} ok={ok:,} empty={empty:,} failed={failed:,}")

        codes = connection.execute(
            "SELECT COALESCE(http_code,0), COUNT(*), MIN(error) FROM tasks"
            " WHERE run_id=? AND status='failed' GROUP BY http_code ORDER BY COUNT(*) DESC LIMIT 15",
            (run_id,),
        ).fetchall()
        if codes:
            lines += ["", "Failures by HTTP status", "-" * 72]
            for code, count, sample in codes:
                lines.append(f"  HTTP {code or '—':<6}{count:>8,}   e.g. {(sample or '')[:90]}")
        return "\n".join(lines)
    finally:
        connection.close()


def write_report(journal_path: Path, report_dir: Path, run_id: str) -> Path:
    """
    Render a run report and save it beside the journal.

    Args:
        journal_path: Journal to read.
        report_dir: Directory for report files.
        run_id: Run to report on.

    Returns:
        Path to the written report.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{run_id}.txt"
    path.write_text(render_report(journal_path, run_id))
    return path
