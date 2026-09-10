"""Automatic data-quality scanning and symbol quarantine.

Scans the wide adjusted-close panel for corruption signatures and maintains a
persistent quarantine list at ``data/quality/quarantine.parquet``.

Policy: automatic quarantine requires INTERNAL corruption evidence in our own
panel (the spike-and-reverse bad-print signature — a real crash does not
un-crash). Cross-vendor disagreement, extreme-return counts, and stale runs
are review flags only: disagreement proves one vendor is wrong but not which
one (validated 2026-07-11: COST/NKE/HUBB diverged from yfinance only in the
1980s where yfinance is the unreliable side, and CRWD diverged because
yfinance missed a 2026 split — FMP was correct every time).

Statuses:
    quarantined  excluded from prices/factors at API load time
    flagged      visible for review, NOT excluded
    cleared      manually reviewed and kept; never re-quarantined by the scanner

The scanner never downgrades a manual ``cleared`` decision.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QUARANTINE_PATH = Path("data/quality/quarantine.parquet")

EXTREME_DAILY_RETURN = 0.75
SPIKE_REVERSAL_RETURN = 0.50
SPIKE_REVERSAL_WINDOW_DAYS = 3
STALE_RUN_DAYS = 15
ENTITY_MISMATCH_MIN_CORRELATION = 0.90
MIN_OVERLAP_DAYS_FOR_CORRELATION = 252

# Checks whose failure means the series cannot be trusted at all. Only
# internal bad-print evidence auto-quarantines; see module docstring.
_HARD_CHECKS = frozenset({"spike_reversal"})


def scan_extreme_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Flag symbols with repeated extreme daily moves (|return| > 75%).

    A single extreme day can be a genuine collapse (SBNY, 2023); three or more
    almost always indicate bad prints or recycled-ticker contamination.

    Args:
        prices: Wide panel, index = dates, columns = symbols, adjusted closes.

    Returns:
        DataFrame with columns ``symbol``, ``check``, ``value``, ``detail`` —
        one row per offending symbol (value = count of extreme days).
    """
    daily_returns = prices.pct_change(fill_method=None)
    extreme_counts = (daily_returns.abs() > EXTREME_DAILY_RETURN).sum()
    offenders = extreme_counts[extreme_counts >= 3]
    return pd.DataFrame(
        {
            "symbol": offenders.index,
            "check": "extreme_returns",
            "value": offenders.values.astype(float),
            "detail": [
                f"{int(n)} days with |daily return| > {EXTREME_DAILY_RETURN:.0%}"
                for n in offenders.values
            ],
        }
    )


def scan_spike_reversals(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Flag bad-print signatures: an extreme move that reverses within days.

    A real crash does not un-crash. A move of > +/-75% followed within 3
    trading days by a > 50% move in the opposite direction is the classic
    corrupted-quote pattern (e.g. a $0.65 stock printing $11,000 for a day).

    Args:
        prices: Wide panel, index = dates, columns = symbols.

    Returns:
        DataFrame with ``symbol``, ``check``, ``value``, ``detail`` rows
        (value = count of spike-reversal events).
    """
    daily_returns = prices.pct_change(fill_method=None)
    spike_up = daily_returns > EXTREME_DAILY_RETURN
    spike_down = daily_returns < -EXTREME_DAILY_RETURN
    rows = []
    for symbol in prices.columns:
        sym_ret = daily_returns[symbol]
        events = 0
        for spikes, opposite_sign in ((spike_up[symbol], -1.0), (spike_down[symbol], 1.0)):
            for spike_date in sym_ret.index[spikes.fillna(False)]:
                loc = sym_ret.index.get_loc(spike_date)
                window = sym_ret.iloc[loc + 1 : loc + 1 + SPIKE_REVERSAL_WINDOW_DAYS]
                if (opposite_sign * window > SPIKE_REVERSAL_RETURN).any():
                    events += 1
        if events >= 1:
            rows.append(
                {
                    "symbol": symbol,
                    "check": "spike_reversal",
                    "value": float(events),
                    "detail": f"{events} extreme moves reversed within {SPIKE_REVERSAL_WINDOW_DAYS} days",
                }
            )
    return pd.DataFrame(rows, columns=["symbol", "check", "value", "detail"])


def scan_stale_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Flag symbols with long runs of identical consecutive prices.

    The FMP panel contains only true trading days, so 15+ identical closes in
    a row indicate a frozen vendor feed or leftover forward-fill.

    Args:
        prices: Wide panel, index = dates, columns = symbols.

    Returns:
        DataFrame with ``symbol``, ``check``, ``value``, ``detail`` rows
        (value = longest identical run in days).
    """
    rows = []
    for symbol in prices.columns:
        series = prices[symbol].dropna()
        if len(series) < STALE_RUN_DAYS:
            continue
        run_ids = (series != series.shift()).cumsum()
        longest_run = int(series.groupby(run_ids).size().max())
        if longest_run >= STALE_RUN_DAYS:
            rows.append(
                {
                    "symbol": symbol,
                    "check": "stale_prices",
                    "value": float(longest_run),
                    "detail": f"longest identical-price run: {longest_run} trading days",
                }
            )
    return pd.DataFrame(rows, columns=["symbol", "check", "value", "detail"])


def scan_entity_mismatch(prices: pd.DataFrame, reference_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Flag symbols whose returns disagree with a second vendor's returns.

    Low correlation over a meaningful overlap means at least one vendor has a
    wrong series (ticker reuse, missed split, bad early data) — but not which
    one, so this is a review flag rather than an auto-quarantine.

    Args:
        prices: Wide panel under test.
        reference_prices: Wide panel from another vendor (same symbol scheme).

    Returns:
        DataFrame with ``symbol``, ``check``, ``value``, ``detail`` rows
        (value = daily return correlation).
    """
    shared = prices.columns.intersection(reference_prices.columns)
    test_returns = prices[shared].pct_change(fill_method=None)
    reference_returns = reference_prices[shared].reindex(prices.index).pct_change(fill_method=None)
    rows = []
    for symbol in shared:
        both = pd.concat([test_returns[symbol], reference_returns[symbol]], axis=1).dropna()
        if len(both) < MIN_OVERLAP_DAYS_FOR_CORRELATION:
            continue
        correlation = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
        if np.isfinite(correlation) and correlation < ENTITY_MISMATCH_MIN_CORRELATION:
            rows.append(
                {
                    "symbol": symbol,
                    "check": "entity_mismatch",
                    "value": correlation,
                    "detail": (
                        f"daily return correlation vs reference vendor = {correlation:.3f} "
                        f"over {len(both)} shared days"
                    ),
                }
            )
    return pd.DataFrame(rows, columns=["symbol", "check", "value", "detail"])


def repair_isolated_bad_prints(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove isolated single-day bad prints from a price panel.

    A bad print is a day whose price jumps > 75% away from the previous close
    and snaps back within 25% of the pre-spike level on the next observation
    (e.g. LEN 1990: $2.07 -> $4.33 -> $2.09). Real repricings persist; prints
    do not. The offending day's price is set to NaN — the raw layer keeps the
    vendor value, only the derived panel is cleaned.

    Args:
        prices: Wide adjusted-close panel.

    Returns:
        Tuple of (repaired panel, repair log). The log has one row per removed
        print: ``symbol``, ``date``, ``bad_price``, ``prev_price``, ``next_price``.

    Example:
        >>> panel, log = repair_isolated_bad_prints(prices)  # doctest: +SKIP
    """
    repaired = prices.copy()
    previous_close = prices.shift(1)
    next_close = prices.shift(-1)

    jump = (prices / previous_close - 1.0).abs() > EXTREME_DAILY_RETURN
    snap_back = (next_close / previous_close - 1.0).abs() < 0.25
    bad_print_mask = jump & snap_back & prices.notna() & previous_close.notna() & next_close.notna()

    log_rows = []
    for symbol in prices.columns:
        for date in prices.index[bad_print_mask[symbol]]:
            log_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "bad_price": float(prices.at[date, symbol]),
                    "prev_price": float(previous_close.at[date, symbol]),
                    "next_price": float(next_close.at[date, symbol]),
                }
            )
    repaired[bad_print_mask] = np.nan
    repair_log = pd.DataFrame(
        log_rows, columns=["symbol", "date", "bad_price", "prev_price", "next_price"]
    )
    if not repair_log.empty:
        logger.info(
            "Repaired %d isolated bad prints across %d symbols",
            len(repair_log),
            repair_log["symbol"].nunique(),
        )
    return repaired, repair_log


def scan_price_panel(
    prices: pd.DataFrame,
    reference_prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run all quality checks and assign statuses.

    The hard check (spike_reversal) yields ``quarantined``; soft checks
    (entity_mismatch, extreme_returns, stale_prices) yield ``flagged``. Index symbols
    (``^GSPC`` etc.) are never quarantined — they are benchmarks, not
    cross-sectional candidates.

    Args:
        prices: Wide adjusted-close panel to scan.
        reference_prices: Optional second-vendor panel for entity checks.

    Returns:
        DataFrame with columns ``symbol``, ``check``, ``value``, ``detail``,
        ``status``, ``scanned_at`` (one row per symbol+check).
    """
    stock_columns = [c for c in prices.columns if not c.startswith("^")]
    stock_prices = prices[stock_columns]

    findings = [
        scan_extreme_returns(stock_prices),
        scan_spike_reversals(stock_prices),
        scan_stale_prices(stock_prices),
    ]
    if reference_prices is not None:
        findings.append(scan_entity_mismatch(stock_prices, reference_prices))

    all_findings = pd.concat(findings, ignore_index=True)
    if all_findings.empty:
        return pd.DataFrame(columns=["symbol", "check", "value", "detail", "status", "scanned_at"])

    all_findings["status"] = np.where(
        all_findings["check"].isin(_HARD_CHECKS), "quarantined", "flagged"
    )
    all_findings["scanned_at"] = pd.Timestamp.now().isoformat()
    return all_findings.sort_values(["status", "symbol"]).reset_index(drop=True)


def merge_with_existing(
    new_findings: pd.DataFrame, existing: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge a fresh scan with the persisted list, preserving manual decisions.

    Any symbol+check with a manual review (a ``cleared`` status or a non-empty
    ``review_note``) keeps its reviewed status and note; everything else takes
    the fresh scan's result.

    Args:
        new_findings: Output of :func:`scan_price_panel`.
        existing: Previously persisted list (or None).

    Returns:
        Merged DataFrame in the same schema.
    """
    if existing is None or existing.empty:
        return new_findings
    existing = existing.copy()
    if "review_note" not in existing.columns:
        existing["review_note"] = ""
    reviewed = existing[
        (existing["status"] == "cleared") | (existing["review_note"].fillna("") != "")
    ][["symbol", "check", "status", "review_note"]].rename(
        columns={"status": "_reviewed_status", "review_note": "_reviewed_note"}
    )
    if reviewed.empty:
        return new_findings
    merged = new_findings.merge(reviewed, on=["symbol", "check"], how="left")
    has_review = merged["_reviewed_status"].notna()
    merged["status"] = merged["status"].where(~has_review, merged["_reviewed_status"])
    merged["review_note"] = merged["_reviewed_note"].fillna("")
    return merged.drop(columns=["_reviewed_status", "_reviewed_note"])


def load_quarantine_list(path: Path = QUARANTINE_PATH) -> pd.DataFrame:
    """Load the persisted quarantine list; empty frame if none exists yet."""
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "check", "value", "detail", "status", "scanned_at"])
    return pd.read_parquet(path)


def load_quarantined_symbols(path: Path = QUARANTINE_PATH) -> set[str]:
    """
    Return the set of symbols to EXCLUDE from prices/factors at load time.

    Only ``status == "quarantined"`` rows exclude a symbol; flagged and
    cleared rows do not.

    Example:
        >>> bad = load_quarantined_symbols()  # doctest: +SKIP
        >>> clean_prices = prices.drop(columns=[s for s in bad if s in prices.columns])
    """
    quarantine = load_quarantine_list(path)
    if quarantine.empty:
        return set()
    return set(quarantine.loc[quarantine["status"] == "quarantined", "symbol"])


def set_review_status(
    symbol: str,
    check: str,
    status: str,
    note: str,
    path: Path = QUARANTINE_PATH,
) -> pd.DataFrame:
    """
    Record a manual review decision for one symbol+check finding.

    Args:
        symbol: Ticker whose finding is being reviewed.
        check: Check name (e.g. ``"spike_reversal"``, ``"entity_mismatch"``).
        status: New status: ``"cleared"`` (keep the data), ``"quarantined"``
            (exclude it), or ``"flagged"`` (undo a decision, back to review).
        note: Free-text justification; stored for the audit trail.
        path: Quarantine list location.

    Returns:
        The updated quarantine DataFrame (also persisted).

    Raises:
        ValueError: If the status is invalid or the finding does not exist.
    """
    valid_statuses = {"cleared", "quarantined", "flagged"}
    if status not in valid_statuses:
        raise ValueError(f"status must be one of {valid_statuses}, got {status!r}")

    quarantine = load_quarantine_list(path)
    if "review_note" not in quarantine.columns:
        quarantine["review_note"] = ""
    mask = (quarantine["symbol"] == symbol) & (quarantine["check"] == check)
    if not mask.any():
        raise ValueError(f"No finding for symbol={symbol!r} check={check!r} in {path}")

    quarantine.loc[mask, "status"] = status
    quarantine.loc[mask, "review_note"] = note
    write_quarantine_list(quarantine, path)
    logger.info("Review: %s/%s -> %s (%s)", symbol, check, status, note)
    return quarantine


def write_quarantine_list(quarantine: pd.DataFrame, path: Path = QUARANTINE_PATH) -> None:
    """Persist the quarantine list (parquet + a human-readable CSV alongside)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    quarantine.to_parquet(path)
    quarantine.to_csv(path.with_suffix(".csv"), index=False)
    n_quarantined = (quarantine["status"] == "quarantined").sum() if not quarantine.empty else 0
    logger.info(
        "Wrote quarantine list: %d rows (%d quarantined) to %s",
        len(quarantine),
        n_quarantined,
        path,
    )
