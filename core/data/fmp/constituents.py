"""FMP S&P 500 constituent change events → point-in-time membership snapshots.

``/historical-sp500-constituent`` returns add/remove events, not daily
membership. We reconstruct snapshots by walking events reverse-chronologically
from the current ``/sp500-constituent`` roster.

The derived snapshot table matches the hanshof CSV schema (``date``, ``tickers``
list) so ``SP500Constituents`` can load either source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import pandas as pd

from core.data.fmp.client import fmp_get
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)


def normalize_equity_ticker(symbol: str) -> str:
    """
    Canonical equity ticker for membership set compares.

    FMP often uses ``BRK-B`` / ``BF-B`` while the historical CSV uses ``BRK.B`` /
    ``BF.B``. Uppercase and map ``.`` → ``-`` so Jaccard is not dominated by
    share-class punctuation. Does not invent rename chains (``FB``≠``META``).
    """
    return str(symbol).strip().upper().replace(".", "-")


def normalize_ticker_set(tickers: Iterable[str]) -> set[str]:
    """Apply :func:`normalize_equity_ticker` to every non-empty symbol."""
    return {normalize_equity_ticker(t) for t in tickers if t}


@dataclass
class MembershipReconciliation:
    """Diff between FMP-derived membership and a reference CSV snapshot."""

    n_fmp_dates: int
    n_csv_dates: int
    shared_dates: int
    mean_jaccard: float
    min_jaccard: float
    worst_date: Optional[str] = None
    sample_only_in_fmp: list[str] = field(default_factory=list)
    sample_only_in_csv: list[str] = field(default_factory=list)
    notation_normalized: bool = True

    def summary(self) -> str:
        notation = "notation-normalized" if self.notation_normalized else "raw-tickers"
        return (
            f"dates fmp={self.n_fmp_dates} csv={self.n_csv_dates} shared={self.shared_dates}; "
            f"jaccard mean={self.mean_jaccard:.3f} min={self.min_jaccard:.3f} "
            f"worst={self.worst_date} ({notation}); "
            f"only_fmp≈{self.sample_only_in_fmp[:5]} only_csv≈{self.sample_only_in_csv[:5]}"
        )


def fetch_current_sp500(api_key: Optional[str] = None) -> set[str]:
    """Current S&P 500 symbols from ``/sp500-constituent``."""
    rows = fmp_get("sp500-constituent", api_key=api_key)
    if not isinstance(rows, list) or not rows:
        raise DataSchemaError(f"Unexpected current SP500 payload: {type(rows)}")
    return {str(r["symbol"]).strip() for r in rows if r.get("symbol")}


def fetch_sp500_change_events(api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch all historical S&P 500 add/remove events.

    Returns:
        DataFrame sorted by ``date`` ascending with columns
        ``date``, ``added_symbol``, ``removed_symbol``, ``reason``.
    """
    rows = fmp_get("historical-sp500-constituent", api_key=api_key)
    if not isinstance(rows, list) or not rows:
        raise DataSchemaError(f"Unexpected historical SP500 payload: {type(rows)}")
    events = pd.DataFrame(
        {
            "date": pd.to_datetime([r["date"] for r in rows]),
            "added_symbol": [str(r.get("symbol") or "").strip() for r in rows],
            "removed_symbol": [str(r.get("removedTicker") or "").strip() for r in rows],
            "reason": [r.get("reason") or "" for r in rows],
        }
    )
    return events.sort_values("date").reset_index(drop=True)


def build_membership_snapshots(
    change_events: pd.DataFrame,
    current_members: set[str],
) -> pd.DataFrame:
    """
    Reconstruct point-in-time membership from change events + today's roster.

    Walks events newest-first, undoing each add/remove to recover the set that
    was in force *before* that event. Emits one snapshot per distinct event
    date (membership after all events on that date).

    Args:
        change_events: Output of :func:`fetch_sp500_change_events`.
        current_members: Today's S&P 500 symbols.

    Returns:
        DataFrame indexed by ``date`` with a ``tickers`` column (list[str],
        sorted) — same shape the hanshof CSV loader expects after parse.
    """
    members = set(current_members)
    # Group events by date (newest first) so same-day adds/removes apply together.
    by_date = change_events.sort_values("date", ascending=False).groupby("date", sort=False)
    snapshots: list[dict[str, Any]] = []
    # Membership after the most recent event = current roster.
    for event_date, group in by_date:
        snapshots.append({"date": event_date, "tickers": sorted(members)})
        # Undo this date's events to recover the prior roster.
        for row in group.itertuples():
            if row.added_symbol:
                members.discard(row.added_symbol)
            if row.removed_symbol:
                members.add(row.removed_symbol)

    # One more snapshot for the earliest pre-history (after undoing everything).
    if change_events["date"].notna().any():
        earliest = change_events["date"].min() - pd.Timedelta(days=1)
        snapshots.append({"date": earliest, "tickers": sorted(members)})

    frame = pd.DataFrame(snapshots).drop_duplicates(subset=["date"], keep="first")
    return frame.set_index("date").sort_index()


def reconcile_membership(
    fmp_snapshots: pd.DataFrame,
    csv_snapshots: pd.DataFrame,
    sample_dates: int = 40,
    *,
    normalize_notation: bool = True,
) -> MembershipReconciliation:
    """
    Compare FMP-derived membership to a reference CSV on overlapping dates.

    Jaccard similarity is computed on up to ``sample_dates`` evenly spaced
    FMP dates, each matched to the last CSV snapshot on or before that date.
    When ``normalize_notation`` is True (default), tickers are canonicalized
    with :func:`normalize_equity_ticker` before set ops.
    """
    fmp_dates = fmp_snapshots.index.sort_values()
    csv_dates = csv_snapshots.index.sort_values()
    jaccards: list[float] = []
    worst_date = None
    worst_j = 2.0
    only_fmp_sample: list[str] = []
    only_csv_sample: list[str] = []

    step = max(len(fmp_dates) // sample_dates, 1)
    sampled = fmp_dates[::step][:sample_dates]
    shared = 0
    for date in sampled:
        csv_eligible = csv_dates[csv_dates <= date]
        if csv_eligible.empty:
            continue
        shared += 1
        raw_fmp = set(fmp_snapshots.loc[date, "tickers"])
        raw_csv = set(csv_snapshots.loc[csv_eligible[-1], "tickers"])
        if normalize_notation:
            fmp_set = normalize_ticker_set(raw_fmp)
            csv_set = normalize_ticker_set(raw_csv)
        else:
            fmp_set = raw_fmp
            csv_set = raw_csv
        union = fmp_set | csv_set
        jaccard = len(fmp_set & csv_set) / len(union) if union else 1.0
        jaccards.append(jaccard)
        if jaccard < worst_j:
            worst_j = jaccard
            worst_date = str(pd.Timestamp(date).date())
            only_fmp_sample = sorted(fmp_set - csv_set)[:8]
            only_csv_sample = sorted(csv_set - fmp_set)[:8]

    return MembershipReconciliation(
        n_fmp_dates=len(fmp_dates),
        n_csv_dates=len(csv_dates),
        shared_dates=shared,
        mean_jaccard=float(sum(jaccards) / len(jaccards)) if jaccards else 0.0,
        min_jaccard=float(min(jaccards)) if jaccards else 0.0,
        worst_date=worst_date,
        sample_only_in_fmp=only_fmp_sample,
        sample_only_in_csv=only_csv_sample,
        notation_normalized=normalize_notation,
    )
