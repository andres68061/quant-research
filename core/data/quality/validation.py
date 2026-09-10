"""Panel invariants — the things that must never be true, checked automatically.

This exists because of a specific failure. After the ADR-0013 universe expansion,
`data/factors/prices.parquet` contained 20,642 exact-zero closes and daily
"returns" up to +101,599,900%. Nothing noticed until a research result came back
as ``-inf``, several steps downstream and hours later.

The repo already had a *quarantine scanner* (`core.data.quality.quarantine`), and it is a
different tool for a different job: it flags **suspicious symbols for human
review** — stale prices, spike-reversals, entity mismatches — and it is
judgement-laden and tuned. It also only runs when someone runs it, and after the
cutover it had not been re-run, so it was describing the old 774-symbol universe.

This module is the complement: **structural invariants that are never acceptable
under any tuning**, checked cheaply enough to run on every build and every audit.
A stock cannot trade at $0.00. A return cannot be infinite. A panel cannot have
duplicate dates. There is no threshold to argue about, so violations are errors,
not findings.

Severity contract:
- ``error``   — the artifact is unusable; a build should fail rather than publish.
- ``warning`` — usable but degraded; surfaced in the audit and the flaw registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Severity = Literal["error", "warning"]

# Above this, a daily move is a vendor defect rather than a return. Kept in sync
# with core.data.returns.DEFAULT_MAX_ABS_RETURN by the test suite.
EXTREME_RETURN_THRESHOLD = 3.0
# A panel this sparse on a given date is not a cross-section.
MIN_SYMBOLS_PER_DATE = 5


@dataclass(frozen=True)
class Violation:
    """One broken invariant."""

    check: str
    severity: Severity
    count: int
    detail: str
    examples: tuple[str, ...] = ()


def validate_price_panel(
    prices: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex | None = None,
) -> list[Violation]:
    """
    Check every structural invariant of a wide price panel.

    Args:
        prices: Wide date x symbol adjusted-close panel.
        trading_calendar: Optional reference calendar; dates outside it are
            reported (vendor bars on weekends/holidays).

    Returns:
        Violations, most severe first. Empty means the panel is structurally
        sound — which is not the same as "the data is good", only that nothing
        impossible is present.
    """
    violations: list[Violation] = []
    index = prices.index

    if index.tz is None:
        violations.append(Violation("index_tz_naive", "error", 1, "Panel index must be tz-aware"))
    if not index.is_monotonic_increasing:
        violations.append(
            Violation("index_not_sorted", "error", 1, "Panel index must be ascending")
        )
    duplicate_dates = int(index.duplicated().sum())
    if duplicate_dates:
        violations.append(
            Violation(
                "index_duplicated",
                "error",
                duplicate_dates,
                "Duplicate dates make every per-date lookup ambiguous",
            )
        )

    values = prices.to_numpy()
    observed = np.isfinite(values)

    non_positive_mask = observed & (values <= 0)
    non_positive = int(non_positive_mask.sum())
    if non_positive:
        columns = prices.columns[non_positive_mask.any(axis=0)]
        violations.append(
            Violation(
                "non_positive_price",
                "error",
                non_positive,
                "A share cannot trade at or below $0.00; pct_change across one of "
                "these is infinite, which destroys any cross-sectional mean",
                tuple(str(c) for c in columns[:10]),
            )
        )

    returns = prices.pct_change(fill_method=None).to_numpy()
    infinite = int(np.isinf(returns).sum())
    if infinite:
        violations.append(
            Violation(
                "infinite_return",
                "error",
                infinite,
                "Infinite returns propagate through every mean and cumulative sum",
            )
        )

    finite_returns = np.where(np.isfinite(returns), returns, np.nan)
    with np.errstate(invalid="ignore"):
        extreme_mask = np.abs(finite_returns) > EXTREME_RETURN_THRESHOLD
    extreme = int(np.nansum(extreme_mask))
    if extreme:
        columns = prices.columns[extreme_mask.any(axis=0)]
        worst = float(np.nanmax(np.abs(finite_returns)))
        violations.append(
            Violation(
                "extreme_return",
                "warning",
                extreme,
                f"Daily moves beyond +/-{EXTREME_RETURN_THRESHOLD:.0%} (worst "
                f"{worst:.0%}) are usually un-adjusted splits; compute returns via "
                "core.data.factors.returns so they are rejected",
                tuple(str(c) for c in columns[:10]),
            )
        )

    if trading_calendar is not None:
        outside = index.difference(trading_calendar)
        if len(outside):
            violations.append(
                Violation(
                    "non_trading_date",
                    "warning",
                    len(outside),
                    "Bars on non-trading days; returns computed across them are wrong",
                    tuple(str(d.date()) for d in outside[:5]),
                )
            )

    thin_dates = int((observed.sum(axis=1) < MIN_SYMBOLS_PER_DATE).sum())
    if thin_dates:
        violations.append(
            Violation(
                "thin_cross_section",
                "warning",
                thin_dates,
                f"Dates with fewer than {MIN_SYMBOLS_PER_DATE} priced symbols",
            )
        )

    order = {"error": 0, "warning": 1}
    return sorted(violations, key=lambda v: (order[v.severity], v.check))


def assert_panel_valid(prices: pd.DataFrame, name: str = "price panel") -> None:
    """
    Raise on any error-severity violation; log warnings.

    Call this at the end of a panel build. Publishing an artifact that violates a
    structural invariant is strictly worse than failing the build: the artifact
    looks complete, gets consumed, and the error surfaces days later inside a
    research result.

    Args:
        prices: Panel to check.
        name: Label used in messages.

    Raises:
        ValueError: If any error-severity invariant is violated.
    """
    violations = validate_price_panel(prices)
    for violation in violations:
        logger.log(
            logging.ERROR if violation.severity == "error" else logging.WARNING,
            "%s: %s (%d) — %s%s",
            name,
            violation.check,
            violation.count,
            violation.detail,
            f" e.g. {list(violation.examples)}" if violation.examples else "",
        )
    errors = [v for v in violations if v.severity == "error"]
    if errors:
        raise ValueError(
            f"{name} failed {len(errors)} structural check(s): "
            + ", ".join(f"{v.check} x{v.count}" for v in errors)
        )


def violations_as_dicts(violations: list[Violation]) -> list[dict[str, object]]:
    """Serialize violations for the audit snapshot and the API."""
    return [
        {
            "check": v.check,
            "severity": v.severity,
            "count": v.count,
            "detail": v.detail,
            "examples": list(v.examples),
        }
        for v in violations
    ]
