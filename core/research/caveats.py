"""THE registry of research caveats. One file, every disclosure, every surface.

Before this module, caveats lived in four places: data flaws in
``core/data/quality/health.py``, sector-index caveats in ``core/strategies/sector_index.py``,
PEAD caveats inline in an API route, and screen caveats inline in a script. Each
was correct and none knew about the others, so no surface could answer "what
should a user of THIS number know?" — which is the only question that matters.

Now: every caveat is declared once here with the **surfaces** it applies to, and
every page/endpoint/report asks :func:`caveats_for_surface`. A caveat that is not
in this registry does not get shown anywhere, and a surface that renders numbers
without querying it is a bug the ``research-disclosure`` skill exists to catch.

Two kinds, deliberately in the same registry because a reader does not care which
is which:

- ``DATA`` — a property of what we hold (coverage gaps, vendor defects, bias).
  Measured counterparts are produced live by ``core.data.quality.health`` and merged in.
- ``METHOD`` — a consequence of how we compute (weighting, rebalance cadence,
  cost assumptions, statistical treatment).

Severity: ``high`` can silently flip a conclusion; ``medium`` biases a known
subset; ``low`` is operational.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Kind = Literal["data", "method"]
Severity = Literal["high", "medium", "low"]

# Canonical surface identifiers. A surface is anything that shows a number to a
# human: a page, an endpoint, a generated report.
SURFACE_SECTOR_PERFORMANCE = "sector_performance"
SURFACE_PEAD = "pead_study"
SURFACE_FACTOR_SCREEN = "factor_screen"
SURFACE_FACTOR_BACKTEST = "factor_backtest"
SURFACE_DATA_HEALTH = "data_health"
SURFACE_UNIVERSE = "universe"

ALL_SURFACES: tuple[str, ...] = (
    SURFACE_SECTOR_PERFORMANCE,
    SURFACE_PEAD,
    SURFACE_FACTOR_SCREEN,
    SURFACE_FACTOR_BACKTEST,
    SURFACE_DATA_HEALTH,
    SURFACE_UNIVERSE,
)


@dataclass(frozen=True)
class Caveat:
    """
    One disclosure: what a reader of a number must know before trusting it.

    Attributes:
        id: Stable slug, referenced by docs and tests.
        kind: ``data`` (what we hold) or ``method`` (how we compute).
        severity: ``high`` / ``medium`` / ``low``.
        title: One line, readable standalone.
        detail: Full explanation, including WHY it exists and what to do about it.
        surfaces: Where this must be displayed.
        remediation: What would remove it, or None when it is inherent.
    """

    id: str
    kind: Kind
    severity: Severity
    title: str
    detail: str
    surfaces: tuple[str, ...] = field(default_factory=tuple)
    remediation: str | None = None


CAVEAT_REGISTRY: tuple[Caveat, ...] = (
    # ---------------- Method: universe & membership ----------------
    Caveat(
        id="membership-is-point-in-time",
        kind="method",
        severity="low",
        title="Index membership is applied point-in-time, not retroactively",
        detail=(
            "A stock counts as an S&P 500 member only on dates it actually was one, "
            "using the historical membership table (data/universe/index_membership.parquet). "
            "This is the correct treatment and is called out because the common error — "
            "applying today's member list to all history — inflates returns by holding "
            "past winners early."
        ),
        surfaces=(SURFACE_SECTOR_PERFORMANCE, SURFACE_FACTOR_BACKTEST, SURFACE_UNIVERSE),
    ),
    Caveat(
        id="sector-labels-current-only",
        kind="data",
        severity="medium",
        title="Sector labels are today's, applied to all history",
        detail=(
            "The vendor exposes only current sector/industry classifications, so a company "
            "reclassified in 2020 carries that label back to 1990. This is mild lookahead: "
            "a stock that became 'Technology' after a pivot appears in Technology for its "
            "entire pre-pivot history. It affects sector aggregates and sector-neutral "
            "factors (value_quality_sn, roe_sn)."
        ),
        surfaces=(SURFACE_SECTOR_PERFORMANCE, SURFACE_FACTOR_BACKTEST, SURFACE_DATA_HEALTH),
        remediation="Source a point-in-time GICS history (licensed) or reconstruct from filings.",
    ),
    Caveat(
        id="screener-cap-floor-today",
        kind="data",
        severity="medium",
        title="The universe's $300M floor is measured at TODAY's market cap",
        detail=(
            "The download universe was screened on current market cap, so a company worth "
            "$200M today but $2B in 2015 is absent, while one worth $350M today but $50M in "
            "2015 is present for its whole history. The universe table decides what we "
            "DOWNLOAD; point-in-time tradability must still be enforced at backtest time "
            "using the market-cap panel."
        ),
        surfaces=(SURFACE_UNIVERSE, SURFACE_FACTOR_BACKTEST, SURFACE_DATA_HEALTH),
    ),
    Caveat(
        id="non-operating-vehicles-in-universe",
        kind="data",
        severity="high",
        title="The universe contains shell companies and SPACs unless filtered",
        detail=(
            "About 20% of the expanded universe (1,779 symbols) are pre-merger SPACs or "
            "blank-check shells — trust accounts with tickers that sit near $10.00 at "
            "near-zero volatility until they merge or liquidate. They have no operations, so "
            "every fundamental factor is meaningless for them, and their artificial "
            "low-volatility profile puts them in the extreme tiers of vol- and "
            "liquidity-sensitive factors. Any cross-sectional result on the full universe "
            "must state whether they were excluded."
        ),
        surfaces=(
            SURFACE_UNIVERSE,
            SURFACE_FACTOR_SCREEN,
            SURFACE_FACTOR_BACKTEST,
            SURFACE_SECTOR_PERFORMANCE,
            SURFACE_DATA_HEALTH,
        ),
        remediation=(
            "core.data.universe.filters.build_universe_filter(exclude_non_operating=True)"
        ),
    ),
    Caveat(
        id="vendor-bad-prints-extreme-returns",
        kind="data",
        severity="high",
        title="The panel contains daily 'returns' up to +100,000,000% from vendor defects",
        detail=(
            "Measured 2026-08-13 on the full canonical panel: 7,691 daily returns above "
            "+100%, 3,142 above +500%, maximum +101,599,900% — almost all un-adjusted "
            "reverse splits and quote errors on delisted micro-caps, plus 20,642 exact-zero "
            "closes that make pct_change infinite. These are not noisy returns, they are "
            "destructive: one infinite value in a cross-sectional mean turns EVERY symbol's "
            "abnormal return that day into infinity, and it propagates through any cumulative "
            "sum. The first expanded-universe PEAD run returned -inf paths for exactly this "
            "reason. Compute returns through core.data.factors.returns, never bare pct_change."
        ),
        surfaces=(
            SURFACE_UNIVERSE,
            SURFACE_FACTOR_SCREEN,
            SURFACE_FACTOR_BACKTEST,
            SURFACE_PEAD,
            SURFACE_DATA_HEALTH,
        ),
        remediation=(
            "core.data.factors.returns.compute_clean_returns rejects |return| > 300% as bad prints "
            "and supports a min_price floor; the panel builder nulls non-positive closes."
        ),
    ),
    Caveat(
        id="sub-dollar-bid-ask-bounce",
        kind="method",
        severity="medium",
        title="Sub-dollar stocks' returns are mostly bid-ask bounce",
        detail=(
            "A one-cent spread on a $0.05 stock is a 20% round trip, so penny-stock return "
            "series are dominated by quote mechanics rather than information. The panel holds "
            "40,009 sub-penny prices across 358 symbols. Cross-sectional research "
            "conventionally applies a $1 or $5 price floor; where one is applied it is stated "
            "in the surface's methodology, and where it is not, extreme-tail results should be "
            "assumed to be bounce."
        ),
        surfaces=(SURFACE_UNIVERSE, SURFACE_FACTOR_SCREEN, SURFACE_PEAD, SURFACE_DATA_HEALTH),
        remediation="Pass min_price to core.data.factors.returns / the event study.",
    ),
    # ---------------- Method: sector index ----------------
    Caveat(
        id="sector-index-daily-compounding",
        kind="method",
        severity="low",
        title="Sector indices compound daily returns — there is no discrete rebalance",
        detail=(
            "Weights are recomputed EVERY trading day from the prior day's market caps "
            "(cap-weighted) or set equal across that day's members (equal-weighted). There "
            "is no monthly/quarterly reconstitution, so no rebalance turnover or cost is "
            "modeled. Equal-weighted in particular implies a daily rebalance, which earns a "
            "real but untradeable rebalancing premium versus a buy-and-hold equal book."
        ),
        surfaces=(SURFACE_SECTOR_PERFORMANCE,),
    ),
    Caveat(
        id="sector-index-gross-returns",
        kind="method",
        severity="medium",
        title="Sector indices are gross: no costs, no taxes, no reconstitution drag",
        detail=(
            "Levels are compounded from dividend-adjusted closes with zero transaction "
            "costs. Real sector ETFs lag these levels by fees plus turnover cost; treat the "
            "series as a research benchmark, not an achievable return."
        ),
        surfaces=(SURFACE_SECTOR_PERFORMANCE,),
    ),
    Caveat(
        id="delisting-exit-at-last-price",
        kind="method",
        severity="medium",
        title="Delisted stocks exit at their last traded price",
        detail=(
            "A name that stops trading leaves the index at its final close. Any terminal "
            "loss beyond that day (bankruptcy wipeout, sub-penny OTC drift) is not charged. "
            "This biases index levels UP relative to a real holder's experience, and the "
            "bias is concentrated in crisis periods when delistings cluster."
        ),
        surfaces=(SURFACE_SECTOR_PERFORMANCE, SURFACE_FACTOR_BACKTEST),
    ),
    Caveat(
        id="thin-membership-holds-flat",
        kind="method",
        severity="low",
        title="Sector-days with too few members hold the index flat",
        detail=(
            "When a sector has fewer than the minimum member count on a date (early history, "
            "or with the S&P filter on), that day's return is set to zero rather than trusting "
            "a two-stock 'sector'. The level therefore flatlines rather than showing noise; "
            "member counts are published alongside the levels so this is visible."
        ),
        surfaces=(SURFACE_SECTOR_PERFORMANCE,),
    ),
    # ---------------- Method: event study / PEAD ----------------
    Caveat(
        id="pead-excludes-day-zero",
        kind="method",
        severity="low",
        title="The announcement-day jump is excluded from the drift",
        detail=(
            "The measured path starts at day +1. The day-0 move is not capturable by a "
            "strategy that learns the surprise from the announcement itself, so including it "
            "would report a return nobody could have earned."
        ),
        surfaces=(SURFACE_PEAD,),
    ),
    Caveat(
        id="pead-abnormal-vs-equal-weight-mean",
        kind="method",
        severity="medium",
        title="Abnormal return is measured against a daily equal-weight universe mean",
        detail=(
            "The benchmark is the cross-sectional mean return of all stocks trading that day "
            "— itself a daily-rebalanced portfolio that earns a rebalancing premium. That "
            "pushes every quantile's level down uniformly. The top-minus-bottom SPREAD nets "
            "it out and is the statistic to read; absolute quantile levels are not."
        ),
        surfaces=(SURFACE_PEAD,),
        remediation="Beta-adjust against a market factor instead of the cross-sectional mean.",
    ),
    Caveat(
        id="pead-large-cap-universe",
        kind="data",
        severity="high",
        title="PEAD is measured on large caps, where the literature does NOT expect it",
        detail=(
            "The published effect concentrates in small caps and names with thin analyst "
            "coverage. Measuring it on S&P-scale companies and finding nothing is consistent "
            "with the literature, NOT evidence the anomaly is gone. A null here should be "
            "read as 'untested where it lives'."
        ),
        surfaces=(SURFACE_PEAD,),
        remediation="Rebuild the surprise panel on the expanded universe and re-run.",
    ),
    # ---------------- Method: backtests & screens ----------------
    Caveat(
        id="flat-transaction-costs",
        kind="method",
        severity="high",
        title="A single flat cost is applied regardless of liquidity",
        detail=(
            "Backtests charge one basis-point figure per trade for every name. That flatters "
            "any strategy that trades illiquid stocks — most of all the liquidity factors, "
            "which deliberately go long the hardest-to-trade names. Use the dollar-ADV cost "
            "schedule in core/data/factors/liquidity.py before believing a liquidity result."
        ),
        surfaces=(SURFACE_FACTOR_SCREEN, SURFACE_FACTOR_BACKTEST),
        remediation="Pass dollar_adv to the runner so per-name costs scale with liquidity.",
    ),
    Caveat(
        id="short-leg-gross-of-borrow",
        kind="method",
        severity="high",
        title="Long/short results are gross of borrow cost and short availability",
        detail=(
            "The short leg assumes every name can be shorted at zero cost. In reality small, "
            "distressed, and heavily-shorted names are expensive or impossible to borrow — "
            "and those are exactly the names most anomaly short legs select."
        ),
        surfaces=(SURFACE_FACTOR_SCREEN, SURFACE_FACTOR_BACKTEST),
    ),
    Caveat(
        id="multiple-testing-correction",
        kind="method",
        severity="high",
        title="Testing many factors makes the best one look good by luck",
        detail=(
            "Across N tested factors the maximum Sharpe is inflated by selection. Results are "
            "reported against a Sidak-corrected threshold at family-wise alpha 0.05 (ADR 0003) "
            "and the FULL cross-section is published — reading only the winner reintroduces "
            "exactly the bias the correction removes."
        ),
        surfaces=(SURFACE_FACTOR_SCREEN,),
    ),
    Caveat(
        id="in-sample-period-single-universe",
        kind="method",
        severity="medium",
        title="One period, one universe, one parameterisation",
        detail=(
            "Every factor is run with identical untuned parameters over a single window. That "
            "avoids per-factor overfitting but means a result is evidence about THIS universe, "
            "period, and cost model — not a general claim about the factor."
        ),
        surfaces=(SURFACE_FACTOR_SCREEN,),
    ),
    Caveat(
        id="placeholder-filing-dates-pre-2000",
        kind="data",
        severity="high",
        title="Pre-2000 fundamentals use an imputed 45-day filing lag",
        detail=(
            "The vendor fills acceptedDate with the period end for pre-EDGAR filings (100% of "
            "1980s rows, ~50% of 1990s). Those rows now carry a conservative 45-day lag and a "
            "publication_date_imputed flag. Any result leaning on pre-2000 history must be "
            "re-run with imputed rows excluded before it is believed (ADR 0012)."
        ),
        surfaces=(SURFACE_FACTOR_SCREEN, SURFACE_FACTOR_BACKTEST, SURFACE_DATA_HEALTH),
    ),
)

_BY_ID = {caveat.id: caveat for caveat in CAVEAT_REGISTRY}

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def caveats_for_surface(surface: str) -> list[Caveat]:
    """
    Every caveat that must be displayed on one surface, most severe first.

    Args:
        surface: One of :data:`ALL_SURFACES`.

    Returns:
        Matching caveats sorted by severity.

    Raises:
        KeyError: For an unknown surface — a typo would silently disclose nothing,
            which is the failure mode this registry exists to prevent.
    """
    if surface not in ALL_SURFACES:
        raise KeyError(f"Unknown surface {surface!r}; known: {ALL_SURFACES}")
    matching = [c for c in CAVEAT_REGISTRY if surface in c.surfaces]
    return sorted(matching, key=lambda c: (_SEVERITY_ORDER[c.severity], c.id))


def caveat_by_id(caveat_id: str) -> Caveat:
    """Look up one caveat; raises KeyError when the id is unknown."""
    return _BY_ID[caveat_id]


def as_dicts(caveats: list[Caveat]) -> list[dict[str, str | None]]:
    """Serialize caveats for API responses and report generators."""
    return [
        {
            "id": c.id,
            "kind": c.kind,
            "severity": c.severity,
            "title": c.title,
            "detail": c.detail,
            "remediation": c.remediation,
        }
        for c in caveats
    ]
