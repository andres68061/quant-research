"""One glossary for the whole platform.

The data-hygiene glossary previously lived inside ``core.data.health`` and was
rendered only on the data-health page. That worked until a second page needed to
define "Sharpe ratio" — at which point the natural move is a second glossary, and
two glossaries drift within a month.

So: one registry, categorised, surfaced anywhere a term appears. A definition
lives here and nowhere else. Pages link to it rather than restating it.

**House style for entries.** Each definition must stand alone — a reader who
knows nothing lands on the term and leaves understanding it. That means:

- say what it *is* before what it is *for*,
- give the number's scale when there is one ("above 2 is the usual bar"),
- name the trap, if the term has one people fall into.

Same rule the ``explain-in-context`` skill applies to prose, applied to reference
material.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CATEGORIES",
    "GLOSSARY",
    "GlossaryEntry",
    "as_dicts",
    "lookup",
    "terms_in_category",
]

Category = Literal["statistics", "strategy", "metrics", "data", "costs"]

CATEGORIES: tuple[tuple[Category, str], ...] = (
    ("statistics", "Statistics and significance"),
    ("strategy", "Strategy construction"),
    ("metrics", "Performance metrics"),
    ("costs", "Costs and liquidity"),
    ("data", "Data and universe"),
)


@dataclass(frozen=True)
class GlossaryEntry:
    """One term, defined so it needs no surrounding context."""

    term: str
    category: Category
    definition: str
    see_also: tuple[str, ...] = ()


GLOSSARY: tuple[GlossaryEntry, ...] = (
    # ---------------------------------------------------------- statistics
    GlossaryEntry(
        "t-statistic",
        "statistics",
        "How many standard errors an estimated average sits away from zero. It "
        "answers 'could this result be luck?' — bigger means less likely. As a "
        "rule of thumb |t| >= 2 corresponds to about a 5% chance of seeing a "
        "result this strong when the true effect is zero. That bar is for ONE "
        "test; testing many things at once requires a higher bar.",
        ("significance bar", "Sidak correction", "p-value"),
    ),
    GlossaryEntry(
        "p-value",
        "statistics",
        "The probability of seeing a result at least this extreme if there were "
        "genuinely no effect. Small means the data is surprising under 'nothing "
        "here'. It is NOT the probability that the strategy works — a common and "
        "expensive misreading.",
        ("t-statistic",),
    ),
    GlossaryEntry(
        "significance bar",
        "statistics",
        "The |t| a result must clear before we call it real. This platform does "
        "not use the textbook 2.0, because we test many factors at once and the "
        "best of many looks impressive by chance. The bar is raised by the Sidak "
        "correction in proportion to how many things were tried — for the 28-factor "
        "screen it is |t| >= 3.12.",
        ("Sidak correction", "multiple testing", "t-statistic"),
    ),
    GlossaryEntry(
        "multiple testing",
        "statistics",
        "The problem that testing many hypotheses produces winners by luck alone. "
        "Flip 28 coins 20 times each and the luckiest coin looks special. Any "
        "screen over many factors has this problem, so the significance bar must "
        "rise with the number of tests.",
        ("Sidak correction", "significance bar"),
    ),
    GlossaryEntry(
        "Sidak correction",
        "statistics",
        "A way of raising the significance bar for the number of tests run. To "
        "keep the chance of ANY false positive across the whole family at 5%, "
        "each individual test is judged at alpha = 1 - (1 - 0.05)^(1/N) for N "
        "tests. With N = 28 that is |t| >= 3.12 rather than 2.0. Slightly less "
        "conservative than Bonferroni, and exact when tests are independent.",
        ("significance bar", "multiple testing"),
    ),
    GlossaryEntry(
        "walk-forward validation",
        "statistics",
        "Re-running a strategy test at multiple points in time rather than once "
        "over all history, so a result is not an artifact of one lucky window. "
        "Train on the past, test on the period immediately after, roll forward, "
        "repeat. Default cadence here is once per year.",
        ("out-of-sample", "overfitting"),
    ),
    GlossaryEntry(
        "overfitting",
        "statistics",
        "Tuning a strategy until it fits the quirks of the history it was built "
        "on rather than any repeatable effect. The tell is a result that is "
        "excellent in-sample and ordinary out-of-sample.",
        ("walk-forward validation", "multiple testing"),
    ),
    GlossaryEntry(
        "control variant",
        "statistics",
        "A deliberately signal-free version of a strategy, run identically, to "
        "measure what the signal actually adds. Example: a long-only factor book "
        "scored Sharpe 0.85 — but holding every stock with no signal at all "
        "scored 0.67, so the signal contributed 0.18, not 0.85. Without the "
        "control, market exposure is mistaken for skill.",
        ("Sharpe ratio", "long-only", "market beta"),
    ),
    # ---------------------------------------------------------- strategy
    GlossaryEntry(
        "cross-sectional strategy",
        "strategy",
        "A strategy that ranks all stocks against each other on some measure at a "
        "point in time, then buys the top slice and (usually) sells the bottom "
        "slice. It bets on relative performance, not on the market's direction.",
        ("long/short", "quantile", "rebalancing"),
    ),
    GlossaryEntry(
        "long/short",
        "strategy",
        "Holding the ranked winners and short-selling the ranked losers in equal "
        "size. Because the two legs largely cancel market movement, what remains "
        "is closer to the signal's own contribution — which is why long/short is "
        "the honest way to test whether a factor works.",
        ("long-only", "market beta", "cross-sectional strategy"),
    ),
    GlossaryEntry(
        "long-only",
        "strategy",
        "Holding only the top slice, with no short leg. Easier and cheaper to "
        "trade, but its return is dominated by simply being in the market, so a "
        "good-looking long-only Sharpe is not evidence that a signal works until "
        "compared against a no-signal control.",
        ("long/short", "market beta", "control variant"),
    ),
    GlossaryEntry(
        "market beta",
        "strategy",
        "How much a position moves with the overall market. A long-only stock "
        "book has beta near 1, so most of its return and risk is the market's, "
        "not the strategy's. Comparing a long-only variant to a long/short "
        "baseline compares two different things.",
        ("long-only", "long/short", "control variant"),
    ),
    GlossaryEntry(
        "quantile",
        "strategy",
        "An equal-sized slice of the ranked universe. Quintiles are fifths (top "
        "20%, bottom 20%); deciles are tenths. Narrower slices give a purer "
        "signal and fewer, more concentrated positions.",
        ("cross-sectional strategy",),
    ),
    GlossaryEntry(
        "rebalancing",
        "strategy",
        "Re-ranking and resetting positions on a schedule — monthly here by "
        "default. More frequent rebalancing tracks the signal more closely and "
        "costs more in trading; less frequent does the reverse.",
        ("turnover", "transaction costs"),
    ),
    GlossaryEntry(
        "signal lag",
        "strategy",
        "Trading days between observing a factor value and acting on it. One day "
        "by default: you cannot trade on a close you have not seen yet. Setting "
        "it to zero is one of the most common ways to invent returns that never "
        "existed.",
        ("lookahead bias", "point-in-time"),
    ),
    GlossaryEntry(
        "PEAD",
        "strategy",
        "Post-earnings announcement drift: after a company reports better than "
        "expected, its price jumps and then keeps drifting the same way for "
        "weeks, because the market under-reacts. One of the most durable "
        "documented anomalies — measured here at t = 10.05 across 267,780 "
        "announcements, though not tradable after realistic costs.",
        ("SUE", "overlapping portfolio", "transaction costs"),
    ),
    GlossaryEntry(
        "SUE",
        "strategy",
        "Standardised unexpected earnings: the earnings surprise divided by how "
        "variable that company's surprises usually are. Standardising makes a "
        "$0.02 beat at a steady utility comparable to a $0.02 beat at an erratic "
        "biotech.",
        ("PEAD",),
    ),
    GlossaryEntry(
        "overlapping portfolio",
        "strategy",
        "The construction that turns an event study into a tradable book. Each "
        "event opens a position held for a fixed window, so on any given day you "
        "hold many cohorts started on different days. With a 60-day hold, about "
        "1/60th of the book turns over daily. Standard since Jegadeesh-Titman "
        "(1993).",
        ("PEAD", "turnover"),
    ),
    GlossaryEntry(
        "net operating assets",
        "strategy",
        "The operating side of the balance sheet — receivables, inventory, plant "
        "— minus operating liabilities, scaled by prior-year assets. It is a "
        "running total of the gap between reported profits and actual cash. When "
        "it balloons, past earnings were accounting entries rather than cash, and "
        "returns tend to disappoint as it unwinds (Hirshleifer, Hou, Teoh & "
        "Zhang, 2004).",
        ("accruals",),
    ),
    GlossaryEntry(
        "accruals",
        "strategy",
        "The part of reported earnings not backed by cash this period. High "
        "accruals predict weaker future returns because they tend to reverse "
        "(Sloan, 1996). Net operating assets is the balance-sheet accumulation of "
        "the same idea.",
        ("net operating assets",),
    ),
    GlossaryEntry(
        "Piotroski F-score",
        "strategy",
        "A 9-point checklist of accounting health: is the company profitable, is "
        "profit backed by cash, is debt falling, are margins improving. Each test "
        "scores 0 or 1. Piotroski's 2000 paper applied it ONLY to cheap "
        "(high book-to-market) stocks — among bargains, it separates recovering "
        "companies from dying ones. Applying it to the whole market is not the "
        "paper's claim.",
        ("value factor",),
    ),
    GlossaryEntry(
        "value factor",
        "strategy",
        "Ranking stocks by how cheap they are relative to a fundamental anchor "
        "such as book value or earnings. The oldest documented equity anomaly, "
        "and one whose edge has weakened substantially since publication.",
        ("Piotroski F-score",),
    ),
    # ---------------------------------------------------------- metrics
    GlossaryEntry(
        "Sharpe ratio",
        "metrics",
        "Average excess return divided by its volatility, annualised. It answers "
        "'how much return per unit of risk?'. Roughly: below 0.5 is weak, around "
        "1.0 is good, above 2.0 in a backtest usually means a bug or overfitting. "
        "Its blind spot is shape — two strategies with identical Sharpe can win "
        "often-and-small or rarely-and-huge, which are very different to hold.",
        ("Sortino ratio", "hit rate", "profit factor"),
    ),
    GlossaryEntry(
        "Sortino ratio",
        "metrics",
        "Like the Sharpe ratio, but dividing by downside volatility only. It "
        "stops punishing a strategy for large gains, which plain volatility does.",
        ("Sharpe ratio",),
    ),
    GlossaryEntry(
        "hit rate",
        "metrics",
        "The fraction of periods with a positive return. A genuine "
        "cross-sectional factor typically sits near 51-53% daily — many tiny "
        "edges, not a few big calls. A very high hit rate with a mediocre Sharpe "
        "means the losses are rare but severe.",
        ("win/loss ratio", "profit factor", "Sharpe ratio"),
    ),
    GlossaryEntry(
        "win/loss ratio",
        "metrics",
        "Average size of a winning period divided by the average size of a losing "
        "one. Read together with hit rate: winning 40% of the time is fine if the "
        "wins are twice the size of the losses.",
        ("hit rate", "profit factor"),
    ),
    GlossaryEntry(
        "profit factor",
        "metrics",
        "Total gains divided by total losses. Above 1.0 means the strategy made "
        "money; 1.5 means gains were half again the losses. Unlike Sharpe it is "
        "not annualised and not risk-adjusted, so it complements rather than "
        "replaces it.",
        ("hit rate", "Sharpe ratio"),
    ),
    GlossaryEntry(
        "maximum drawdown",
        "metrics",
        "The largest peak-to-trough fall in cumulative value. It is the question "
        "'how bad did it get before it recovered?', and it is usually what "
        "decides whether a strategy is actually holdable.",
        ("Calmar ratio",),
    ),
    GlossaryEntry(
        "Calmar ratio",
        "metrics",
        "Annualised return divided by maximum drawdown — return per unit of worst "
        "pain, rather than per unit of volatility.",
        ("maximum drawdown", "Sharpe ratio"),
    ),
    GlossaryEntry(
        "gross vs net returns",
        "metrics",
        "Gross is before trading costs, net is after. The gap is often the whole "
        "result: the PEAD book here goes from Sharpe 1.11 gross to -0.11 net once "
        "costs scale with each stock's liquidity. Any return quoted without "
        "saying which one it is should be treated as gross.",
        ("transaction costs", "dollar ADV"),
    ),
    # ---------------------------------------------------------- costs
    GlossaryEntry(
        "transaction costs",
        "costs",
        "What trading takes out: commissions, the bid-ask spread, and price "
        "impact from your own order. Quoted in basis points of the value traded. "
        "A flat assumption (say 10 bps for everything) is convenient and "
        "flattering, because real costs are far higher in small illiquid names.",
        ("basis point", "dollar ADV", "gross vs net returns"),
    ),
    GlossaryEntry(
        "basis point",
        "costs",
        "One hundredth of a percentage point. 10 bps = 0.10%. Costs and spreads "
        "are quoted in bps because the numbers involved are small.",
        ("transaction costs",),
    ),
    GlossaryEntry(
        "dollar ADV",
        "costs",
        "Average daily dollar volume — price times shares traded, averaged over a "
        "window (21 days here). The standard proxy for liquidity: trading a size "
        "that is a large fraction of ADV moves the price against you. Costs in "
        "this platform scale by ADV bucket rather than being flat.",
        ("transaction costs", "liquidity"),
    ),
    GlossaryEntry(
        "liquidity",
        "costs",
        "How much can be traded without moving the price. Low-liquidity stocks "
        "are where documented anomalies are usually strongest and where they are "
        "usually impossible to harvest, because the cost of trading them eats the "
        "edge.",
        ("dollar ADV", "transaction costs"),
    ),
    GlossaryEntry(
        "turnover",
        "costs",
        "The fraction of the portfolio replaced per period. It is the multiplier "
        "on transaction costs: doubling turnover doubles the cost drag.",
        ("rebalancing", "transaction costs"),
    ),
    # ---------------------------------------------------------- data
    GlossaryEntry(
        "survivorship bias",
        "data",
        "Measuring only companies that survived to today. It inflates returns "
        "because the failures are invisible. Countered here by downloading "
        "delisted companies and letting them contribute until their last traded "
        "day.",
        ("universe", "point-in-time"),
    ),
    GlossaryEntry(
        "point-in-time",
        "data",
        "Using only information that was public on the date being simulated. A "
        "quarterly report dated March 31 was not knowable until it was filed "
        "about 35 days later; using the March 31 date is lookahead.",
        ("lookahead bias", "publication date"),
    ),
    GlossaryEntry(
        "lookahead bias",
        "data",
        "Using information in a backtest that was not available at the time. The "
        "most common source of spectacular fake results, and usually invisible "
        "unless you check the dates deliberately.",
        ("point-in-time", "signal lag"),
    ),
    GlossaryEntry(
        "publication date",
        "data",
        "When a value became publicly knowable — a filing's acceptance date, a "
        "statistical release day. Distinct from the reference date (the period "
        "the value describes). Signals must align on publication date.",
        ("point-in-time", "reference date"),
    ),
    GlossaryEntry(
        "reference date",
        "data",
        "The period a value describes: June CPI refers to June 30, a Q2 balance "
        "sheet to the quarter end. Not when it was known.",
        ("publication date", "point-in-time"),
    ),
    GlossaryEntry(
        "universe",
        "data",
        "The set of symbols eligible to be held on a given date, respecting "
        "point-in-time membership. Not a fixed watchlist — a company enters when "
        "it lists and leaves when it delists.",
        ("survivorship bias", "index membership"),
    ),
    GlossaryEntry(
        "index membership",
        "data",
        "Which index a stock belonged to, and when. Stored as intervals "
        "(valid_from, valid_to), so a backtest can restrict to S&P 500 members as "
        "of each rebalance date rather than as of today.",
        ("universe", "point-in-time"),
    ),
    GlossaryEntry(
        "panel",
        "data",
        "A wide table with dates down the rows and symbols across the columns — "
        "the price panel is the canonical one. A 'long panel' is the same data "
        "with (date, symbol) as a two-level index and factors as columns.",
        ("canonical price panel",),
    ),
    GlossaryEntry(
        "canonical price panel",
        "data",
        "data/factors/prices.parquet — the wide date x symbol table of adjusted "
        "closes whose columns define the platform's universe and whose index "
        "defines the trading calendar. Everything downstream keys off it.",
        ("panel", "universe"),
    ),
    GlossaryEntry(
        "adjusted close",
        "data",
        "Closing price restated for splits and dividends, so a return computed "
        "across the adjustment is a real return rather than an artifact. The only "
        "price used for return math here.",
        ("bad print",),
    ),
    GlossaryEntry(
        "raw vs derived layer",
        "data",
        "Raw is the immutable vendor payloads under data/raw/ — refetched, never "
        "edited. Derived is everything rebuilt from raw by scripts (panels, "
        "factors), which is safe to delete and regenerate.",
        ("canonical price panel",),
    ),
    GlossaryEntry(
        "bad print",
        "data",
        "An isolated vendor quote error that snaps back the next day. Repaired in "
        "the derived layer and kept verbatim in raw, so the correction is "
        "reversible. Distinct from an outlier, which can be real.",
        ("adjusted close", "quarantine"),
    ),
    GlossaryEntry(
        "quarantine",
        "data",
        "A symbol excluded at load time because a quality check flagged it. "
        "'Flagged' means it needs review; 'cleared' means reviewed and kept.",
        ("bad print",),
    ),
    GlossaryEntry(
        "ticker reuse",
        "data",
        "One ticker used by two different companies at different times. If not "
        "handled, one price column silently splices two companies together. "
        "Guarded by permanent security ids and membership dates.",
        ("permanent security id", "universe"),
    ),
    GlossaryEntry(
        "permanent security id",
        "data",
        "An internal, opaque, never-reused identifier (qid) for one tradable "
        "security. Tickers change and get reassigned; the qid does not, so data "
        "survives renames and vendor switches. Distinct from an issuer id (SEC "
        "CIK), which identifies the company and may cover several securities.",
        ("ticker reuse",),
    ),
    GlossaryEntry(
        "S&P 500 union panel (the '774')",
        "data",
        "The original universe: every company that has EVER been in the S&P 500 "
        "since 1996 (~1,200 tickers, ~774 with usable price history), not just "
        "today's 500. It included dead names, so it was survivorship-aware — but "
        "only among large caps, which is why small-cap effects were untestable. "
        "Archived under data/factors/archive/ and still the reproduction "
        "reference for pre-cutover results (ADR 0013).",
        ("canonical price panel", "survivorship bias", "universe"),
    ),
    GlossaryEntry(
        "staleness cap",
        "data",
        "The maximum number of trading days a fundamental value forward-fills "
        "before it is treated as dead (273 here, about 13 months). Without a cap, "
        "a company that stops filing keeps contributing stale data forever.",
        ("point-in-time",),
    ),
)

_BY_TERM = {entry.term.lower(): entry for entry in GLOSSARY}


def lookup(term: str) -> GlossaryEntry | None:
    """Find an entry by term, case-insensitively."""
    return _BY_TERM.get(term.strip().lower())


def terms_in_category(category: Category) -> list[GlossaryEntry]:
    """Every entry in one category, in registry order."""
    return [entry for entry in GLOSSARY if entry.category == category]


def as_dicts() -> list[dict[str, object]]:
    """Serialise the glossary for the API."""
    return [
        {
            "term": entry.term,
            "category": entry.category,
            "definition": entry.definition,
            "see_also": list(entry.see_also),
        }
        for entry in GLOSSARY
    ]
