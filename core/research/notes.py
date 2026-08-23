"""Research notes — the durable, readable record of what an experiment found.

The gap this fills. An experiment currently produces three things: a JSON blob in
``data/quality/``, a log file, and an entry in the roadmap or failure log. None of
those is *readable* — the JSON has the numbers but no argument, the log has the
process, and the failure-log entry has the conclusion but not the table that
supports it. The reasoning existed only in whatever conversation produced it.

A note is the fourth artifact: the numbers, the argument, and the caveats in one
place, structured so every note answers the same questions in the same order.

**The required sections, and why each is required.**

- ``question`` — the test in one plain paragraph, no jargon. If this cannot be
  written plainly, the experiment is not yet well posed.
- ``hypothesis`` — why it might work, with the citation. Separates "an effect
  someone documented" from "a pattern found by searching".
- ``method`` — construction, universe, costs, period. What was actually run.
- ``results`` — every variant tried, not the best one. Reporting only the winner
  is how multiple testing hides.
- ``control`` — the deliberately signal-free comparison. **Required**, because
  the single most common error here has been mistaking market exposure for
  skill: a long-only book scoring 0.85 against a no-signal control of 0.67 adds
  0.18, not 0.85.
- ``verdict`` — one of four fixed statuses, so notes stay comparable.
- ``caveats`` — what would change the conclusion.
- ``reproduce`` — the command. A result nobody can re-run is an anecdote.

Notes are written here rather than generated from the JSON because the argument
is the part that matters and it cannot be inferred from numbers. The numbers are
copied from the persisted experiment files and are checked against them by
``tests/test_notes.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "NOTES",
    "ResearchNote",
    "ResultRow",
    "Verdict",
    "as_dicts",
    "get_note",
]

Verdict = Literal["validated", "interesting", "no_edge", "real_not_tradable"]

VERDICT_LABELS: dict[Verdict, str] = {
    "validated": "Validated",
    "interesting": "Interesting, not validated",
    "no_edge": "No edge",
    "real_not_tradable": "Real, but not tradable",
}


@dataclass(frozen=True)
class ResultRow:
    """One variant tried, with the metrics that decide it."""

    variant: str
    gross_sharpe: float | None = None
    net_sharpe: float | None = None
    net_annual_return: float | None = None
    t_stat: float | None = None
    hit_rate: float | None = None
    max_drawdown: float | None = None
    note: str | None = None
    is_control: bool = False
    is_headline: bool = False


@dataclass(frozen=True)
class ResearchNote:
    """One experiment, written so a reader needs nothing else."""

    id: str
    title: str
    run_date: str
    verdict: Verdict
    one_liner: str
    question: str
    hypothesis: str
    reference: str
    method: str
    results: tuple[ResultRow, ...]
    what_it_means: str
    control_reading: str
    caveats: tuple[str, ...]
    reproduce: str
    decade_table: tuple[tuple[str, str, str, str], ...] = ()
    glossary_terms: tuple[str, ...] = ()
    logged_in: str = ""
    charts: tuple[dict[str, object], ...] = field(default_factory=tuple)


NOTES: tuple[ResearchNote, ...] = (
    # ------------------------------------------------------------------ PEAD
    ResearchNote(
        id="pead-tradability",
        title="Post-earnings drift: real, and unreachable after costs",
        run_date="2026-08-14",
        verdict="real_not_tradable",
        one_liner=(
            "The drift exists and is statistically overwhelming. Trading it loses money "
            "once costs scale with each stock's liquidity — flat 10 bps says net Sharpe "
            "0.70, realistic costs say -0.11."
        ),
        question=(
            "When a company reports earnings better than analysts expected, its share "
            "price jumps that day — and then keeps drifting in the same direction for "
            "weeks afterwards, because the market under-reacts to the news. That drift "
            "is one of the most documented effects in finance. The question here is not "
            "whether it exists; we had already measured it at t = 10.05 across 267,780 "
            "announcements. The question is whether you can actually make money from it: "
            "buy the biggest positive surprises, short the biggest negative ones, hold "
            "each for 60 trading days, subtract what the trading really costs — do you "
            "end up ahead?"
        ),
        hypothesis=(
            "Investors under-react to earnings news, so prices adjust gradually instead "
            "of instantly. The gradual part is capturable if you can trade it cheaply "
            "enough."
        ),
        reference="Bernard & Thomas (1989); Foster, Olsen & Shevlin (1984)",
        method=(
            "An overlapping book, the standard construction for turning an event study "
            "into a return series. Every announcement opens a position the day AFTER the "
            "announcement (you cannot trade on news you have not seen) and holds it for "
            "60 trading days. Because announcements arrive continuously, on any given "
            "day the book holds roughly 60 overlapping cohorts and about 1/60th of it "
            "turns over. Positions go long the top surprise quintile and short the "
            "bottom, gross exposure normalised to 1.0 daily. Shells and SPACs excluded; "
            "returns require price >= $1 and |return| <= 300% to reject vendor defects."
        ),
        results=(
            ResultRow(
                "A — long/short, flat 10 bps",
                gross_sharpe=1.106,
                net_sharpe=0.697,
                net_annual_return=0.0184,
                hit_rate=0.428,
                max_drawdown=-0.12,
                note="The repo's default cost assumption. Looks like a working strategy.",
            ),
            ResultRow(
                "B — long/short, liquidity-scaled costs",
                gross_sharpe=1.106,
                net_sharpe=-0.112,
                net_annual_return=-0.003,
                hit_rate=0.408,
                max_drawdown=-0.241,
                is_headline=True,
                note="Same book, honest costs: 3.2%/yr of cost against 2.9%/yr of gross return.",
            ),
            ResultRow(
                "C — liquid names only",
                gross_sharpe=0.721,
                net_sharpe=0.186,
                net_annual_return=0.0061,
                hit_rate=0.412,
                max_drawdown=-0.155,
                note="Restricting to tradable names halves the GROSS edge too — the effect "
                "genuinely lives in the illiquid tail.",
            ),
            ResultRow(
                "D — 120-day hold",
                gross_sharpe=0.817,
                net_sharpe=0.147,
                net_annual_return=0.0038,
                hit_rate=0.411,
                max_drawdown=-0.149,
                note="Halving turnover halves the cost — and the edge with it.",
            ),
            ResultRow(
                "E — long-only top quintile",
                gross_sharpe=1.021,
                net_sharpe=0.848,
                net_annual_return=0.1546,
                hit_rate=0.451,
                max_drawdown=-0.569,
                note="The apparent rescue. Read it against the control below, not on its own.",
            ),
            ResultRow(
                "F — long-only, ALL announcers, no signal (control)",
                gross_sharpe=0.817,
                net_sharpe=0.665,
                net_annual_return=0.1175,
                hit_rate=0.445,
                max_drawdown=-0.578,
                is_control=True,
                note="Holds every company that reported, ranked by nothing.",
            ),
        ),
        control_reading=(
            "Variant E scored net Sharpe 0.848 and looked like the answer. The control, "
            "F, holds exactly the same kind of book but replaces the signal with a "
            "constant — every announcer, no ranking at all — and scores 0.665. The "
            "signal's genuine contribution is the difference: **+0.18 Sharpe**, not 0.85. "
            "The rest is simply being long the stock market during a period when the "
            "stock market went up. Without running F, this experiment would have been "
            "written up as a success."
        ),
        what_it_means=(
            "The gap between A and B is the entire finding, and it is a finding about "
            "method rather than about earnings. A flat 10 bps cost assumption — the "
            "default in this repo and in most published work — turns a dead strategy "
            "into an apparently good one. Real costs are not flat: trading a thinly "
            "traded small-cap moves its price against you far more than trading Apple "
            "does, so cost has to scale with each stock's dollar volume.\n\n"
            "The rescues fail informatively. Restricting to liquid names (C) does cut "
            "the cost, but it also halves the gross edge, which tells us the drift is "
            "concentrated in exactly the stocks that are expensive to trade. Holding "
            "longer (D) cuts turnover and cost, and cuts the signal by the same "
            "proportion. This is the classic shape of a real anomaly you cannot reach: "
            "every lever that reduces the cost reduces the edge at least as fast.\n\n"
            "This does not retract the earlier PEAD result. The drift is real and the "
            "event study stands. What is retracted is any implication that it is a "
            "strategy."
        ),
        caveats=(
            "The ADV cost schedule is an estimate, not measured fills. It is calibrated "
            "to public studies of price impact by liquidity bucket; a real broker's "
            "costs could be somewhat lower.",
            "Quintile breakpoints are computed across all announcers in the window, not "
            "sector-by-sector, so sector composition drifts with the earnings calendar.",
            "The book is unlevered and gross exposure is normalised daily. A levered "
            "implementation would scale both the return and the cost, leaving the "
            "conclusion unchanged.",
            "Short availability and borrow cost are not modelled. Both make the "
            "long/short variants worse, not better.",
        ),
        reproduce=("/opt/anaconda3/envs/quant/bin/python scripts/experiment_pead_tradable.py"),
        glossary_terms=(
            "PEAD",
            "SUE",
            "overlapping portfolio",
            "dollar ADV",
            "transaction costs",
            "gross vs net returns",
            "control variant",
            "long-only",
            "market beta",
        ),
        logged_in="docs/FAILED_STRATEGIES_LOG.md",
    ),
    # ------------------------------------------------------------- Piotroski
    ResearchNote(
        id="conditional-piotroski",
        title="Piotroski's F-score inside the value quintile: adds nothing",
        run_date="2026-08-13",
        verdict="no_edge",
        one_liner=(
            "Applied the way the paper intends — only to cheap stocks — the 9-point "
            "quality score does not beat ignoring it. Plain value beats both."
        ),
        question=(
            "Joseph Piotroski's 2000 paper proposes a 9-point checklist of accounting "
            "health: is the company profitable, is that profit backed by real cash, is "
            "debt falling, are margins and efficiency improving. Each test scores 0 or 1, "
            "so a company ends up somewhere between 0 and 9. The crucial detail, and the "
            "one usually dropped: he applied it ONLY to cheap stocks. Among bargains, the "
            "score is meant to separate companies that are cheap because they are "
            "temporarily out of favour from companies that are cheap because they are "
            "dying. Our earlier 28-factor screen had tested the score on the entire "
            "market, which is not the paper's claim. So we tested it the paper's way: "
            "take the cheapest 20% of stocks, then within that group buy the ones with "
            "high scores. Does the score add anything to being cheap?"
        ),
        hypothesis=(
            "Among cheap stocks, fundamental strength separates recoveries from value "
            "traps, so a quality filter should improve on value alone."
        ),
        reference="Piotroski (2000), Journal of Accounting Research",
        method=(
            "Long/short and long-only variants on top/bottom 30% tiers, monthly "
            "rebalance, 10 bps one-way, S&P 500 point-in-time membership, shells and "
            "SPACs excluded. The value quintile is the top 20% by book-to-market "
            "computed per date. Period 2000-01-03 to 2026-08-07 (6,689 trading days)."
        ),
        results=(
            ResultRow(
                "A — F-score within value, long/short",
                net_sharpe=-0.385,
                net_annual_return=-0.1055,
                t_stat=-1.99,
                max_drawdown=-0.991,
                note="The short leg is distressed value stocks, which rocket in recoveries.",
            ),
            ResultRow(
                "B — F-score within value, long-only",
                net_sharpe=0.508,
                net_annual_return=0.1341,
                t_stat=2.62,
                max_drawdown=-0.752,
                is_headline=True,
                note="Using the score.",
            ),
            ResultRow(
                "F — value quintile, score IGNORED (control)",
                net_sharpe=0.539,
                net_annual_return=0.17,
                t_stat=2.78,
                max_drawdown=-0.821,
                is_control=True,
                note="Same stocks, held flat, no scoring at all — and it does better.",
            ),
            ResultRow(
                "E — plain book-to-market, long-only",
                net_sharpe=0.678,
                net_annual_return=0.234,
                t_stat=3.49,
                max_drawdown=-0.597,
                note="Just buy cheap stocks. Beats every score-based variant.",
            ),
            ResultRow(
                "C — plain book-to-market, long/short",
                net_sharpe=0.387,
                net_annual_return=0.1082,
                t_stat=1.99,
                max_drawdown=-0.543,
                note="The honest baseline for a long/short comparison.",
            ),
            ResultRow(
                "D — F-score standalone, whole market, long/short",
                net_sharpe=-0.252,
                net_annual_return=-0.0235,
                t_stat=-1.3,
                max_drawdown=-0.621,
                note="The original screen's version — not the paper's claim.",
            ),
        ),
        control_reading=(
            "Variant B (0.508) is the strategy; variant F (0.539) is the same basket of "
            "cheap stocks with the score thrown away. F wins. The score is not adding "
            "information — it is subtracting some, by concentrating the book into fewer "
            "names for no compensating gain.\n\n"
            "There is a second, sharper lesson in this table. The first version of this "
            "experiment compared B (long-only, 0.508) against C (long/short, 0.387) and "
            "concluded the conditional score worked. That comparison is meaningless: a "
            "long-only book carries market exposure that a long/short book cancels, so it "
            "compares a strategy plus the stock market against a strategy alone. Adding "
            "the like-for-like controls reversed the conclusion completely."
        ),
        what_it_means=(
            "On this universe and this period, the F-score adds nothing to value, and "
            "value on its own is the better strategy. That is a negative result about "
            "our universe, not a refutation of the paper: Piotroski's sample was "
            "small-cap, thinly-followed US stocks in 1976-1996, where accounting "
            "information was genuinely less processed. Our test runs on S&P 500 members "
            "from 2000 onward — large, heavily-analysed companies whose balance sheets "
            "are picked over by thousands of people. If the effect comes from "
            "information nobody bothered to read, large caps are exactly where it should "
            "be absent.\n\n"
            "The long/short variant (A, -0.385, with a 99% drawdown) is worth "
            "understanding rather than dismissing. Its short leg is low-score cheap "
            "stocks — that is, distressed companies. Distressed companies are the ones "
            "that go up 400% when they survive. Shorting them means being crushed in "
            "every recovery, which is precisely what the drawdown records."
        ),
        caveats=(
            "S&P 500 membership only. The paper's effect is documented on small caps, "
            "which this test cannot reach; the expanded universe makes that test "
            "possible and it has not been run.",
            "Book-to-market comes from vendor fundamentals joined on filing date. "
            "Placeholder filing dates in older data (ADR 0012) add noise pre-2010.",
            "Tier width of 30% for the score and 20% for value was chosen to match the "
            "original screen, not tuned. Tuning it would itself require a wider "
            "significance bar.",
        ),
        reproduce=(
            "/opt/anaconda3/envs/quant/bin/python scripts/experiment_conditional_piotroski.py"
        ),
        decade_table=(
            ("A — F-score in value, L/S", "-0.57", "-0.19", "-0.23"),
            ("B — F-score in value, long", "0.43", "0.75", "0.35"),
            ("E — plain value, long", "0.80", "0.75", "0.52"),
            ("F — value, score ignored", "0.53", "0.76", "—"),
        ),
        glossary_terms=(
            "Piotroski F-score",
            "value factor",
            "long/short",
            "long-only",
            "control variant",
            "market beta",
            "maximum drawdown",
        ),
        logged_in="docs/FAILED_STRATEGIES_LOG.md",
    ),
    # ------------------------------------------------------------------- NOA
    ResearchNote(
        id="net-operating-assets",
        title="Net operating assets: the one factor of 28 that strengthened",
        run_date="2026-08-11",
        verdict="interesting",
        one_liner=(
            "Sharpe 0.42, t = 2.16 — below the 3.12 multiple-testing bar, so not "
            "validated. What makes it worth keeping is the shape: 0.27 / 0.53 / 0.65 "
            "across three decades, when every other survivor decayed."
        ),
        question=(
            "We tested 28 published stock-market anomalies — profitability, accruals, "
            "value, quality, momentum and the rest — through one identical pipeline: "
            "rank every stock, buy the top 20%, short the bottom 20%, rebalance monthly, "
            "subtract 10 bps of costs, and see whether you made money over 2000-2026. "
            "None of them cleared the significance bar. But one behaved differently from "
            "all the others in a way that is hard to explain by luck, and this note is "
            "about that one: net operating assets."
        ),
        hypothesis=(
            "Net operating assets is the operating side of the balance sheet — "
            "receivables, inventory, plant — minus operating liabilities, scaled by "
            "prior-year assets. Its meaning is that it is a running total of the gap "
            "between the profits a company reports and the cash it actually collects. "
            "Every quarter that reported earnings exceed cash earnings, the difference "
            "has to be parked somewhere on the balance sheet, and this is where it "
            "accumulates. So a high and rising value means years of earnings that were "
            "accounting entries rather than money. Those entries eventually reverse, and "
            "returns disappoint when they do. We rank by the NEGATIVE of it (hence "
            "neg_net_operating_assets), so 'high' means the side you want to own."
        ),
        reference="Hirshleifer, Hou, Teoh & Zhang (2004), Journal of Accounting and Economics",
        method=(
            "Identical to the other 27 factors in the screen: rank all stocks "
            "cross-sectionally each month, long the top 20% and short the bottom 20%, "
            "equally weighted, 10 bps one-way costs, S&P 500 point-in-time membership, "
            "one-day signal lag. Period 2000-01-03 to 2026-08-07 (6,689 trading days). "
            "The significance bar is Sidak-corrected for 28 simultaneous tests: |t| >= "
            "3.12 rather than the usual 2.0."
        ),
        results=(
            ResultRow(
                "neg_net_operating_assets",
                net_sharpe=0.42,
                net_annual_return=0.0461,
                t_stat=2.16,
                max_drawdown=-0.28,
                is_headline=True,
                note="Below the 3.12 bar. Not validated — but stable across decades.",
            ),
            ResultRow(
                "amihud_illiquidity (highest t in the screen)",
                net_sharpe=0.468,
                t_stat=2.41,
                note="Higher t, but decays 1.04 -> 0.11 -> -0.20. The classic "
                "arbitraged-away shape.",
            ),
            ResultRow(
                "sales_to_price",
                net_sharpe=0.42,
                t_stat=2.16,
                note="Same headline numbers, decaying profile: 0.74 -> -0.01 -> 0.34.",
            ),
            ResultRow(
                "operating_profitability",
                net_sharpe=0.366,
                t_stat=1.89,
                note="Also stable (0.37 / 0.39 / 0.46) but weaker throughout.",
            ),
            ResultRow(
                "piotroski_f",
                net_sharpe=-0.332,
                t_stat=-1.71,
                note="Negative on the whole market — see the conditional-Piotroski note.",
            ),
        ),
        control_reading=(
            "The relevant control here is not a no-signal variant but the other 27 "
            "factors, run identically. That is what makes the decade profile "
            "interpretable: every factor faced the same universe, costs and period, so "
            "the difference in shape is about the factors, not the pipeline. "
            "amihud_illiquidity has the higher t-statistic and would be the pick on that "
            "number alone — it also has the steepest decay in the entire screen."
        ),
        what_it_means=(
            "Two readings, and the honest answer is that we cannot yet distinguish them.\n\n"
            "The interesting reading: a published anomaly usually decays after "
            "publication, because people read the paper and trade it away. Decay is the "
            "fingerprint of an effect that WAS real and is being competed out. Stability "
            "or strengthening is what a structural effect looks like — something rooted "
            "in how accounting works rather than in who has read what. Of the 28 factors "
            "tested, this is the only one whose profile strengthens monotonically across "
            "the three decades.\n\n"
            "The boring reading: with 28 factors and three decades each, some factor was "
            "always going to have a monotonically rising profile by chance. That is 28 "
            "chances at a 1-in-6 ordering. Expected count of monotone-rising profiles "
            "under pure noise: about 4.7. Finding one is not surprising.\n\n"
            "This is exactly why it is filed as 'interesting' and not 'validated'. The "
            "test that separates the readings is out-of-sample data the screen has not "
            "seen: the post-cutover universe includes small and mid caps, where the "
            "original paper documented the effect most strongly, and where it has never "
            "been tested here."
        ),
        caveats=(
            "t = 2.16 is below the 3.12 bar for 28 tests. Reported as a lead, not a " "result.",
            "The decade split is three samples of roughly 2,200 days each; the "
            "per-decade Sharpes carry wide error bars and their ordering is not itself "
            "significant.",
            "S&P 500 members only. The original paper's effect is strongest in smaller "
            "companies, which this test excludes.",
            "Balance-sheet inputs are joined on filing date, but older filings carry "
            "placeholder dates (ADR 0012), so the pre-2010 portion has more timing noise "
            "than the rest.",
        ),
        reproduce=("/opt/anaconda3/envs/quant/bin/python scripts/screen_factor_library.py"),
        decade_table=(
            ("neg_net_operating_assets", "0.27", "0.53", "0.65"),
            ("amihud_illiquidity", "1.04", "0.11", "-0.20"),
            ("sales_to_price", "0.74", "-0.01", "0.34"),
            ("operating_profitability", "0.37", "0.39", "0.46"),
        ),
        glossary_terms=(
            "net operating assets",
            "accruals",
            "significance bar",
            "Sidak correction",
            "multiple testing",
            "cross-sectional strategy",
        ),
        logged_in="docs/ROADMAP.md",
    ),
)

_BY_ID = {note.id: note for note in NOTES}


def get_note(note_id: str) -> ResearchNote | None:
    """Find a note by id."""
    return _BY_ID.get(note_id)


def as_dicts(full: bool = True) -> list[dict[str, object]]:
    """
    Serialise notes for the API.

    Args:
        full: When False, return only the index fields, so the list view does not
            ship every note's full prose.
    """
    payload: list[dict[str, object]] = []
    for note in NOTES:
        summary: dict[str, object] = {
            "id": note.id,
            "title": note.title,
            "run_date": note.run_date,
            "verdict": note.verdict,
            "verdict_label": VERDICT_LABELS[note.verdict],
            "one_liner": note.one_liner,
        }
        if full:
            summary.update(
                {
                    "question": note.question,
                    "hypothesis": note.hypothesis,
                    "reference": note.reference,
                    "method": note.method,
                    "results": [
                        {
                            "variant": row.variant,
                            "gross_sharpe": row.gross_sharpe,
                            "net_sharpe": row.net_sharpe,
                            "net_annual_return": row.net_annual_return,
                            "t_stat": row.t_stat,
                            "hit_rate": row.hit_rate,
                            "max_drawdown": row.max_drawdown,
                            "note": row.note,
                            "is_control": row.is_control,
                            "is_headline": row.is_headline,
                        }
                        for row in note.results
                    ],
                    "control_reading": note.control_reading,
                    "what_it_means": note.what_it_means,
                    "caveats": list(note.caveats),
                    "reproduce": note.reproduce,
                    "decade_table": [list(row) for row in note.decade_table],
                    "glossary_terms": list(note.glossary_terms),
                    "logged_in": note.logged_in,
                }
            )
        payload.append(summary)
    return payload
