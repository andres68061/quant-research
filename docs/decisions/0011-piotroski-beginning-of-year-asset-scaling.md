# 0011. Scale Piotroski F-score flows by beginning-of-year assets

Date: 2026-08-07
Status: accepted

## Context

The Piotroski (2000) F-score has nine binary tests. Three of them compare a flow
item to total assets: `ROA > 0`, `ΔROA > 0`, and `ΔAssetTurnover > 0`. The paper
scales those flows by total assets at the **beginning of the year**, not by
contemporaneous period-end assets.

Our first implementation used contemporaneous assets, which is the intuitive
reading and what several vendor implementations do. On a fast-growing company
this systematically misfires: assets grow faster than earnings during an
expansion, so a firm with genuinely rising profitability is scored as
deteriorating. AAPL's most recent quarter scored 8/9 under contemporaneous
scaling and 9/9 under the paper's — revenue grew 14.2% while assets grew 15.6%,
so contemporaneous turnover fell even though turnover on opening assets rose.

FMP's `financial_scores` endpoint reported 9. That disagreement is what surfaced
the bug; without an independent implementation to reconcile against, a
one-signal error on a 0–9 score would have been invisible.

## Decision

Follow the paper. `core/data/factors/quality_scores.py` scales ROA, CFO and
asset turnover by `total_assets_lag4` (assets four quarters back), and scores
the prior year using `total_assets_lag8`. The leverage test uses average total
assets, also per the paper. This requires `statement_metrics` to emit a
`total_assets_lag8` column, whose only consumer is this score.

The generic `roa` **factor** in `fundamental_factors.py` keeps contemporaneous
scaling — that is the standard factor definition and is a different quantity
from the score's internal ROA input. The two are intentionally not shared.

Reconciliation against FMP's `financial_scores` snapshot is what surfaced the
bug, and AAPL and MSFT both match exactly after the change.

**Across all 709 reconcilable symbols the match is much looser: 41% exact, 84%
within one point, Spearman 0.72.** That is expected and is not evidence of a
remaining bug — the disagreement is symmetric (median 0, mean −0.22) and the two
distributions are nearly identical (ours 6.05 ± 1.58, vendor 6.26 ± 1.51), which
is the signature of a definitional difference rather than an error. The
difference is that **Piotroski specified the score on annual data and we compute
it on TTM quarterly data.** Four of the nine tests are year-over-year deltas, and
a TTM window straddling two fiscal years will disagree with an annual window on
borderline cases.

We keep TTM quarterly deliberately: annual data updates once a year and is up to
twelve months stale, which is unusable for a daily backtest. The cost is that our
`piotroski_f` is not interchangeable with a vendor's, and cross-vendor
comparisons of individual scores are meaningless.

Altman Z reconciles far more tightly (n=684, correlation 0.98, median relative
error 0.94%, 83% within 5%) because it is a continuous formula over levels with
no year-over-year deltas.

## Alternatives rejected

- **Keep contemporaneous scaling.** Simpler and needs no `lag8` column, but it
  is not the published score. Anyone comparing our results to the Piotroski
  literature — the entire reason to use a 26-year-old scoring rule rather than
  fitting our own — would be comparing against something else wearing the same
  name.

- **Use the vendor's `financial_scores` directly instead of computing it.** The
  endpoint is a `snapshot` with no history (see ADR 0010), so it cannot be
  backtested at all. It is only usable as a reconciliation target, which is
  exactly how we use it.

- **Average of opening and closing assets for all three tests.** A defensible
  smoothing choice and what the leverage test uses, but the paper is explicit
  about beginning-of-year for ROA and turnover. Deviating would reintroduce the
  same "which Piotroski is this?" ambiguity.

## Consequences

Scores need eight quarters of history plus four for the TTM window, so a
symbol's first F-score arrives roughly three years after its first filing —
later than under contemporaneous scaling. Combined with the existing rule that
withholds a score when fewer than seven tests are evaluable, `piotroski_f`
coverage is 89% of the fundamentals panel versus 98% for simple ratios. That is
the correct trade: a partial score is not comparable with a complete one.

Our score will disagree with vendor implementations that use contemporaneous
scaling. That is intended, and this ADR is the answer when the disagreement is
noticed again.
