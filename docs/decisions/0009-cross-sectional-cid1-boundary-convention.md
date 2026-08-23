# 0009. Cross-sectional Cid-1: fixed trailing window, +inf at pain = 0, atan2 for ranking

Date: 2026-07-30
Status: accepted

## Context

Using Cid-1 (total return ÷ cost-basis pain) as a per-stock cross-sectional
characteristic breaks the strategy-level definition twice. First, the
strategy metric is path-dependent from an arbitrary inception date, so
numbers are not comparable across 500 stocks. Second,
`calculate_cid1_ratio` returns **0.0 when pain == 0** — sensible as a
"no evidence" answer for one equity curve, but fatal for ranking: a stock
that never closed below its window-start value (the *best* path under this
metric's ideology) would tie with dead-flat losers.

## Decision

`calculate_trailing_cid1_cross_section` (core/metrics/cross_section.py):

- Common **252-trading-day trailing window** ending at each evaluation
  date; symbols with any missing close in the window are excluded (logged).
- `cid1_ratio` maps the boundary to **+inf** when pain == 0 and return > 0
  (0.0 only for the degenerate flat path) — the limit the ratio actually
  approaches as pain → 0.
- `cid1_angle = atan2(total_return, cost_basis_pain)` is the ranking /
  regression variable: finite everywhere, identical ordering to the ratio
  where pain > 0, continuous at the boundary (pain-free paths → π/2).
  Quantile sorts break the π/2 ties by total return.

## Alternatives rejected

- **Keep the 0.0 convention** — ranks the best paths with the worst; any
  cross-sectional result would be an artifact of the boundary.
- **Floor pain at an epsilon** — produces astronomical, scale-dependent
  ratio values that poison regressions and depend on the chosen epsilon.
- **Exclude pain-free stocks** — selection bias: systematically removes
  the strongest recent performers from every cross-section (~6% of the
  top-500 universe on an average rebalance date).

## Consequences

Strategy-level Cid-1 (`calculate_cid1_ratio`) keeps its 0.0 convention —
the two functions intentionally disagree at the boundary. Regressions use
cross-sectional z-scored ranks of `cid1_angle`, never the raw ratio.
Revisit if a use case needs cardinal (not ordinal) cross-sectional values,
or if the pain-free share of the universe grows enough that π/2 ties
dominate a quantile bucket.
