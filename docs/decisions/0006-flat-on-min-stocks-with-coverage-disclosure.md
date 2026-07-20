# 0006. Go flat when min_stocks fails; disclose invested coverage

Date: 2026-07-19
Status: accepted

## Context

Sparse fundamentals factors (e.g. `value_quality`) often have rebalance dates
with fewer than `min_stocks` valid names. The portfolio then holds cash and
the equity curve goes horizontal. Replay also omitted long/short headcounts,
so every frame showed FLAT — looking like a regime filter or a bug.

## Decision

Keep going to cash (0% return) when ranking coverage fails. Surface
`InvestedCoverage` on backtest + replay responses and in the Factor Backtest
UI (pct days invested, flat count, longest flat streak, warning copy). Replay
must use the same run params and label position from `n_long` / `n_short`.

## Alternatives rejected

- **Hold last weights through coverage holes** — pretends we still had a
  signal; hides data gaps inside a smooth curve.
- **Rank on whatever names remain below min_stocks** — unstable micro-books;
  `min_stocks` exists to refuse that.
- **Credit T-bills on cash by default** — better economics later, but would
  change every historical Sharpe; disclose `cash_earns_zero` first.

## Consequences

Flat stretches remain possible and are intentional. UI/API must warn when
`pct_days_invested < 0.95`. Revisit if we add optional RF-on-cash or if
fundamentals coverage is rebuilt to research-grade for VQ.
