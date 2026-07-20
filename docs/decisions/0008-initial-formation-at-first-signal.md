# 0008. Form the initial portfolio at the first actionable signal

Date: 2026-07-19
Status: accepted

## Context

Rebalance dates were only period-ends, so a backtest starting mid-period sat
100% in cash until the first period-end — 52 trading days for a quarterly
backtest starting 2021-07-19. A live portfolio launched that day would invest
immediately; the idle stretch was a scheduling artifact, and it biased
invested-coverage and every return metric for the first partial period.

## Decision

`calculate_portfolio_returns` adds one extra rebalance at the first date with
any nonzero signal, when that date precedes the first scheduled period-end
(with `signal_lag_days=1` this is typically the window's second bar).
`create_weighted_portfolio` forms weights on the first bar. The periodic
schedule is unchanged after initial formation.

## Alternatives rejected

- **Keep waiting for the first period-end** (status quo) — understates
  invested coverage and makes results depend on where the start date falls
  inside a quarter.
- **Always rebalance on bar 0** — with `signal_lag_days >= 1` there is no
  actionable signal on the first bar; forming there would either do nothing
  or require peeking at the same-day factor value (lookahead).
- **Shift the whole schedule to anniversary dates of the start** — breaks
  comparability with calendar-period conventions used everywhere else in the
  repo and in the literature.

## Consequences

Every backtest's first partial period now participates; historical numbers
(failure log, replay precomputes) shift slightly and are not comparable to
runs before this change. First holding period is shorter than the rest —
same as live trading. Sparse factors whose coverage starts after the window
start still go flat until coverage exists (ADR 0006 semantics unchanged).
