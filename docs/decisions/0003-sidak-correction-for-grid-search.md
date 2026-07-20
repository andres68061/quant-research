# 0003. Šidák trial-count correction for grid-searched significance tests

Date: 2026-07-19
Status: accepted

## Context

The Sortino momentum page grid-searches 40 (X, K) cells, then bootstrap-tests
the selected cell. A raw p-value on the best of 40 trials is meaningless —
selection bias under multiple testing (Bailey & López de Prado 2014). The
repo needed a correction, and already had a Deflated Sharpe Ratio module.

## Decision

`bootstrap_significance_test` reports `p_value_adjusted = 1 - (1 - p)^N`
(Šidák) with `N` defaulting to the full grid size (`GRID_N_TRIALS = 40`),
plus `significant_after_correction`. The UI leads with the adjusted value.
`N` counts all *attempted* cells, including ones skipped for min-signals —
under-counting trials overstates significance; when in doubt count more.

## Alternatives rejected

- **Deflated Sharpe Ratio here** — DSR corrects a *Sharpe* selected from
  trials, using the cross-trial Sharpe variance. The grid's metric is a
  hit rate, not a Sharpe; forcing it through DSR would misuse the formula.
  DSR (`core/metrics/deflated_sharpe.py`) remains the tool for
  Sharpe-based selection (used in the pairs-basket verdict, ROADMAP).
- **Bonferroni (`p * N`)** — nearly identical at small p but not a
  probability (can exceed 1); Šidák is exact under independence at the
  same cost.
- **Recomputing the null "best of 40" distribution by simulation** — most
  accurate (handles correlated trials, which Šidák over-corrects), but 40x
  the bootstrap compute per request on an interactive endpoint. Revisit if
  the corrected p-value ever becomes a decision gate rather than a display.

## Consequences

Grid cells are correlated, so Šidák is conservative — a cell surviving the
correction is strong evidence; one failing it is not proof of nothing.
Callers testing a pre-registered (not grid-selected) pair pass `n_trials=1`.
