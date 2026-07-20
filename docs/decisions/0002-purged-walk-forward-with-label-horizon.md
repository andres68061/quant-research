# 0002. Purge walk-forward training windows by label horizon

Date: 2026-07-19
Status: accepted

## Context

Forward-looking labels built with `prices.shift(-horizon)` mean the last
`horizon - 1` rows of a training window are computed from returns inside
the adjacent test window — the model partially trains on the outcome it is
tested on (López de Prado, *Advances in Financial ML*, ch. 7). The shipped
ML pipeline uses next-day labels (horizon 1, no overlap), but the validator
offered no protection for any horizon > 1.

## Decision

`WalkForwardValidator` takes `label_horizon_days` (default 1) and
`embargo_days` (default 0); it purges `(horizon - 1) + embargo` rows from
the end of every training window. The caller must declare the horizon —
the validator cannot infer it from the data. Invalid configs raise
`ConfigError` at construction. Defaults reproduce legacy splits exactly.

## Alternatives rejected

- **Full purged K-fold with overlap bookkeeping per de Prado** — strictly
  more general, but this validator only produces forward-adjacent test
  windows, where "purge the trailing rows" is exactly equivalent and ~20
  lines instead of a dependency on interval trees.
- **Inferring the horizon from label autocorrelation** — magic; wrong
  inferences would silently under-purge.
- **Changing the default to a conservative purge (e.g. 5)** — would
  silently shrink training sets for the shipped horizon-1 pipeline, which
  needs no purge.

## Consequences

Any future multi-day-horizon label (e.g. `n_day_direction(horizon=5)`)
is safe *only if* the caller passes the matching `label_horizon_days`;
`core/features/labels.py` docstring carries the warning. The API exposes
both parameters on `POST /run-ml-strategy`.
