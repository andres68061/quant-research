# 0010. Classify every vendor dataset by point-in-time status before use

Date: 2026-08-07
Status: accepted

## Context

Expanding the FMP footprint from 10 endpoints to 26 brought in datasets whose
date columns mean very different things, and the difference is invisible at the
schema level. Three examples that all look alike:

- `earnings` rows dated `2024-01-29` — the day the result was **announced**.
  Knowable that day.
- `ratios` rows dated `2024-03-31` — the **fiscal period** the ratio describes.
  Not knowable until the 10-Q was filed, typically 30–45 days later.
- `financial_scores` — a single row with **no date at all**, recomputed from
  today's fundamentals. Not knowable at any past date.

Ranking on `ratios` joined naively on its `date` column leaks roughly six weeks
of future information into every backtest. Nothing in the payload signals this;
the column is called `date` in all three cases. The repo's existing
point-in-time vocabulary (`reference_date` / `publication_date` / `as_of_date`)
already names the distinction, but there was no place to record which vendor
dataset is which, so the answer would have to be re-derived — or guessed —
every time someone reached for a new endpoint.

## Decision

Every per-symbol FMP dataset is declared in a registry
(`core/data/fmp/datasets.py`) with a mandatory `pit_status` field:

- **`point_in_time`** — rows carry the date the information became public
  (announcement date, filing date, observation date). Usable directly.
- **`period_end_only`** — rows are stamped with the fiscal period, not the
  publication date. Must be joined to `filingDate`/`acceptedDate` from the raw
  statements on `(symbol, date)` before any backtest use.
- **`snapshot`** — a single current-value row. Reconciliation and live
  screening only; never backtesting.

A test asserts every registered dataset declares one of the three. The status
is repeated in `docs/DATA_INVENTORY.md` so it is visible without reading code.

Vendor-computed ratios (`key_metrics`, `ratios`, `financial_growth`,
`enterprise_values`) are all `period_end_only`. Where we need those quantities
for research **now**, we compute them ourselves from the raw statements via
`core/data/factors/statement_metrics.py`, which aligns on `acceptedDate` and is
point-in-time by construction.

## Alternatives rejected

- **Trust the vendor's `date` column uniformly.** The failure is silent and
  directionally favourable — leaked fundamentals make backtests look better, so
  nothing about the result would prompt a second look. This is the single most
  expensive mistake available in this repo.

- **Apply a blanket 45-day lag to every fundamental dataset.** Cheap, but wrong
  in both directions: it delays genuinely point-in-time data like `earnings` by
  six weeks (destroying PEAD, which lives entirely in the announcement window),
  while still under-lagging late filers. Real filing dates are already in the
  raw statements — there is no reason to approximate them.

- **Only ingest point-in-time datasets.** Would exclude `financial_scores`,
  which is our reconciliation target for our own Piotroski/Altman
  implementations and caught a real specification bug (see ADR 0011). Snapshots
  have legitimate non-backtest uses; the fix is labelling, not exclusion.

- **Document it in prose only.** Prose drifts. A required dataclass field
  cannot be omitted when someone adds a dataset in a hurry.

## Consequences

Adding a dataset now requires an explicit judgement about its date semantics,
which is a deliberate speed bump at the point where the mistake is cheapest to
avoid. The `period_end_only` datasets are downloaded but **not yet wired into
any factor** — the statement-filing join is still to be written, and any future
use of them must do that join first rather than treating the download as
sufficient.

Snapshot datasets accumulate no history unless we snapshot them repeatedly. If
analyst-estimate revision signals become interesting, `analyst_estimates` needs
a scheduled capture that appends dated snapshots; today's file is a single
observation and cannot be replayed historically.
