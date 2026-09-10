# 0017. Group scripts and data modules by kind, and raise the module cap to 500 lines

Date: 2026-09-09
Status: accepted

## Context

The three-layer split (`frontend/` → `api/` → `core/`, with I/O kept out of
`core/` computation) was sound and stays. What had sprawled was everything
inside and beside it:

- `scripts/` held 57 flat files: vendor fetchers, panel builders, one-off
  experiments and scheduling tooling in one pile. It was the first folder a
  visitor opened, and the one that made the repo read as "no structure".
- `core/data/` held 57 files across five unrelated concerns: vendor clients,
  factor construction, universe eligibility, data-quality checks, and panel
  storage.
- Five `core/` packages existed for two to four files each (`replay`,
  `optimization`, `surfaces`, `utils`, `index`), so understanding one feature
  meant opening a package to find a single module inside it.
- The repository root had 15 folders, of which `_archive/`, `outputs/`,
  `results/`, `models/` and `logs/` were gitignored artifacts sitting as
  visible peers of `core/`.

An external review diagnosed this as a SOLID problem. It is not: this is a
data pipeline made of functions over DataFrames, which is the right design for
quant research, and per-class principles do not describe it. The concrete
failure was that a folder did not tell you what kind of thing it held.

Separately, `CLAUDE.md` capped modules at 300 lines. That cap and "fewer files
to trace" pull against each other: a cohesive 400-line module must be split
into fragments that only make sense together, which is exactly the
"trace ten files to understand one thing" cost the review complained about.

## Decision

1. **`scripts/` is grouped by what a script does to data**, in four folders
   whose names are the verbs:
   `ingest/` (writes the raw layer, incl. the ingestion supervisor),
   `build/` (rebuilds the derived layer), `experiments/` (one-off research,
   whose findings are recorded in `docs/`), `ops/` (crontab, launchd, watchdog,
   audits).
2. **`core/data/` is grouped by concern**: `vendors/` (the only code that
   talks to a provider), `factors/` (derived quantities), `universe/`
   (eligibility and identity), `quality/` (validation, quarantine, health,
   watchdog), `store/` (panel reads and writes, DuckDB).
3. **Tiny packages are folded into their natural homes**: replay and
   mean-variance into `core/backtest/`, option pricing into `core/metrics/`,
   index construction into `core/strategies/`, the Cid-1 study into
   `core/research/`, DuckDB helpers into `core/data/store/`, the ML results
   cache into `core/models/`, commodity ML features into `core/features/`.
4. **Run-time artifacts live under one gitignored `runtime/`**
   (`runtime/logs/`, `runtime/outputs/ml_results/`). `_archive/`, the empty
   `results/` and `models/`, and stray root logs are deleted. The root goes
   from 15 folders to 11.
5. **The module cap rises from 300 to 500 lines**, with the explicit rule to
   prefer one cohesive module over fragments a reader must trace across.
6. **`ARCHITECTURE.md` carries a one-screen directory map** with a "where does
   a new file go" rule, so the next file lands in the right place without a
   discussion.

Only files moved and references were rewritten; no function changed. The
commit that applies this decision lists every old → new path.

## Alternatives rejected

- **Leave the layout and add navigation docs.** Docs drift; a folder name
  is read every time, a doc once.
- **Apply SOLID-style decomposition** (interfaces per data source, one class
  per responsibility). Would multiply files and indirection for a codebase
  whose unit of work is a function returning a DataFrame; solves a problem the
  repo does not have.
- **A `src/` layout with an installable package.** Adds packaging ceremony
  without addressing the sprawl, which was within `core/data/` and
  `scripts/`, not at the import root.
- **Keep the 300-line cap and split more.** This is the mechanism that
  produced the fragmentation being fixed.

## Consequences

- Import paths under `core.data.*` changed for 34 modules; anything outside the
  repo that imported them (personal notebooks, shell aliases) must be updated.
- The launchd plist and crontab reference `scripts/ingest/` and
  `scripts/ops/`; both are re-rendered by the install scripts and were
  reinstalled with this change.
- Test files whose name was tied to a dissolved package were renamed to
  mirror the new module (`test_surfaces.py` → `test_metrics_vol_surface.py`,
  etc.).
