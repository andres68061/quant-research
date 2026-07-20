# 0007. Enforce lifecycle bounds only when backed by evidence

Date: 2026-07-19
Status: accepted

## Context

`symbol_lifecycle.parquet` windows are rebuilt monthly but re-applied to the
price panel after every daily update. ~97% of windows carry only a
`price_span` note: their `valid_to` is just the panel's last date on the day
the windows were built. Re-applying those stale caps deleted every price
fetched after the build — 2026-07-13→17 lost all but 1 of 774 symbols, which
backtests then read as a mass delisting (a fake -100% for long-only books).

## Decision

`apply_lifecycle_to_panel` enforces a bound only when `source_notes` shows
real lifecycle evidence for it: `ipoDate` for `valid_from`; `delistedDate`,
`symbol_change`, or `price_gap` for `valid_to`. A `price_span`-only window is
a no-op. Guarded by `update_prices` refusing to extend the panel when a new
date has < 50% of the trailing-median symbol coverage.

## Alternatives rejected

- **Rebuild windows on every price update** — a full FMP registry refetch per
  day for information that changes ~monthly; and a build against an
  already-damaged panel bakes the damage into the windows.
- **Cap `valid_to` at build time to `NaT` unless evidenced** — same semantics
  but requires a migration of the stored parquet and touches every consumer;
  enforcing at apply time fixes existing files retroactively.
- **Drop lifecycle re-application entirely** — loses the ticker-reuse and
  post-delisting truncation guards, which are the feature's whole point.

## Consequences

Ticker-reuse / delisting truncation still applies, but a symbol whose real
delisting FMP never records will keep its trailing prices (acceptable: the
quality scan and staleness cap catch dead series). Revisit if window rebuilds
become cheap enough to run daily against a validated panel.
