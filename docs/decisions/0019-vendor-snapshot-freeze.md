# 0019. When a paid vendor lapses, freeze its raw layer as a dated snapshot

Date: 2026-09-12
Status: accepted

## Context

The FMP subscription ended in September 2026. Its raw layer on disk is
16 GB across 836k files: the 8,900-symbol price panel, statements and
ratios, commodities, VIX, sector labels, index membership - everything the
cross-sectional research runs on, complete through 2026-09-09.

For three nights the scheduled jobs failed at the first FMP call. Because
the price step ran first and raised, every step after it - FRED macro,
Fama-French, VIX, DuckDB views - was skipped too, and the watchdog went red
each night for a failure nobody could act on. Left alone, that trains the
reader to ignore the watchdog, which is the one outcome monitoring must not
produce.

The question was whether to replace the feed (yfinance), renew, or stop.

## Decision

**The FMP raw layer is frozen as a dated snapshot, `FMP_SNAPSHOT_AS_OF =
2026-09-09`, and the platform treats that as a first-class state**, not a
broken feed:

- `config/settings.py` exposes `FMP_SNAPSHOT_AS_OF` (from `.env`) and
  `FMP_ENABLED`. Every FMP step in the scheduled jobs checks the flag and
  skips with a message; FRED, Fama-French and market caps keep running.
- The three FMP-only cron lines (commodities, wave-1 nightly, wave-2 weekly)
  are commented out in `scripts/ops/crontab.txt` with the reason.
- The watchdog judges the price panel and every FMP-sourced series against
  the snapshot date rather than the wall clock: a panel that stops on the
  declared date is healthy; one that stops short of it is an error.
- The Data Monitor labels frozen series and shows the snapshot date.
- One caveat, `fmp-raw-layer-frozen-2026-09-09`, is registered on every
  research surface, so no number is shown without the reader knowing the
  data does not advance.

The reasoning, not just the mechanics: **for cross-sectional and anomaly
research a frozen, dated snapshot is better than a live feed.** Every
result becomes exactly reproducible against an immutable raw layer, and the
freeze date is an honest holdout - anything found on data through
2026-09-09 can be tested out-of-sample on a later refetch of the months
after it. The only research role of live data is that growing
out-of-sample window, which is a slow cost, not a nightly one.

## Alternatives rejected

- **Switch daily feeds to yfinance.** A second, lower-quality vendor to
  reconcile against the first (different adjustment conventions, no
  point-in-time fundamentals, silent symbol changes) for data the research
  does not need day-to-day.
- **Leave the jobs failing.** Nightly red for an unactionable failure is
  how monitoring gets ignored.
- **Delete the FMP jobs and code.** They are correct and will be needed the
  day a vendor is live again; a flag that skips is cheaper than a rewrite.
- **Renew now.** Nothing currently planned needs the feed; renew when an
  out-of-sample test is due, and refetch only the gap.

## Consequences

- `data/factors/prices.parquet` ends 2026-09-09; commodities 2026-09-11;
  fundamentals at the last wave-2 run. Backtests must not extend past the
  snapshot, and any walk-forward re-evaluation cadence stops there.
- The watchdog's equity-panel freshness check is only meaningful again once
  `FMP_SNAPSHOT_AS_OF` is unset; unsetting it is the one-line way to go
  live.
- Re-enabling: set a key, remove `FMP_SNAPSHOT_AS_OF`, uncomment the three
  cron lines, run `scripts/ingest/backfill_expanded_universe.py` for the
  gap, and re-run the health audit.
