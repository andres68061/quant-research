---
name: strategy-experiment-log
description: Use before starting a new strategy/formation/selection-criterion experiment in this repo (to check it hasn't already been tried and rejected), and after finishing one (to record the result in the right place). Triggers on requests like "try a new pairs formation approach", "test a different lookback window", "does X selection criterion work", or any variant of "how do we fix this strategy" for a strategy that already has entries in docs/FAILED_STRATEGIES_LOG.md.
---

# Strategy experiment log

This repo keeps two documents that must never be mixed:

- **`docs/ROADMAP.md`** — what we want to build next. Forward-looking only.
- **`docs/FAILED_STRATEGIES_LOG.md`** — what we tried and the real numbers
  showing it didn't work. Entries are never deleted, even after a
  replacement ships — a negative result is still evidence, and re-trying a
  logged failure wastes a full backtest cycle re-discovering it.

## Before starting a new strategy experiment

1. Read `docs/FAILED_STRATEGIES_LOG.md` and search for the strategy name
   (e.g. "pairs stat-arb index"). If a highly similar variant is already
   logged as failed, say so before spending compute re-running it — either
   skip it, or explain concretely why this attempt differs enough from the
   logged ones to be worth trying anyway.
2. Also skim `docs/ROADMAP.md` for the same strategy — there may already be
   a specific next-step plan recorded from a prior session.

## After finishing a strategy experiment

Update **exactly one** of the two documents before ending the turn:

- **Result is negative** (didn't beat the baseline, or performed worse):
  append a new dated entry to `docs/FAILED_STRATEGIES_LOG.md` under the
  relevant strategy's section (create the section if new). Include: what
  was tried, the concrete parameters, the real metric(s) (not just "it was
  worse" — the actual Sharpe/Pain Ratio/whatever numbers), and — if you can
  identify one — *why* it failed, not just that it did.
- **Result is positive** (ships as a real improvement): move/summarize the
  win into `docs/ROADMAP.md`'s "Recently shipped" section, and if a
  `docs/FAILED_STRATEGIES_LOG.md` entry for the superseded approach exists,
  add a one-line "superseded by ..." pointer at the top of that entry
  (still don't delete the entry itself).
- **Result is a scoped idea, not yet tried**: add it to `docs/ROADMAP.md`
  as a forward-looking item, not the failure log — the failure log is only
  for things actually run with real results.

Never leave a completed experiment undocumented in both files — if you ran
a real backtest with real numbers, one of the two docs must reflect it by
the end of the turn.
