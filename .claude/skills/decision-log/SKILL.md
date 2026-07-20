---
name: decision-log
description: Use when making or revisiting a non-trivial engineering/methodology decision in this repo — choosing between statistical methods or corrections, setting a default parameter with methodological consequences, changing toolchain/CI/dependency policy, or picking an algorithm where a reviewer would expect an alternative. Check docs/decisions/ BEFORE re-deciding a settled question; append an ADR AFTER deciding. Triggers on "why did we choose X", "should we switch to Y", "add/replace a library", "change the default of", or any design choice that needs defending later.
---

# Decision log (ADRs)

`docs/decisions/` holds one short file per engineering/methodology
decision — the *why* the code can't carry. Format and index live in
`docs/decisions/README.md`. This is the engineering counterpart of
`docs/FAILED_STRATEGIES_LOG.md` (which stays strictly for strategy
experiments with backtest numbers — never mix the two).

## Before deciding

1. Search `docs/decisions/` for the topic. If an accepted ADR already
   covers it, follow it — or, if there's a real reason to deviate, say so
   explicitly and write a superseding ADR rather than silently diverging.
2. Also check CLAUDE.md and `docs/FAILED_STRATEGIES_LOG.md`; some
   "decisions" are actually conventions or empirical results already
   settled there.

## After deciding

If the decision (a) had a plausible alternative a reviewer might expect,
(b) isn't derivable from code/tests alone, and (c) outlives this session,
then before ending the turn:

1. Add `docs/decisions/NNNN-short-slug.md` (next number in sequence)
   using the template in `docs/decisions/README.md`: Context, Decision,
   **Alternatives rejected** (each with one honest sentence — this section
   is the point), Consequences (including what would trigger revisiting).
2. Add the entry to the index in `docs/decisions/README.md`.
3. Keep it under ~40 lines. If you can't name a rejected alternative, it
   probably wasn't a decision — skip the ADR.

When superseding: mark the old ADR `Status: superseded by NNNN`, never
delete it.

## What does NOT get an ADR

Pure convention (naming, formatting), anything CLAUDE.md already mandates,
strategy experiment outcomes (those go through the `strategy-experiment-log`
skill), and one-off implementation details a docstring citation covers.
