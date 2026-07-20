# Decision log (ADRs)

Short records of engineering and methodology decisions — the *why* behind
choices the code alone can't explain. Companion to the two research docs:

- `docs/FAILED_STRATEGIES_LOG.md` — strategy experiments that failed (with numbers)
- `docs/ROADMAP.md` — what to build next
- **this directory** — why implementations are shaped the way they are

A reviewer (or a future session) should be able to pick any non-obvious
design choice in `core/` or the toolchain and find its defense here.

## When to write one

Write an ADR when a decision (a) had a plausible alternative a reviewer
might expect, (b) is not derivable from the code or its tests, and (c) will
outlive the session that made it. Examples: choosing a statistical
correction, a default parameter with methodological consequences, a
toolchain scope, a dependency policy. Do **not** write ADRs for choices
that are pure convention (naming, formatting) or already covered by
CLAUDE.md, the failed-strategies log, or a docstring citation.

## Format

One file per decision: `NNNN-short-slug.md`, numbered sequentially.
Keep each under ~40 lines:

```markdown
# NNNN. Title (imperative: "Use X for Y")

Date: YYYY-MM-DD
Status: accepted | superseded by NNNN

## Context
What problem forced a choice, in 2-4 sentences.

## Decision
What we chose, precisely.

## Alternatives rejected
Each alternative + one sentence on why not.

## Consequences
What this commits us to; what would trigger revisiting.
```

Statuses are never deleted — a superseded ADR gets a pointer to its
replacement, same rule as the failed-strategies log.

## Index

- [0001](0001-record-architecture-decisions.md) — Record architecture decisions
- [0002](0002-purged-walk-forward-with-label-horizon.md) — Purge walk-forward training windows by label horizon
- [0003](0003-sidak-correction-for-grid-search.md) — Šidák trial-count correction for grid-searched significance tests
- [0004](0004-ci-scope-and-dependency-tiers.md) — CI scope and dependency tiers
- [0005](0005-newey-west-hac-for-alpha-inference.md) — Newey-West HAC errors for factor-alpha inference
- [0006](0006-flat-on-min-stocks-with-coverage-disclosure.md) — Go flat when min_stocks fails; disclose invested coverage
- [0007](0007-evidence-based-lifecycle-truncation.md) — Enforce lifecycle bounds only when backed by evidence
- [0008](0008-initial-formation-at-first-signal.md) — Form the initial portfolio at the first actionable signal
