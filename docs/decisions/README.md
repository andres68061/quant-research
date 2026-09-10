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
- [0009](0009-cross-sectional-cid1-boundary-convention.md) — Cross-sectional Cid-1: fixed trailing window, +inf at pain = 0, atan2 for ranking
- [0010](0010-vendor-metric-point-in-time-classification.md) — Classify every vendor dataset by point-in-time status before use
- [0011](0011-piotroski-beginning-of-year-asset-scaling.md) — Scale Piotroski F-score flows by beginning-of-year assets
- [0012](0012-fallback-publication-lag-for-placeholder-filing-dates.md) — Substitute a 45-day filing lag when acceptedDate is a placeholder
- [0013](0013-canonical-panel-cutover-to-expanded-universe.md) — Cut the canonical panel over from 774 S&P names to the full US universe
- [0014](0014-lazy-factor-loading-and-api-universe-policy.md) — Load factor columns lazily; give the API an explicit universe policy
- [0015](0015-permanent-security-identifiers.md) — Permanent security identifiers (`qid`), separate from issuer identity
- [0016](0016-vendor-agnostic-ingestion-framework.md) — Replace per-dataset fetch scripts with a declarative manifest and a shared runner
- [0017](0017-repository-layout-by-kind.md) — Group scripts and data modules by kind, and raise the module cap to 500 lines
