# 0001. Record architecture decisions

Date: 2026-07-19
Status: accepted

## Context

An external code review (2026-07-19) noted that reviewers of an
AI-assisted repo will probe whether every design decision can be defended
without the tooling. The repo already preserves *strategy* rationale
(`docs/FAILED_STRATEGIES_LOG.md`) but engineering/methodology rationale
lived only in session transcripts and commit messages, which don't survive
or aren't searchable.

## Decision

Keep a lightweight ADR log in `docs/decisions/`, one file per decision,
following the format in that directory's README. The `decision-log` skill
(`.claude/skills/decision-log/`) instructs agent sessions to check the log
before revisiting a settled question and to append an ADR when they make a
qualifying decision.

## Alternatives rejected

- **Rationale only in docstrings** — good for local "why", but cross-cutting
  decisions (toolchain scope, dependency policy) have no single code home.
- **Rationale in CLAUDE.md** — CLAUDE.md states rules; it would bloat
  unreadably if it also carried every rule's derivation.
- **Enforcement via a git hook/CI check** — "did this change embody a
  design decision?" is a judgment call no mechanical check can make; a
  skill + convention is the honest enforcement level (same as the
  strategy-experiment-log).

## Consequences

Non-obvious choices get a durable, linkable defense. Cost: a few minutes
per decision. Revisit if the log stops being consulted (dead process is
worse than no process).
