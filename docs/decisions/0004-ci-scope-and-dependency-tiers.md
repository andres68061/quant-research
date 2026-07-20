# 0004. CI scope and dependency tiers

Date: 2026-07-19
Status: accepted

## Context

Adding GitHub Actions required deciding what "green" means. The repo's
full `requirements.txt` includes TensorFlow (~600 MB, used only by the
LSTM model, imported lazily, zero test coverage) and previously ta-lib
(C-library build pain, imported nowhere but an env-check script). Lint
also failed wholesale on `_archive/` (frozen pre-migration code) and
`notebooks/`.

## Decision

Three dependency tiers: `requirements.txt` (floors, full dev env),
`requirements-ci.txt` (test subset, no TF/jupyter, lint tools pinned to
the local versions), `requirements.lock.txt` (full `pip freeze` snapshot
for reproducibility claims). ta-lib removed entirely. Lint tools exclude
`_archive/` and `notebooks/` via `pyproject.toml` so local `make lint`
and CI check the same surface: everything actively maintained.

## Alternatives rejected

- **CI installs full requirements.txt** — minutes of TF install to test
  code no test imports.
- **Deleting `_archive/`** — it documents the migration; excluding beats
  rewriting dead code to satisfy a linter.
- **pip-tools/uv compiled lockfile** — better long-term, but the conda
  env is the actual source of truth here; a freeze snapshot is honest
  about that. Revisit if the project moves off conda.

## Consequences

CI is fast and deterministic; the lockfile pins what "worked on my
machine" means. Cost: `requirements-ci.txt` floors must be kept in sync
with `requirements.txt` by hand (noted in its header). LSTM remains
untested in CI by design — if it gains tests, TF joins the CI tier.
