---
name: research-disclosure
description: Use whenever you build or change anything that shows a research number to a human — a new API endpoint returning metrics, a new page/tab/chart, a generated report, a backtest surface, or a new index/aggregate. Enforces that every such surface declares its methodology and pulls its caveats from the single registry (core/research/caveats.py) instead of inventing an inline list or omitting them. Triggers on "add a page", "new endpoint returning returns/Sharpe/levels", "chart the performance of", "build an index", "report the results of", or any PR that renders a number a user could act on.
---

# Research disclosure

A number without its assumptions is worse than no number: it looks like
knowledge. This skill exists because caveats kept being written inline, in four
different files, each correct and none complete — so no surface could answer
"what should a reader of THIS number know?"

## The rule

**One registry, every caveat: `core/research/caveats.py`.**

A surface (page, endpoint, report) may not carry its own caveat prose. It
declares which surface it is and asks the registry:

```python
from core.research.caveats import SURFACE_PEAD, as_dicts, caveats_for_surface

return {..., "caveats": as_dicts(caveats_for_surface(SURFACE_PEAD))}
```

Adding a surface means adding its id to `ALL_SURFACES` and tagging the caveats
that apply. `caveats_for_surface` raises on an unknown id specifically so a typo
fails loudly rather than silently disclosing nothing.

## Checklist for any surface that shows a number

1. **Methodology block.** Alongside the data, return/render *how it was
   computed* — not just what. At minimum, whichever apply:
   - **rebalance / recompute cadence** (daily? monthly? event-driven? none?)
   - **weighting** (cap, equal, prior-day or same-day weights)
   - **universe and membership rule** (which symbols, filtered how, point-in-time?)
   - **costs** (gross? flat bps? liquidity-scaled? borrow?)
   - **return basis** (dividend-adjusted? total return? excess of what?)
   - **sample bounds** (period, minimum observations, what happens below them)
   - **how many symbols actually contributed** — the count, not the intent

2. **Caveats from the registry**, never inline. If the right caveat does not
   exist yet, add it to the registry with its `surfaces` tuple.

3. **Both directions.** State what the method *handles* as well as what it
   doesn't. "Survivorship-aware (dead names contribute until their last traded
   day)" is as important as "sector labels are current-only" — otherwise a
   reader cannot tell a considered choice from an oversight.

4. **Render it, don't just return it.** A caveat in a JSON payload that the page
   never displays is not a disclosure. Put it in the UI next to the number
   (a `<details>` block is enough; hidden-by-default is fine, absent is not).

5. **Numbers must explain themselves.** Any count shown to a user needs enough
   context to prevent the obvious misreading — what population it describes, and
   why it differs from the count above it. See `build_funnel` in
   `core/data/health.py` for the pattern (`scope` / `definition` / `why_smaller`).

## Anti-patterns

- Inline caveat lists in a route, script, or component.
- "Gross returns" stated in a docstring but not in the API response.
- A chart of index levels with no rebalance frequency anywhere on the page.
- Reporting a filtered count (929 labeled symbols) as if it were the universe
  (8,908) — always say which population a number describes.
- Adding a caveat to the registry but forgetting its `surfaces`, so it appears
  nowhere.
- Removing a caveat because it is inconvenient rather than because the
  underlying issue was fixed. If it was fixed, say so in the commit and, if
  methodologically interesting, write an ADR.

## Where the other records live (do not duplicate them here)

| Record | Purpose |
|---|---|
| `core/research/caveats.py` | **Caveats** — what a reader must know. The one registry. |
| `core/data/health.py::known_flaws` | **Measured data flaws** — recomputed from disk each audit. |
| `docs/DATA_HEALTH.md` | Rendered data state + flaws (generated; do not hand-edit the marked section). |
| `docs/decisions/` | **Why** an implementation choice was made, with alternatives rejected. |
| `docs/FAILED_STRATEGIES_LOG.md` | **Experiment outcomes** that were negative, with real numbers. |
| `docs/ROADMAP.md` | Forward-looking work only. |
