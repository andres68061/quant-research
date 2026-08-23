# Documentation index — start here

There are 30+ documents in this directory. This page exists so you never have to
guess which one answers your question, and so nothing gets written in two places.

## The four records that matter

Every fact about the platform's integrity lives in exactly one of these. If you
are about to write a caveat, a flaw, a decision, or a result — this table tells
you where it goes, and the other three are wrong by definition.

| Question | Record | Source of truth |
|---|---|---|
| **"What must I know before trusting this number?"** | [DATA_HEALTH.md](DATA_HEALTH.md) | `core/research/caveats.py` (caveats, all surfaces) + `core/data/health.py::known_flaws` (measured flaws) |
| **"Why is it implemented this way?"** | [decisions/](decisions/) | One ADR per decision, with alternatives rejected |
| **"Did we already try this and fail?"** | [FAILED_STRATEGIES_LOG.md](FAILED_STRATEGIES_LOG.md) | Real numbers from real runs; entries are never deleted |
| **"What are we building next?"** | [ROADMAP.md](ROADMAP.md) | Forward-looking only; no results |
| **"What did that experiment actually show?"** | `/research-notes` | `core/research/notes.py` — every variant plus its control; figures asserted against the experiment JSON by `tests/test_notes.py` |
| **"What does this word mean?"** | `/glossary` | `core/research/glossary.py` — the single definition registry, read by every surface |

### Why there are two research records

The failure log is the **ledger**: one line per outcome, never deleted, so a dead
end stays dead. A research note is the **argument**: the full variant table, the
control it must be read against, what it means, and what would change it. The log
answers "have we tried this?"; the note answers "and what did we learn?". A note
always names the log or roadmap entry it corresponds to.

### Why caveats are in code, not prose

Caveats used to be written inline wherever a number was computed — four
locations, each correct, none complete. They now live in a single registry
(`core/research/caveats.py`) tagged with the **surfaces** they apply to, so:

- every page/endpoint asks `caveats_for_surface(...)` and cannot forget one,
- the same caveat text appears identically on the sector page, the PEAD page,
  and in DATA_HEALTH.md,
- adding a research surface without disclosures fails the
  `research-disclosure` skill's checklist.

**Measured** flaws (coverage gaps, vendor defects, calendar anomalies) are
different: they are recomputed from disk by `scripts/audit_data_health.py` on
every run, because asserting them in prose is how docs go stale. Both streams
render into DATA_HEALTH.md and the `/data-health` page.

## Live surfaces

| Page | What it answers |
|---|---|
| `/explorer` | **"What do we have on this company?"**, "screen the universe", "run a query". Company profile (identity, permanent id, index history, every factor, price chart), structured screener that shows its generated SQL, and a read-only SQL console |
| `/research-notes` | What each experiment found — every variant, its control, the verdict, and how to re-run it |
| `/glossary` | Every term the platform uses, defined to stand alone, cross-linked |
| `/data-health` | Universe funnel with definitions, survivorship, dataset coverage, flaw + caveat registry, per-symbol drilldown, glossary |
| `/methodology` | Every formula, each with a plain-language reading beside it |
| `/sectors` → Performance | Point-in-time sector indices with full methodology disclosure |
| `/pead` | Post-earnings drift event study |
| `/data-coverage` | Dataset inventory and quarantine review |

A **watchdog banner** appears on every page whenever the scheduled data checks
find a problem, so a failure does not depend on anyone reading a log.

## Data documents

| Document | Scope |
|---|---|
| [DATA_INVENTORY.md](DATA_INVENTORY.md) | **What artifacts exist** and which script produces each |
| [DATA_MODEL.md](DATA_MODEL.md) | **How they are organised** — panel families, raw vs derived, fact vs interval tables |
| [DATA_HEALTH.md](DATA_HEALTH.md) | **How good they are** — coverage, bias, flaws (generated section) |
| [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md) | Raw vs derived layering |
| [MONITORING.md](MONITORING.md) | **How a data problem reaches you** — the four guard layers, the watchdog, alerting |
| [SP500_MEMBERSHIP.md](SP500_MEMBERSHIP.md) | Index membership sourcing |
| [vendor/fmp/](vendor/fmp/) | Endpoint catalog + probed entitlements |

## Research output

| Document | Scope |
|---|---|
| [research/](research/) | Generated experiment reports (factor screens etc.), dated |
| [FACTOR_BACKTEST_AUDIT.md](FACTOR_BACKTEST_AUDIT.md) | Methodology audit of the factor pipeline |
| [MACRO_VINTAGES.md](MACRO_VINTAGES.md) | Point-in-time macro handling |

## Skills that enforce all of this

| Skill | Fires when |
|---|---|
| `research-disclosure` | Any surface that shows a research number |
| `strategy-evaluation` | Judging whether a signal/factor/strategy is any good |
| `explain-in-context` | Any user-facing explanation (self-contained, no orphan jargon) |
| `data-inventory-sync` | Any data artifact, fetcher, or entitlement change |
| `decision-log` | Any non-trivial engineering/methodology decision |
| `strategy-experiment-log` | Before and after any strategy experiment |
