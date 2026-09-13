# Monitoring — how a data problem reaches you

The question this answers: *"we should be able to prevent stuff from harming
important stuff, flag it, have a process to fix it, and inform me it happened —
but I don't read logs."*

There are four layers, and they are deliberately different from each other. Each
catches a failure the others structurally cannot.

## The four layers

| Layer | When it runs | Catches | Module |
|---|---|---|---|
| **1. Write guard** | during a build | a partial or collapsing write reaching a production panel | `core/data/store/artifacts.py` |
| **2. Structural validation** | at build + audit | impossible values (a $0.00 share, an infinite return) | `core/data/quality/validation.py` |
| **3. Quarantine scan** | on demand | *suspicious* symbols needing human judgement | `core/data/quality/quarantine.py` |
| **4. Watchdog** | every 4h + daily | everything that goes wrong **between** runs | `core/data/quality/watchdog.py` |

Layer 4 is the one that was missing. Layers 1–3 all fire *while something is
running*. None of them notices a cron job that silently stopped firing, a panel
that stopped advancing, or a rebuild the OS killed halfway — failures whose
signature is that **nothing happens**, and which leave the platform serving stale
or truncated data that looks completely normal.

## What the watchdog checks

- **Panels present and readable** — a truncated write leaves a file that exists
  but has no parquet footer.
- **Panels not collapsed** — row counts against the last clean baseline. This is
  the between-runs counterpart to the write guard: it catches damage done by
  something that never went through `core.data.store.artifacts` at all.
- **Panel freshness** — the canonical price panel is still advancing with the
  market (tolerance 5 days, so a long weekend plus a holiday is fine).
- **Scheduled jobs still firing** — each cron log was touched recently and does
  not end in a traceback.
- **Structural invariants** (`--deep`, daily) — the full `validate_price_panel`
  scan.

## How it reaches you

Three channels, increasingly intrusive:

1. `data/quality/watchdog_status.json` — machine-readable, served by
   `GET /watchdog`.
2. **A banner on every page in the app**, red for errors and amber for warnings,
   expandable to the failing checks. This is the channel that replaces reading a
   log: it is where you already are.
3. **A macOS notification, on error only.** Active — it arrives whether or not
   the app is open.

Plus a non-zero exit code, so any wrapping job fails loudly too.

## Why the baseline only advances on a clean run

`run_watchdog.py` records current row counts as the new reference **only when the
run passes**. If a panel collapses, the collapse stays visible on every
subsequent run instead of quietly becoming the new normal. That single rule is
what stops a monitoring system from ratifying the damage it was meant to catch.

## Why some findings are deliberately not alerts

`ACKNOWLEDGED_INVARIANTS` marks conditions that are permanently true *by design*
— the 1,826 extreme returns in the raw price panel are expected under the ADR
0012 fidelity rule, since the raw layer keeps vendor values verbatim and
`core.data.factors.returns` rejects them at compute time.

They report as `ok` with the reason attached, not as warnings. **A monitor that
is always yellow is a monitor nobody reads**, and these are already disclosed in
the caveat registry rather than being news. Adding an entry there is a decision
to be made deliberately, not a way to silence something inconvenient.

## Schedule

Installed from `scripts/ops/crontab.txt`:

```
0 */4 * * *   run_watchdog.py           fast checks, every 4 hours
40 18 * * *   run_watchdog.py --deep    after the 6pm update jobs land
```

The deep run is at 18:40 specifically so it validates what that evening's update
just wrote, rather than yesterday's artifacts.

**To install:** `crontab scripts/ops/crontab.txt`
**To set a fresh baseline after an intentional change:**
`python scripts/ops/run_watchdog.py --set-baseline`

Until the cron entry exists, the watchdog reports a warning about itself
(`job:watchdog.log — no log yet`), which is correct: an uninstalled monitor is a
finding.
