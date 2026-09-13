---
name: data-inventory-sync
description: Use after adding, removing, or reshaping any data artifact, ingestion script, or vendor dependency in this repo — new parquet output, new columns in a factor panel, a new fetcher under scripts/, a changed cron job, or a change in what an API key is entitled to. Keeps docs/data/DATA_INVENTORY.md, docs/data/DATA_ARCHITECTURE.md and the vendor docs from drifting away from what is actually on disk. Triggers on "add a dataset", "new fetcher", "download X from FMP", "build a new panel", "add a factor column", "why does the doc say", or any commit that touches data/ layout or scripts/fetch_*.
---

# Data inventory sync

Data docs rot faster than code because nothing fails when they are wrong. This
checklist runs at the **end** of any change to what the repo stores or fetches.

## When this fires

Any of:

- A new parquet/DuckDB artifact appears under `data/`.
- An existing artifact gains, loses, or renames columns.
- A new fetcher lands in `scripts/` or a new module under `core/data/`.
- A cron entry changes.
- A vendor's plan or entitlements change (new subscription, endpoint starts 402ing).

## The rule that matters most

**Never write a vendor capability claim from memory or from the vendor's
marketing page.** Claims like "bulk requires Ultimate" or "we use the bulk
endpoint" must be traceable to a probe or to the code that makes the call.
For FMP, the probe is:

```bash
/opt/anaconda3/envs/quant/bin/python scripts/ingest/probe_fmp_entitlements.py --restricted-only
```

The stale "fundamentals come from FMP bulk" line in `DATA_INVENTORY.md` survived
for months while the code was making per-symbol calls and bulk was returning 402.
That is the failure mode this skill exists to prevent.

## Checklist

0. **Rerun the health audit — this is mandatory, not optional.**

   ```bash
   /opt/anaconda3/envs/quant/bin/python scripts/ops/audit_data_health.py
   ```

   This recomputes every number in `docs/data/DATA_HEALTH.md`'s generated section and
   `data/quality/data_health.json` (which the `/data-health` page serves) from
   the files on disk. A data change that ends without rerunning it leaves the
   health record describing a state that no longer exists — which is worse than
   no record, because it looks authoritative.

   If the change **introduced or fixed a flaw** (leakage, coverage gap, calendar
   anomaly, bias), edit the registry in `core/data/quality/health.py::known_flaws` —
   that one function feeds the doc, the API, and the frontend page. A fixed flaw
   is *removed* there and, if methodologically interesting, memorialized as an
   ADR instead. Chat is not a record; the registry is.

1. **Ground truth first.** List what is actually on disk before editing prose:

   ```bash
   du -sh data/*/ && ls data/factors/
   /opt/anaconda3/envs/quant/bin/python -c "
   import pandas as pd; d=pd.read_parquet('data/factors/<file>.parquet')
   print(d.shape); print(list(d.columns)); print(d.index.names)"
   ```

2. **`docs/data/DATA_INVENTORY.md`** — the artifact-level record:
   - §1 table: path, shape, producer script, whether the API loads it at startup.
   - §2: ingestion sources — script name **and** module path must both resolve.
   - §3: factor-family gap map, if a new factor family became available.
   - §4: cron, if scheduling changed. Mirror into `scripts/ops/crontab.txt`.
   - §6: vendor entitlements, if a probe result changed.

3. **`docs/data/DATA_ARCHITECTURE.md`** — only if the raw/derived boundary moved
   (a new raw layer, or a derived artifact that is now a source of truth).

4. **`docs/sources/vendor/<vendor>/`** — if endpoint behaviour differs from the snapshot,
   annotate the file rather than silently trusting the snapshot. Say when it was
   probed.

5. **Point-in-time vocabulary.** Any new dated column is named `reference_date`,
   `publication_date`, or `as_of_date` — never bare `date` in a new dataset. If a
   new artifact has no publication date, say so explicitly in the doc so nobody
   assumes it is PIT-safe.

6. **State the cost.** For a new fetcher record the call count and wall-clock of a
   full backfill. Future-you needs to know whether a rerun is 5 minutes or 5 hours.

## Anti-patterns

- Adding a row to §1 without checking the file's real shape.
- Documenting the artifact you intended to write rather than the one that exists.
- Leaving a "not yet wired into the pipeline" note in place after wiring it in.
- Describing an endpoint's availability without a probe date.
