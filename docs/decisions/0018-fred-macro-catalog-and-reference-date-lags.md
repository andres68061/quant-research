# 0018. FRED is the source for rates and macro; publication lags count from the FRED reference date

Date: 2026-09-10
Status: accepted

## Context

Two things were true at once about the macro layer.

**It was thin.** `data/raw/macro_fred.parquet` held five series (CPI YoY,
unemployment, fed funds, 10Y yield, 10Y–2Y spread). There was no yield curve
to draw - one tenor and one spread - and nothing on real rates, breakevens,
credit, claims, payrolls, activity, money, or the dollar. The FMP
`treasury-rates` and `economic-indicators` endpoints had been ingested as if
they were the answer, but a probe on 2026-09-10 showed both return at most
about one quarter per call whatever `from`/`to` window is requested
(`treasury-rates` for 2000–2009 returned 61 rows ending 2009-12-31; CPI from
1990 returned one row). What was on disk was a three-month snapshot.

**Its lags were wrong.** FRED stamps a monthly value on the first day of the
reference month: June CPI sits on `2026-06-01`. The lag table
(`cpi_yoy: 30, unrate: 10, fed_funds: 5`) was written as "days after
release" but applied to that reference date, so June CPI became visible on
July 1 (released ~July 12), June unemployment on June 11 (released ~July 3),
and the June fed-funds monthly average on June 6, before the month had
ended. Every consumer of `macro.parquet` - the regime HMM's five macro
z-scores in particular - saw macro data roughly a month before it existed.
`docs/sources/MACRO_VINTAGES.md` stated the right intent ("CPI usually mid-month after
reference month") next to the wrong arithmetic; nothing checked one against
the other.

## Decision

1. **FRED is the history source for rates and macro.** The catalog moves to
   `core/data/factors/macro_catalog.py` and grows to 38 series: the
   constant-maturity Treasury curve (1M–30Y), 10Y–2Y and 10Y–3M, TIPS 5Y/10Y,
   5Y/10Y breakevens, Baa–10Y, ICE BofA HY and IG OAS, fed funds (monthly and
   daily), SOFR, CPI / core CPI / PCE YoY, unemployment, payrolls, initial
   claims, industrial production and retail sales YoY, housing starts,
   Michigan sentiment, M2 YoY, Fed total assets, broad dollar, VIX, WTI. One
   FRED call per series returns full history; the whole refetch is ~30 s.
   The FMP economics raw files stay on disk, documented as a snapshot.

2. **Each catalog entry carries its own metadata**: FRED id, group,
   frequency, publication lag, transform (`level` or `yoy_pct`), unit. The
   derivation code reads the transform from the catalog instead of
   special-casing CPI by name. `DEFAULT_FRED_SERIES_MAP` and
   `MACRO_PUBLICATION_LAGS_DAYS` remain as views of the catalog so nothing
   downstream changes its imports.

3. **Publication lag is defined as calendar days from the FRED reference
   date to the first day the value could have been known.** For monthly
   series that is the remainder of the reference month plus the release day
   plus a few days of slack: CPI 48, unemployment and payrolls 40, fed funds
   monthly average 33, industrial production and retail sales 50, housing
   starts 53, PCE 64, M2 61, Michigan sentiment 62 (FRED receives it a month
   after the university publishes). Daily market series are 1 except the
   broad dollar (7, weekly H.10 release) and WTI (8, EIA posts weekly). The
   anchors are written next to the numbers in the catalog so the next
   reader can check them against a release calendar.

4. **YoY series are stored as fractions** (0.033 = 3.3%) with unit `ratio`,
   so nothing mistakes the column for percent.

## Alternatives rejected

- **Teach the ingestion engine quarter-sized windows to get FMP history.**
  ~100 calls per series for data FRED gives in one, and the engine's
  `date_chunk_years` is deliberately integer years. Not worth an engine
  change for a worse source.
- **Keep the old lags and document them as approximate.** They were not
  approximate; they were early by a month, which is lookahead. A leak that
  is documented is still a leak.
- **Switch to ALFRED vintages now.** The right end state (a true
  `publication_date` per observation, with revisions) but a separate
  project; `MACRO_USES_TRUE_VINTAGES` stays False and the monitor discloses
  it (`macro-latest-revision-only` caveat).
- **Shift the reference date to month-end instead of changing the lags.**
  Would hide the fact that FRED's convention is the first of the month, and
  every notebook that reads the raw layer would need the same shift.

## Consequences

- `macro.parquet` and `macro_z.parquet` go from 5 to 38 columns and every
  monthly series moves later by roughly a month. Any result that used the
  old regime-HMM features should be re-run before it is cited; the direction
  of the change is toward less information, so a result that survives is
  stronger.
- The two ICE BofA OAS series only reach back to 2023-09 under FRED's
  current licence; the monitor discloses this (`oas-series-start-2023`).
- A test now asserts that every monthly lag exceeds 31 days, so the
  arithmetic cannot silently regress to the old convention.
