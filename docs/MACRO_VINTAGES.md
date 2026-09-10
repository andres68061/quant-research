# Macro publication lag vs true vintages

_Last reviewed: 2026-09-10._

## What we ship today

Macro features use **fixed calendar-day publication lags** in
`MACRO_PUBLICATION_LAGS_DAYS` (`core/data/factors/macro.py`):

The catalog (`core/data/factors/macro_catalog.py::FRED_SERIES_CATALOG`, 38
series) holds each lag next to the release-calendar anchor that justifies it.
**The lag is measured from the FRED reference date**, which for a monthly
series is the *first day of the reference month*, so it covers the remainder
of that month plus the release delay:

| series_id | FRED id | Lag (days) | Anchor |
|---|---|---|---|
| `cpi_yoy`, `core_cpi_yoy` | CPIAUCSL, CPILFESL | 48 | CPI ~10th–15th of the following month |
| `unrate`, `payems` | UNRATE, PAYEMS | 40 | Employment Situation, first Friday |
| `fed_funds` | FEDFUNDS | 33 | monthly average, knowable once the month ends |
| `indpro_yoy`, `retail_sales_yoy` | INDPRO, RSAFS | 50 | ~15th–17th |
| `housing_starts` | HOUST | 53 | ~17th–19th |
| `pce_yoy` | PCEPI | 64 | last business day of the following month |
| `m2_yoy` | M2SL | 61 | fourth Tuesday |
| `umcsent` | UMCSENT | 62 | FRED receives it ~a month after the university |
| `initial_claims` | ICSA | 5 | weekly, Thursday for the prior Saturday |
| `fed_assets` | WALCL | 2 | weekly H.4.1 |
| `dollar_broad` | DTWEXBGS | 7 | weekly H.10 |
| `wti` | DCOILWTICO | 8 | EIA posts weekly, in arrears |
| Treasury curve, spreads, TIPS, breakevens, OAS, DFF, SOFR, VIX | DGS*, T10Y*, DFII*, T*YIE, BAA10Y, BAML*, DFF, SOFR, VIXCLS | 1 | daily market closes |

Until 2026-09-10 the table read `cpi_yoy: 30, unrate: 10, fed_funds: 5`,
written as days-after-release but applied to the reference date, so monthly
values were visible about a month early. See ADR 0018.

Pipeline:

1. Raw long panel (`data/raw/macro_fred.parquet`) stores **reference_date** values
   from current FRED (latest revised series).
2. `apply_macro_publication_lag` shifts each series by its fixed lag → conservative
   **as_of / publication proxy**.
3. Business-day forward-fill builds `data/factors/macro.parquet`.

`MACRO_USES_TRUE_VINTAGES = False`.

## What true vintages would mean

**ALFRED** (Archival FRED) keeps every published vintage of a series: the value
of “June CPI” as first released in July, then as revised in August, etc.

Without vintages, a backtest that loads today’s FRED history can see **revised**
CPI/UNRATE numbers that were not knowable on the historical as-of date — even
after applying a fixed lag. Fixed lags stop *calendar* leakage; they do not stop
*revision* leakage.

## Decision

For research overlays and regime notebooks, fixed lags are **good enough** and
already enforced. Full ALFRED ingestion is deferred until a strategy needs
revision-sensitive macro (e.g. nowcasting with first-print surprises).

When implementing vintages later:

- Store raw as `(series_id, reference_date, vintage_date, value)` under
  `data/raw/alfred/`.
- Join on `vintage_date <= as_of_date` (and prefer first print vs latest-as-of
  explicitly).
- Keep fixed-lag path as a fallback for series ALFRED does not cover.
