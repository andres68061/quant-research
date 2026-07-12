# Macro publication lag vs true vintages

_Last reviewed: 2026-07-12._

## What we ship today

Macro features use **fixed calendar-day publication lags** in
`MACRO_PUBLICATION_LAGS_DAYS` (`core/data/factors/macro.py`):

| series_id | FRED id | Lag (days) | Intent |
|---|---|---|---|
| `cpi_yoy` | CPIAUCSL | 30 | CPI usually mid-month after reference month |
| `unrate` | UNRATE | 10 | Employment situation ~first Friday |
| `fed_funds` | FEDFUNDS | 5 | Monthly average, slight delay |
| `dgs10` / `t10y2y` | DGS10 / T10Y2Y | 1 | Near daily market closes |

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
