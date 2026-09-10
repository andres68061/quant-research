# FMP — Economics, Commodities, Forex & Crypto

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

## Economics

| Endpoint | Example | Description |
|---|---|---|
| `/treasury-rates` | — | Treasury rates, all maturities (latest + historical) |
| `/economic-indicators?name={n}` | `?name=GDP` | GDP, unemployment, inflation, etc. |
| `/economic-calendar` | — | Upcoming economic data releases |
| `/market-risk-premium` | — | Market risk premium by date |

**Probed 2026-09-10 — both economics endpoints are capped at ~one quarter per
call regardless of the `from`/`to` window requested.** `treasury-rates` with
`from=2000-01-01&to=2009-12-31` returned 61 rows (2009-10-02 → 2009-12-31);
`economic-indicators?name=CPI&from=1990-01-01` returned 1 row. Full history would
need ~4 calls per year per series, which the ingestion engine's year-based
chunking does not express. The stored raw files
(`data/raw/fmp/treasury_rates/`, `data/raw/fmp/economic_indicators/`) are
therefore a **recent-quarter snapshot only**. **FRED is the history source for
rates and macro** (`core/data/factors/macro_catalog.py`, 38 series, full
history in one call each); see ADR 0018.

## Commodities

| Endpoint | Example | Description |
|---|---|---|
| `/commodities-list` | — | All tracked commodities (energy, metals, ag) |
| `/quote?symbol={s}` | `?symbol=GCUSD` | Real-time commodity quote (gold example) |
| `/quote-short?symbol={s}` | — | Compact commodity quote |
| `/batch-commodity-quotes` | — | All commodity quotes |
| `/historical-price-eod/{light,full}?symbol=GCUSD` | — | EOD commodity history |
| `/historical-chart/{1min,5min,1hour}?symbol=GCUSD` | — | Intraday commodity bars |

## Forex

| Endpoint | Example | Description |
|---|---|---|
| `/forex-list` | — | All currency pairs |
| `/quote?symbol=EURUSD` / `/quote-short?...` | — | Real-time forex quotes |
| `/batch-forex-quotes` | — | All forex quotes |
| `/historical-price-eod/{light,full}?symbol=EURUSD` | — | EOD forex history |
| `/historical-chart/{1min,5min,1hour}?symbol=EURUSD` | — | Intraday forex bars |

## Crypto

| Endpoint | Example | Description |
|---|---|---|
| `/cryptocurrency-list` | — | All cryptocurrencies |
| `/quote?symbol=BTCUSD` / `/quote-short?...` | — | Real-time crypto quotes |
| `/batch-crypto-quotes` | — | All crypto quotes |
| `/historical-price-eod/{light,full}?symbol=BTCUSD` | — | EOD crypto history |
| `/historical-chart/{1min,5min,1hour}?symbol=BTCUSD` | — | Intraday crypto bars |

Repo note: commodities pipeline (`scripts/ingest/fetch_commodities.py`,
`scripts/ingest/update_commodities.py`) and the FRED macro layer
(`scripts/ingest/fetch_raw_macro.py`) could source from these endpoints; FRED remains
preferable for macro series with proper publication-lag semantics.
