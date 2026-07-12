# FMP — Economics, Commodities, Forex & Crypto

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

## Economics

| Endpoint | Example | Description |
|---|---|---|
| `/treasury-rates` | — | Treasury rates, all maturities (latest + historical) |
| `/economic-indicators?name={n}` | `?name=GDP` | GDP, unemployment, inflation, etc. |
| `/economic-calendar` | — | Upcoming economic data releases |
| `/market-risk-premium` | — | Market risk premium by date |

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

Repo note: commodities pipeline (`scripts/fetch_commodities.py`,
`scripts/update_commodities.py`) and the FRED macro layer
(`scripts/fetch_raw_macro.py`) could source from these endpoints; FRED remains
preferable for macro series with proper publication-lag semantics.
