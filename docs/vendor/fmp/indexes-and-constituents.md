# FMP — Indexes & Constituents

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

## Index quotes & prices

| Endpoint | Example | Description |
|---|---|---|
| `/index-list` | — | All stock market indexes (symbol, name, exchange, currency) |
| `/quote?symbol={s}` | `?symbol=^GSPC` | Real-time index quote |
| `/quote-short?symbol={s}` | — | Compact index quote |
| `/batch-index-quotes` | — | All index quotes in one call |
| `/historical-price-eod/light?symbol=^GSPC` | — | EOD index history (light) |
| `/historical-price-eod/full?symbol=^GSPC` | — | EOD index history (OHLCV) |
| `/historical-chart/{1min,5min,1hour}?symbol=^GSPC` | — | Intraday index bars |

## Constituents (critical for survivorship-free backtests)

| Endpoint | Description |
|---|---|
| `/sp500-constituent` | Current S&P 500 members |
| `/nasdaq-constituent` | Current Nasdaq members |
| `/dowjones-constituent` | Current Dow Jones members |
| `/historical-sp500-constituent` | Historical S&P 500 additions/removals with dates |
| `/historical-nasdaq-constituent` | Historical Nasdaq changes |
| `/historical-dowjones-constituent` | Historical Dow changes |

Repo note: `core/data/sp500_constituents.py` + `sp500_universe_filter()` implement
point-in-time membership. `historical-sp500-constituent` is the FMP source to
rebuild/verify that membership table during the data migration.
