# FMP — Quotes (Real-Time)

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

The `/quote` and `/quote-short` endpoints are shared across asset classes: stocks
(`AAPL`), indexes (`^GSPC`), commodities (`GCUSD`), forex (`EURUSD`), crypto (`BTCUSD`).

## Single-symbol

| Endpoint | Example | Description |
|---|---|---|
| `/quote?symbol={s}` | `/quote?symbol=AAPL` | Full real-time quote: price, change, volume, day range |
| `/quote-short?symbol={s}` | — | Compact quote: price, change, volume |
| `/aftermarket-trade?symbol={s}` | — | Post-market trades: price, size, timestamp |
| `/aftermarket-quote?symbol={s}` | — | Post-market bid/ask and volume |
| `/stock-price-change?symbol={s}` | — | % change over 1D/5D/1M/3M/6M/YTD/1Y/3Y/5Y/10Y/max |

## Batch

| Endpoint | Example | Description |
|---|---|---|
| `/batch-quote?symbols=A,B` | `?symbols=AAPL,MSFT` | Full quotes for multiple symbols |
| `/batch-quote-short?symbols=A,B` | — | Short quotes for multiple symbols |
| `/batch-aftermarket-trade?symbols=A,B` | — | Aftermarket trades, multiple symbols |
| `/batch-aftermarket-quote?symbols=A,B` | — | Aftermarket quotes, multiple symbols |
| `/batch-exchange-quote?exchange={x}` | `?exchange=NASDAQ` | All quotes on an exchange |
| `/batch-mutualfund-quotes` | — | All mutual fund quotes |
| `/batch-etf-quotes` | — | All ETF quotes |
| `/batch-commodity-quotes` | — | All commodity quotes |
| `/batch-crypto-quotes` | — | All crypto quotes |
| `/batch-forex-quotes` | — | All forex pair quotes |
| `/batch-index-quotes` | — | All index quotes |
