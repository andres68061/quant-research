# FMP — Historical Prices & Charts

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.
These endpoints are shared across asset classes: stocks (`AAPL`), indexes (`^GSPC`),
commodities (`GCUSD`), forex (`EURUSD`), crypto (`BTCUSD`). Date filtering via
`from`/`to` (YYYY-MM-DD).

## End-of-day (the key endpoints for this repo's price panel)

| Endpoint | Example | Description |
|---|---|---|
| `/historical-price-eod/light?symbol={s}` | `/historical-price-eod/light?symbol=AAPL` | Date, close, volume only |
| `/historical-price-eod/full?symbol={s}` | `/historical-price-eod/full?symbol=AAPL` | OHLC, volume, change, %change, VWAP |
| `/historical-price-eod/non-split-adjusted?symbol={s}` | — | Prices WITHOUT split adjustment |
| `/historical-price-eod/dividend-adjusted?symbol={s}` | — | Prices adjusted for dividends (total-return style) |

Migration note: the repo's `data/factors/prices.parquet` uses split+dividend
adjusted closes (yfinance `Adj Close` semantics). The closest FMP equivalent is
`historical-price-eod/dividend-adjusted`; validate a few symbols against the
current panel before switching (`/full` returns split-adjusted OHLC with a
separate `adjClose` field on some plans — inspect the payload).

## Intraday

| Endpoint | Example | Description |
|---|---|---|
| `/historical-chart/1min?symbol={s}` | `?symbol=AAPL` | 1-minute OHLCV bars (Ultimate for full history) |
| `/historical-chart/5min?symbol={s}` | — | 5-minute bars |
| `/historical-chart/15min?symbol={s}` | — | 15-minute bars |
| `/historical-chart/30min?symbol={s}` | — | 30-minute bars |
| `/historical-chart/1hour?symbol={s}` | — | 1-hour bars |
| `/historical-chart/4hour?symbol={s}` | — | 4-hour bars |
