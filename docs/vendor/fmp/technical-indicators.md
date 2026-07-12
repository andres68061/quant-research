# FMP — Technical Indicators

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

All indicator endpoints share the same parameter shape:
`?symbol={s}&periodLength={n}&timeframe={tf}` where `timeframe` is one of
`1min`, `5min`, `15min`, `30min`, `1hour`, `4hour`, `1day`.

| Endpoint | Indicator |
|---|---|
| `/technical-indicators/sma?symbol=AAPL&periodLength=10&timeframe=1day` | Simple Moving Average |
| `/technical-indicators/ema?...` | Exponential Moving Average |
| `/technical-indicators/wma?...` | Weighted Moving Average |
| `/technical-indicators/dema?...` | Double Exponential MA |
| `/technical-indicators/tema?...` | Triple Exponential MA |
| `/technical-indicators/rsi?...` | Relative Strength Index |
| `/technical-indicators/standarddeviation?...` | Rolling standard deviation |
| `/technical-indicators/williams?...` | Williams %R |
| `/technical-indicators/adx?...` | Average Directional Index |

Repo note: prefer computing indicators locally in `core/` from raw prices (keeps
one source of truth and avoids per-indicator API calls); use these endpoints only
for cross-checking or ad-hoc exploration.
