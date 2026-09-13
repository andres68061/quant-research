# FMP — News & Earnings Transcripts

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.
News endpoints support `page`/`limit`; search variants take `symbols=`.

## News feeds

| Endpoint | Example | Description |
|---|---|---|
| `/fmp-articles?page=0&limit=20` | — | FMP-authored articles |
| `/news/general-latest?page=0&limit=20` | — | General financial news |
| `/news/press-releases-latest?page=0&limit=20` | — | Latest company press releases |
| `/news/stock-latest?page=0&limit=20` | — | Latest stock-specific news |
| `/news/crypto-latest?page=0&limit=20` | — | Latest crypto news |
| `/news/forex-latest?page=0&limit=20` | — | Latest forex news |

## News search

| Endpoint | Example | Description |
|---|---|---|
| `/news/press-releases?symbols={s}` | `?symbols=AAPL` | Press releases for specific symbols |
| `/news/stock?symbols={s}` | `?symbols=AAPL` | Stock news for specific symbols |
| `/news/crypto?symbols={s}` | `?symbols=BTCUSD` | Crypto news by coin |
| `/news/forex?symbols={s}` | `?symbols=EURUSD` | Forex news by pair |

## Earnings transcripts (Ultimate plan)

| Endpoint | Example | Description |
|---|---|---|
| `/earning-call-transcript-latest` | — | Most recent transcripts across companies |
| `/earning-call-transcript?symbol={s}&year={y}&quarter={q}` | `?symbol=AAPL&year=2020&quarter=3` | Full transcript text for one call |
| `/earning-call-transcript-dates?symbol={s}` | — | Transcript dates by fiscal year/quarter |
| `/earnings-transcript-list` | — | All symbols with transcript counts |
