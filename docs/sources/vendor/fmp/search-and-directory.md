# FMP — Company Search & Stock Directory

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

## Company Search

| Endpoint | Example | Description |
|---|---|---|
| `/search-symbol?query={q}` | `/search-symbol?query=AAPL` | Find ticker symbols across global markets |
| `/search-name?query={q}` | `/search-name?query=AA` | Search by full/partial company or asset name |
| `/search-cik?cik={cik}` | `/search-cik?cik=320193` | Central Index Key lookup for SEC filings |
| `/search-cusip?cusip={cusip}` | `/search-cusip?cusip=037833100` | Security lookup by CUSIP |
| `/search-isin?isin={isin}` | `/search-isin?isin=US0378331005` | Security lookup by ISIN |
| `/company-screener` | `/company-screener` | Screen by market cap, price, volume, beta, sector, country, etc. |
| `/search-exchange-variants?symbol={s}` | `/search-exchange-variants?symbol=AAPL` | All exchanges where a symbol is listed |

## Stock Directory / Lists

| Endpoint | Example | Description |
|---|---|---|
| `/stock-list` | `/stock-list` | All available company symbols |
| `/financial-statement-symbol-list` | — | Companies with financial statements available |
| `/cik-list?page=0&limit=1000` | — | All SEC-registered CIK numbers |
| `/symbol-change` | — | Symbol changes (mergers, splits, renames) |
| `/etf-list` | — | All ETF symbols |
| `/actively-trading-list` | — | Currently actively trading instruments |
| `/earnings-transcript-list` | — | Companies with earnings transcripts + counts |
| `/available-exchanges` | — | All supported exchanges |
| `/available-sectors` | — | All sector labels |
| `/available-industries` | — | All industry labels |
| `/available-countries` | — | All countries with listed symbols |
