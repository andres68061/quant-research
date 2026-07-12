# FMP — Company Information

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

| Endpoint | Example | Description |
|---|---|---|
| `/profile?symbol={s}` | `/profile?symbol=AAPL` | Company profile: market cap, price, industry, sector, description |
| `/profile-cik?cik={cik}` | `/profile-cik?cik=320193` | Profile lookup by CIK |
| `/company-notes?symbol={s}` | — | Company-issued notes (CIK, title, exchange) |
| `/stock-peers?symbol={s}` | `/stock-peers?symbol=AAPL` | Peer companies in same sector / market-cap range |
| `/delisted-companies?page=0&limit=100` | — | Companies delisted from US exchanges (useful for survivorship-free panels) |
| `/employee-count?symbol={s}` | — | Latest employee count with SEC filing links |
| `/historical-employee-count?symbol={s}` | — | Employee count history by reporting period |
| `/market-capitalization?symbol={s}` | — | Market cap on a given date |
| `/market-capitalization-batch?symbols=A,B,C` | `?symbols=AAPL,MSFT,GOOG` | Market cap for multiple companies at once |
| `/historical-market-capitalization?symbol={s}` | — | Market cap time series (size-factor input) |
| `/shares-float?symbol={s}` | — | Free float / outstanding shares for one company |
| `/shares-float-all?page=0&limit=1000` | — | Float data for all companies |
| `/mergers-acquisitions-latest?page=0&limit=100` | — | Latest M&A transactions |
| `/mergers-acquisitions-search?name={n}` | `?name=Apple` | Search M&A by company name |
| `/key-executives?symbol={s}` | — | Executive names, titles, compensation |
| `/governance-executive-compensation?symbol={s}` | — | Detailed executive compensation from filings |
| `/executive-compensation-benchmark` | — | Average executive compensation by industry |
