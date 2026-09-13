# FMP — SEC Filings, Insider Trades & Form 13F

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

## SEC filings

| Endpoint | Example | Description |
|---|---|---|
| `/sec-filings-8k?from={d}&to={d}&page=0&limit=100` | — | Latest 8-K filings (material events) |
| `/sec-filings-financials?from={d}&to={d}&page=0&limit=100` | — | Latest financial-statement filings |
| `/sec-filings-search/form-type?formType=8-K&from={d}&to={d}` | — | Filings filtered by form type (10-K, 10-Q, 8-K...) |
| `/sec-filings-search/symbol?symbol={s}&from={d}&to={d}` | `?symbol=AAPL&from=2024-01-01&to=2024-03-01` | Filings for a symbol |
| `/sec-filings-search/cik?cik={cik}&from={d}&to={d}` | — | Filings by CIK |
| `/sec-filings-company-search/name?company={n}` | `?company=Berkshire` | Company search by name |
| `/sec-filings-company-search/symbol?symbol={s}` | — | Company search by symbol |
| `/sec-filings-company-search/cik?cik={cik}` | — | Company search by CIK |
| `/sec-profile?symbol={s}` | — | Full SEC company profile |
| `/standard-industrial-classification-list` | — | All SIC codes and titles |
| `/industry-classification-search` | — | Search SIC classifications |
| `/all-industry-classification` | — | SIC data for all companies |

## Insider trades

| Endpoint | Example | Description |
|---|---|---|
| `/insider-trading/latest?page=0&limit=100` | — | Latest insider transactions |
| `/insider-trading/search?page=0&limit=100` | — | Search insider trades (by symbol etc.) |
| `/insider-trading/reporting-name?name={n}` | `?name=Zuckerberg` | Trades by reporting person |
| `/insider-trading-transaction-type` | — | All transaction type codes |
| `/insider-trading/statistics?symbol={s}` | — | Aggregated insider buy/sell statistics |
| `/acquisition-of-beneficial-ownership?symbol={s}` | — | Ownership changes during acquisitions |

## Form 13F institutional ownership (Ultimate plan)

| Endpoint | Example | Description |
|---|---|---|
| `/institutional-ownership/latest?page=0&limit=100` | — | Latest 13F filings |
| `/institutional-ownership/extract?cik={cik}&year={y}&quarter={q}` | `?cik=0001388838&year=2023&quarter=3` | Raw holdings from a filing |
| `/institutional-ownership/dates?cik={cik}` | — | Filing dates for an institution |
| `/institutional-ownership/extract-analytics/holder?symbol={s}&year={y}&quarter={q}` | — | Holder-level analytics for a stock |
| `/institutional-ownership/holder-performance-summary?cik={cik}` | — | Institution performance vs benchmarks |
| `/institutional-ownership/holder-industry-breakdown?cik={cik}&year={y}&quarter={q}` | — | Institution's industry allocation |
| `/institutional-ownership/symbol-positions-summary?symbol={s}&year={y}&quarter={q}` | — | Institutional positions summary for a stock |
| `/institutional-ownership/industry-summary?year={y}&quarter={q}` | — | Industry-level ownership summary |
