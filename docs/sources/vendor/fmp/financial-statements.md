# FMP — Financial Statements, Metrics & Ratios

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.
Statement endpoints accept `period` (annual/quarter) and `limit` parameters.

## Core statements

| Endpoint | Example | Description |
|---|---|---|
| `/income-statement?symbol={s}` | `/income-statement?symbol=AAPL` | Income statements (annual/quarterly) |
| `/balance-sheet-statement?symbol={s}` | — | Balance sheets |
| `/cash-flow-statement?symbol={s}` | — | Cash flow statements |
| `/latest-financial-statements?page=0&limit=250` | — | Most recently filed statements across companies |
| `/income-statement-ttm?symbol={s}` | — | Trailing-twelve-month income statement |
| `/balance-sheet-statement-ttm?symbol={s}` | — | TTM balance sheet |
| `/cash-flow-statement-ttm?symbol={s}` | — | TTM cash flow |

## Metrics, ratios, scores

| Endpoint | Example | Description |
|---|---|---|
| `/key-metrics?symbol={s}` | — | Revenue, net income, P/E, and other key metrics |
| `/ratios?symbol={s}` | — | Profitability / liquidity / efficiency ratios |
| `/key-metrics-ttm?symbol={s}` | — | TTM key metrics |
| `/ratios-ttm?symbol={s}` | — | TTM ratios |
| `/financial-scores?symbol={s}` | — | Altman Z-Score, Piotroski Score |
| `/owner-earnings?symbol={s}` | — | Buffett-style owner earnings |
| `/enterprise-values?symbol={s}` | — | Enterprise value time series |

## Growth

| Endpoint | Description |
|---|---|
| `/income-statement-growth?symbol={s}` | Period-over-period income statement growth |
| `/balance-sheet-statement-growth?symbol={s}` | Balance sheet growth |
| `/cash-flow-statement-growth?symbol={s}` | Cash flow growth |
| `/financial-growth?symbol={s}` | Combined statement growth metrics |

## As-reported & filings-derived

| Endpoint | Example | Description |
|---|---|---|
| `/financial-reports-dates?symbol={s}` | — | Available report dates |
| `/financial-reports-json?symbol={s}&year={y}&period=FY` | `?symbol=AAPL&year=2022&period=FY` | Full 10-K as JSON |
| `/financial-reports-xlsx?symbol={s}&year={y}&period=FY` | — | Full 10-K as XLSX download |
| `/revenue-product-segmentation?symbol={s}` | — | Revenue by product line |
| `/revenue-geographic-segmentation?symbol={s}` | — | Revenue by region |
| `/income-statement-as-reported?symbol={s}` | — | Income statement exactly as filed |
| `/balance-sheet-statement-as-reported?symbol={s}` | — | Balance sheet as filed |
| `/cash-flow-statement-as-reported?symbol={s}` | — | Cash flow as filed |
| `/financial-statement-full-as-reported?symbol={s}` | — | All three statements as filed |
