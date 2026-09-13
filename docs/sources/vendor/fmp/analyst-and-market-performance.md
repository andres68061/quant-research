# FMP — Analyst Data & Market Performance

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

## Analyst estimates, ratings, targets, grades

| Endpoint | Example | Description |
|---|---|---|
| `/analyst-estimates?symbol={s}&period=annual&page=0&limit=10` | — | Analyst revenue/EPS forecasts |
| `/ratings-snapshot?symbol={s}` | — | Current financial rating from key ratios |
| `/ratings-historical?symbol={s}` | — | Rating history by date |
| `/price-target-summary?symbol={s}` | — | Average price targets across timeframes |
| `/price-target-consensus?symbol={s}` | — | High/low/median/consensus targets |
| `/grades?symbol={s}` | — | Latest analyst upgrades/downgrades |
| `/grades-historical?symbol={s}` | — | Grade change history |
| `/grades-consensus?symbol={s}` | — | Buy/hold/sell rating counts |

## Sector & industry performance

| Endpoint | Example | Description |
|---|---|---|
| `/sector-performance-snapshot?date={d}` | `?date=2024-02-01` | Sector performance on a date |
| `/industry-performance-snapshot?date={d}` | — | Industry performance on a date |
| `/historical-sector-performance?sector={x}` | `?sector=Energy` | Sector performance time series |
| `/historical-industry-performance?industry={x}` | `?industry=Biotechnology` | Industry performance time series |
| `/sector-pe-snapshot?date={d}` | — | Sector P/E ratios on a date |
| `/industry-pe-snapshot?date={d}` | — | Industry P/E ratios on a date |
| `/historical-sector-pe?sector={x}` | — | Sector P/E time series |
| `/historical-industry-pe?industry={x}` | — | Industry P/E time series |

## Movers

| Endpoint | Description |
|---|---|
| `/biggest-gainers` | Largest daily price gainers |
| `/biggest-losers` | Largest daily price losers |
| `/most-actives` | Highest-volume stocks |
