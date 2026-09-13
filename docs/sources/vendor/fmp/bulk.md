# FMP — Bulk Endpoints

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.
Bulk endpoints return whole-universe data and generally require the **Ultimate** plan.
Some return CSV rather than JSON; check the payload.

| Endpoint | Example | Description |
|---|---|---|
| `/profile-bulk?part={n}` | `?part=0` | All company profiles (paged by `part`) |
| `/rating-bulk` | — | Ratings for all stocks |
| `/dcf-bulk` | — | DCF valuations for all stocks |
| `/scores-bulk` | — | Financial scores for all stocks |
| `/price-target-summary-bulk` | — | Price targets for all symbols |
| `/etf-holder-bulk?part={n}` | `?part=1` | All ETF holdings |
| `/upgrades-downgrades-consensus-bulk` | — | Analyst consensus for all symbols |
| `/key-metrics-ttm-bulk` | — | TTM key metrics, all companies |
| `/ratios-ttm-bulk` | — | TTM ratios, all companies |
| `/peers-bulk` | — | Peer lists for all stocks |
| `/earnings-surprises-bulk?year={y}` | `?year=2025` | Annual earnings surprises, all companies |
| `/income-statement-bulk?year={y}&period={p}` | `?year=2025&period=Q1` | Income statements, all companies |
| `/income-statement-growth-bulk?year={y}&period={p}` | — | Income statement growth, all companies |
| `/balance-sheet-statement-bulk?year={y}&period={p}` | — | Balance sheets, all companies |
| `/balance-sheet-statement-growth-bulk?year={y}&period={p}` | — | Balance sheet growth, all companies |
| `/cash-flow-statement-bulk?year={y}&period={p}` | — | Cash flows, all companies |
| `/cash-flow-statement-growth-bulk?year={y}&period={p}` | — | Cash flow growth, all companies |
| `/eod-bulk?date={d}` | `?date=2024-10-22` | EOD prices for ALL symbols on one date |

Migration note: `/eod-bulk` iterated over dates is the most efficient way to
rebuild the full `prices.parquet` panel (one call per trading day instead of one
call per symbol).
