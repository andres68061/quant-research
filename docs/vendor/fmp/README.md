# FMP (Financial Modeling Prep) API Reference

Local snapshot of the FMP **stable** API documentation, organized by category for
use by agents and skills. Source: [FMP docs](https://site.financialmodelingprep.com/developer/docs/stable),
snapshotted 2026-07-11. Descriptions are FMP's; per-endpoint parameter tables are
JS-rendered on the site and not captured here — when exact parameters matter,
verify against the live docs page or probe the endpoint.

## Basics

- **Base URL**: `https://financialmodelingprep.com/stable/`
- **Auth**: append `?apikey=$FMP_API_KEY` (or `&apikey=` if other params exist), or send header `apikey: $FMP_API_KEY`.
- **Key storage**: the key lives in `.env` as `FMP_API_KEY` and is read via `config/settings.py` (`FMP_API_KEY`). Never hardcode it in code, notebooks, or docs.
- **Common params**: `symbol`, `symbols` (comma-separated), `from`/`to` (YYYY-MM-DD), `page`/`limit`, `period` (`annual`/`quarter`/`FY`/`Q1`...), `cik`, `exchange`.

## Error codes

| Code | Meaning | Action |
|---|---|---|
| 403 | Invalid/missing API key | Check `.env` / dashboard |
| 429 | Rate limit exceeded | Back off exponentially; reduce threads |
| 500 | Internal server error | Check API status page; retry with backoff |

## Plan limits (as of snapshot)

| Plan | Calls | History | Coverage highlights |
|---|---|---|---|
| Basic (free) | 250/day | EOD only | Profile/reference, 150+ endpoints |
| Starter $29/mo | 300/min | 5 years | US, annual fundamentals, news, crypto/forex |
| Premium $69/mo | 750/min | 30 years | +UK/Canada, full fundamentals+ratios, intraday, technicals, calendars, custom DCF |
| Ultimate $139/mo | 3,000/min | Full | +Global, transcripts, ETF/MF holdings, 13F, 1-min intraday, bulk/batch |

Note: bulk endpoints (`docs/vendor/fmp/bulk.md`) generally require Ultimate.

## Category files

| File | Contents |
|---|---|
| [search-and-directory.md](search-and-directory.md) | Symbol/name/CIK/CUSIP/ISIN search, screener, symbol lists, exchanges/sectors |
| [company-information.md](company-information.md) | Profile, market cap (incl. historical), float, peers, delisted companies, M&A, executives |
| [quotes.md](quotes.md) | Real-time quotes, batch quotes, aftermarket, price change |
| [financial-statements.md](financial-statements.md) | Income/balance/cash-flow (+TTM, growth, as-reported), key metrics, ratios, scores |
| [prices-and-charts.md](prices-and-charts.md) | EOD historical prices (light/full/unadjusted/dividend-adjusted), intraday 1min–4hour |
| [calendars-earnings-dividends-splits.md](calendars-earnings-dividends-splits.md) | Earnings, dividends, splits, IPOs + their calendars |
| [news-and-transcripts.md](news-and-transcripts.md) | Stock/crypto/forex/general news, press releases, earnings call transcripts |
| [analyst-and-market-performance.md](analyst-and-market-performance.md) | Estimates, ratings, price targets, grades; sector/industry performance and P/E snapshots; gainers/losers |
| [technical-indicators.md](technical-indicators.md) | SMA, EMA, WMA, DEMA, TEMA, RSI, stddev, Williams, ADX |
| [etf-and-funds.md](etf-and-funds.md) | ETF/MF holdings, info, sector/country weightings, disclosures |
| [sec-insider-13f.md](sec-insider-13f.md) | SEC filings search, insider trades, Form 13F institutional ownership |
| [indexes-and-constituents.md](indexes-and-constituents.md) | Index quotes and **point-in-time S&P 500 / Nasdaq / Dow constituents** |
| [economics-commodities-forex-crypto.md](economics-commodities-forex-crypto.md) | Treasury rates, economic indicators/calendar, commodities, forex, crypto |
| [other-datasets.md](other-datasets.md) | DCF valuation, market hours, Senate/House trades, ESG, COT, fundraisers |
| [bulk.md](bulk.md) | Bulk/batch endpoints for whole-universe downloads |

## Endpoints most relevant to this repo's data migration

Replacing the yfinance pipeline (`scripts/backfill_all.py`, `scripts/update_daily.py`):

| Need | FMP endpoint |
|---|---|
| Adjusted daily closes (replaces yfinance) | `/historical-price-eod/full?symbol=X` (also `dividend-adjusted`, `non-split-adjusted` variants) |
| Point-in-time S&P 500 membership | `/historical-sp500-constituent` + `/sp500-constituent` |
| Delisted names (fix corrupted panel) | `/delisted-companies` |
| Historical market caps (size factor) | `/historical-market-capitalization?symbol=X` |
| Splits/dividends sanity checks | `/splits?symbol=X`, `/dividends?symbol=X` |
| Sector/industry classifications | `/profile?symbol=X` or `/all-industry-classification` |
| Whole-universe EOD in one call | `/eod-bulk?date=YYYY-MM-DD` (Ultimate) |
