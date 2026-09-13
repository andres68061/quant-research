# FMP — ETF & Mutual Funds

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.
Most of this category requires the Ultimate plan (ETF/MF holdings).

| Endpoint | Example | Description |
|---|---|---|
| `/etf/holdings?symbol={s}` | `/etf/holdings?symbol=SPY` | Securities and weights inside an ETF/fund |
| `/etf/info?symbol={s}` | `/etf/info?symbol=SPY` | Fund name, expense ratio, AUM |
| `/etf/country-weightings?symbol={s}` | — | Asset allocation by country |
| `/etf/asset-exposure?symbol={s}` | `?symbol=AAPL` | Which ETFs hold a given stock (value, shares, weight) |
| `/etf/sector-weightings?symbol={s}` | `?symbol=SPY` | ETF sector allocation percentages |
| `/funds/disclosure-holders-latest?symbol={s}` | — | Latest MF/ETF disclosures holding a symbol |
| `/funds/disclosure?symbol={s}&year={y}&quarter={q}` | `?symbol=VWO&year=2023&quarter=4` | Fund disclosure filings detail |
| `/funds/disclosure-holders-search?name={n}` | — | Search disclosures by fund name |
| `/funds/disclosure-dates?symbol={s}` | — | Disclosure filing dates for a fund |
