# FMP — DCF, Market Hours, Senate/House, ESG, COT, Fundraisers

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.

## Discounted Cash Flow

| Endpoint | Description |
|---|---|
| `/discounted-cash-flow?symbol={s}` | Standard DCF valuation |
| `/levered-discounted-cash-flow?symbol={s}` | DCF including debt impact |
| `/custom-discounted-cash-flow?symbol={s}` | Custom-assumption DCF (many tunable inputs) |
| `/custom-levered-discounted-cash-flow?symbol={s}` | Custom levered DCF |

## Market hours

| Endpoint | Example | Description |
|---|---|---|
| `/exchange-market-hours?exchange={x}` | `?exchange=NASDAQ` | Open/close times for an exchange |
| `/holidays-by-exchange?exchange={x}` | — | Exchange holiday calendar |
| `/all-exchange-market-hours` | — | Hours for all exchanges |

## Congressional trading

| Endpoint | Example | Description |
|---|---|---|
| `/senate-latest?page=0&limit=100` | — | Latest Senate financial disclosures |
| `/house-latest?page=0&limit=100` | — | Latest House financial disclosures |
| `/senate-trades?symbol={s}` | `?symbol=AAPL` | Senate trades in a stock |
| `/senate-trades-by-name?name={n}` | — | Senate trades by senator name |
| `/house-trades?symbol={s}` | — | House trades in a stock |
| `/house-trades-by-name?name={n}` | — | House trades by member name |

## ESG

| Endpoint | Description |
|---|---|
| `/esg-disclosures?symbol={s}` | ESG disclosure search |
| `/esg-ratings?symbol={s}` | ESG ratings |
| `/esg-benchmark` | Industry ESG benchmark comparison |

## Commitment of Traders

| Endpoint | Description |
|---|---|
| `/commitment-of-traders-report` | COT long/short positions by sector |
| `/commitment-of-traders-analysis` | COT sentiment analysis by date range |
| `/commitment-of-traders-list` | Available COT report symbols |

## Fundraisers

| Endpoint | Description |
|---|---|
| `/crowdfunding-offerings-latest?page=0&limit=100` | Latest crowdfunding campaigns |
| `/crowdfunding-offerings-search?name={n}` | Search crowdfunding campaigns |
| `/crowdfunding-offerings?cik={cik}` | Campaigns by company CIK |
| `/fundraising-latest?page=0&limit=10` | Latest equity offerings |
| `/fundraising-search?name={n}` | Search equity offerings |
| `/fundraising?cik={cik}` | Equity offerings by CIK |
