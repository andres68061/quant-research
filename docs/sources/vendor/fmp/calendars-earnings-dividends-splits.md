# FMP — Earnings, Dividends, Splits, IPOs & Calendars

Base URL: `https://financialmodelingprep.com/stable/` — append `?apikey=$FMP_API_KEY`.
Calendar endpoints accept `from`/`to` date filters.

## Dividends

| Endpoint | Example | Description |
|---|---|---|
| `/dividends?symbol={s}` | `/dividends?symbol=AAPL` | Dividend history: record, payment, declaration dates |
| `/dividends-calendar` | — | Upcoming dividend events across all stocks |

## Earnings

| Endpoint | Example | Description |
|---|---|---|
| `/earnings?symbol={s}` | `/earnings?symbol=AAPL` | Earnings dates, EPS estimates vs actuals, revenue projections |
| `/earnings-calendar` | — | Upcoming and past earnings announcements, all companies |

## Splits

| Endpoint | Example | Description |
|---|---|---|
| `/splits?symbol={s}` | `/splits?symbol=AAPL` | Split dates and ratios for one company |
| `/splits-calendar` | — | Upcoming splits across companies |

## IPOs

| Endpoint | Description |
|---|---|
| `/ipos-calendar` | Upcoming IPOs: date, company, expected pricing, exchange |
| `/ipos-disclosure` | IPO regulatory filings with SEC links |
| `/ipos-prospectus` | Prospectus details: offer price, discounts, proceeds |

## Economics calendar (see also economics file)

| Endpoint | Description |
|---|---|
| `/economic-calendar` | Upcoming economic data releases |
