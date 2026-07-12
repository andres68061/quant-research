---
name: fmp-api
description: Reference for the Financial Modeling Prep (FMP) API used for market data in this repo. Use when fetching stock prices, fundamentals, index constituents, market caps, delistings, news, or any financial data from FMP, when writing or editing scripts that call financialmodelingprep.com, or during the yfinance-to-FMP data migration.
---

# FMP API Usage

## Basics

- Base URL: `https://financialmodelingprep.com/stable/`
- Auth: `?apikey=` query param (or `apikey:` header). The key is `FMP_API_KEY`
  in `.env`, loaded via `config/settings.py` (`from config.settings import FMP_API_KEY`).
  **Never** hardcode the key in code, notebooks, docs, or logs.
- Wrap calls with retries + exponential backoff + timeout (project rule for all
  external APIs). Handle 429 by backing off; 403 means bad/missing key.

## Endpoint reference (local snapshot)

Full endpoint catalog lives in `docs/vendor/fmp/` — read `README.md` there for
the category index. Most relevant files:

- `prices-and-charts.md` — EOD/intraday historical prices (the price-panel migration)
- `indexes-and-constituents.md` — point-in-time S&P 500 membership
- `company-information.md` — historical market caps, delisted companies
- `bulk.md` — whole-universe downloads (`/eod-bulk?date=` is one call per trading day)

Per-endpoint parameter tables were not captured in the snapshot (JS-rendered);
when exact parameters matter, probe the endpoint or check the live docs.

## Quick example

```python
import requests

from config.settings import FMP_API_KEY

response = requests.get(
    "https://financialmodelingprep.com/stable/historical-price-eod/full",
    params={"symbol": "AAPL", "apikey": FMP_API_KEY},
    timeout=30,
)
response.raise_for_status()
daily_bars = response.json()
```

## Repo conventions for new FMP fetchers

- Fetch scripts go in `scripts/`, pure transforms in `core/data/`.
- Write raw responses to a raw layer parquet before deriving anything
  (mirror the FRED pattern: `data/raw/...` → derived `data/factors/...`).
- Validate shape/dtypes/NaN counts on load; keep indexes tz-aware.
