# IB1MXXN.MX Data Fix

## Issue
The Mexican ETF `IB1MXXN.MX` (iShares Treasury Bond 0-1yr UCITS ETF) was added to the database but had **0 data points**.

## Root Cause
**Timezone mismatch:**
- yfinance returns timezone-naive data for Mexican securities
- Our database uses `America/New_York` timezone
- When reindexing, timezone-naive data couldn't align with timezone-aware index
- Result: All values became NaN

## Solution
1. Re-fetch using `yf.download()` with `auto_adjust=True`
2. Detect timezone of fetched data (None)
3. Convert to `America/New_York` timezone using `tz_localize()`
4. Reindex to match existing database dates
5. Save and rebuild factors

## Results
✅ **IB1MXXN.MX now has 1,500 data points**

- **Date range:** 2019-08-21 to 2026-01-30
- **Latest price:** $9,096.30
- **ETF Name:** iShares Treasury Bond 0-1yr UCITS ETF
- **Exchange:** MEX (Bolsa Mexicana de Valores)
- **Quote Type:** ETF

## Code Used
```python
import yfinance as yf
import pandas as pd

# Fetch data
data = yf.download("IB1MXXN.MX", period="max", auto_adjust=True)

# Load existing prices
prices = pd.read_parquet('data/factors/prices.parquet')

# Get Close column
new_series = data['Close'] if 'Close' in data.columns else data

# Convert timezone to match existing data
new_series.index = new_series.index.tz_localize(prices.index.tz)

# Reindex and update
prices['IB1MXXN.MX'] = new_series.reindex(prices.index)

# Save
prices.to_parquet('data/factors/prices.parquet')
```

## Verification
ChatGPT was correct - `IB1MXXN.MX` is the proper ticker for this Mexican ETF.

Other attempted tickers that failed:
- `IB1MXX.XC` - 404 Not Found
- `IB1MXX` - 404 Not Found
- `IB1MXXN.MX` - ✅ Works!

## Lesson Learned
When fetching international securities with yfinance:
1. Always check timezone of returned data
2. Match timezone to existing database
3. Use `tz_localize()` for timezone-naive data
4. Use `tz_convert()` for timezone-aware data with different timezone
5. Verify data points after reindexing
