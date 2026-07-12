# Roadmap / Next-Steps Log

Working log for the platform build-out. Each entry has enough context to
resume cold ("continue with the roadmap" should be sufficient instruction).
Update this file whenever an item ships or a decision changes.

_Last updated: 2026-07-12 (commodities→FMP, Finnhub dropped, STI/BNY, value+quality NB)._

## Recently shipped

- FMP prices, market caps, VIX, sectors, PIT fundamentals, quarantine,
  lifecycle truncation, ADV-scaled costs (prior commit).
- **Commodities → FMP** (`core/data/commodities.py`). Internal keys unchanged
  (`GLD`, `WTI`, …); mapped to FMP ETFs/futures (`GLD`, `CLUSD`, `BZUSD`, …).
  Uses dividend-adjusted EOD (futures that 402 on `/full` still work there).
  **Series break**: energy/ag/industrial columns were Alpha Vantage *spot*;
  they are now FMP *futures*. Backup: `data/commodities/prices_backup_pre_fmp.parquet`.
- **Finnhub dropped**: deleted `core/data/finnhub_data.py`; removed from
  `config/settings.py`, `.env.example`, `CLAUDE.md`, `setup_environment.sh`.
  Required keys: `FMP_API_KEY`, `FRED_API_KEY`.
- **STI → quarantined** (extreme_returns + stale_prices). Recycled post-2022
  junk excluded at API load.
- **BNY continuation**: `scripts/add_fmp_symbol.py --symbols BNY --rebuild-factors`.
  FMP `BNY` carries full BNY Mellon history (corr≈1.0 vs `BK` through the
  2026-05-21 rename); `BK` stays truncated at the rename by lifecycle.
- **Value + quality notebook**: `notebooks/15_strategy_value_quality_sector_neutral.ipynb`
  + `core/signals/sector_neutral.py` (sector demean → z-score composite).

## 0. Survivorship gap — leave disclosed

Prefer 2015+ windows. Needs Norgate/CRSP/Tiingo to close.

## 1–5. DONE

Fundamentals, lifecycle v1, vendor consolidation (commodities included),
S&P refresh (CSV canonical until `--promote`), ADV costs.

## Open / optional

1. S&P CSV `--promote` after notation review; show CSV age on coverage page
2. Monthly lifecycle refresh inside `update_daily.py`
3. Register `value_quality_sn` as a strategy after notebook holdout looks stable
4. Macro vintages (low priority)
5. Remove stale `FINNHUB_API_KEY` line from local `.env` (not in git)
