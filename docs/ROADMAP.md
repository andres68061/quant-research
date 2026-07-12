# Roadmap / Next-Steps Log

Working log for the platform build-out. Each entry has enough context to
resume cold ("continue with the roadmap" should be sufficient instruction).
Update this file whenever an item ships or a decision changes.

_Last updated: 2026-07-11 night (ADV-scaled transaction costs)._

## Recently shipped

- FMP prices, market caps, VIX, PIT fundamentals, value/quality strategies,
  quarantine, survivorship leave-disclosed (see prior entries).
- **Sectors → FMP `/profile`** (`core/data/sector_classification.py`). No more
  yfinance for sectors. Limitation: today's sector applied to history (mild
  lookahead; documented). Finnhub no longer needed for this path.
- **Symbol lifecycle truncation** (`core/data/lifecycle.py`,
  `scripts/build_symbol_lifecycle.py`): windows from price gaps + trusted
  FMP delisted/symbol-change (only when the series ends within 30–90 days of
  the event — avoids recycled-ticker false positives). Applied to derived
  `prices.parquet` (6,156 cells cleared). Example: BK ends 2026-05-21 on the
  BK→BNY rename; AET ends at the 2018 CVS deal. STI: FMP only has post-2022 junk
  (SunTrust era absent) — still flagged, should be escalated to quarantined.
- **ADV-scaled transaction costs** (`core/data/liquidity.py`,
  `data/factors/dollar_adv_21d.parquet`): buckets ≥$100M→5bps, ≥$20M→10bps,
  ≥$5M→20bps, else 40bps. Wired into `calculate_portfolio_returns` /
  factor runner / API backtest+replay. Rebuilds in `update_daily.py`.
  On SP500 short-term reversal 2018–2025, scaled costs cut cumulative drag
  ~12%→7% vs flat 10bps (most names are liquid).

## 0. Survivorship gap — leave disclosed

Prefer 2015+ windows. Needs Norgate/CRSP/Tiingo to close.

## 1. Fundamentals — DONE

Open: sector-neutral / value+quality composite notebook.

## 2. Symbol lifecycle — DONE (v1)

Open: fetch continuation tickers after renames (BNY for BK); escalate STI to
quarantined; optional monthly refresh in `update_daily.py`.

## 3. Vendor consolidation — NEARLY DONE

| Source | Status |
|---|---|
| Prices / mcaps / VIX / sectors | ✅ FMP |
| S&P membership refresh | ✅ FMP (CSV canonical until `--promote`) |
| Commodities | ⏳ yfinance; FMP `GCUSD` etc. works on Premium — migrate when needed |
| Finnhub | ⏳ unused in live flows after sectors move — safe to drop from required env |

## 4. S&P membership — MOSTLY DONE

Optional `--promote` after notation review; show CSV age on coverage page.

## 5. Transaction-cost realism — DONE

ADV panel + bucket costs live. Flat `transaction_cost` remains the fallback
when ADV is missing.

## Next (pick one)

1. **Commodities → FMP** (finish vendor consolidation)
2. **Drop Finnhub** from required env / settings
3. **STI → quarantined** + fetch **BNY** continuation for BK
4. Sector-neutral / value+quality research notebook

## 6–7. Macro vintages / smaller items

Unchanged (low priority).
