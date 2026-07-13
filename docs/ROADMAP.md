# Roadmap / Next-Steps Log

Working log for the platform build-out. Each entry has enough context to
resume cold ("continue with the roadmap" should be sufficient instruction).
Update this file whenever an item ships or a decision changes.

_Last updated: 2026-07-12 (Gatev SSD pairs formation)._

## Recently shipped

- **Gatev distance formation** — normalize-to-1 SSD ranking on formation
  window, ADV-ranked sector universe, OOS z-score backtest;
  `core/strategies/pairs_gatev.py`, `/pairs` method toggle.
- **Pairs walk-forward screen** — same-sector Engle–Granger train filter +
  OOS PnL; `POST /screen-pairs`.
- **Pairs trading (Engle–Granger)** — `core/signals/pairs.py`,
  `POST /run-pairs-backtest`, `/pairs` UI, registry
  `pairs_cointegration`, `notebooks/17_strategy_pairs_cointegration.ipynb`.
- **`near_52w_high` (George & Hwang 2004)** — proximity to 252-day high
  factor + registry + panel patch + research notebook.
- Prior: resume-alignment (SHAP, Risk tab, quarantine UI, EDGAR, `beta_60d`);
  FMP stack, lifecycle, value_quality / roe_sn.


## 0. Survivorship gap — leave disclosed

Prefer 2015+ windows. Needs Norgate/CRSP/Tiingo to close.

## Open / optional

1. True ALFRED macro vintages (ingest) when a nowcast / revision strategy needs them
2. FMP membership `--promote` only after rename/recycle review (notation already fixed)
3. PIT sector source (replace today's FMP sector labels used in `*_sn` factors)
4. Persist `neg_vol_60d` / `neg_beta_60d` columns if we want one-click BAB/low-vol longs
  (ranker is descending — high factor = long)
