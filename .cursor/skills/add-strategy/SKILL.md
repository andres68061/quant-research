---
name: add-strategy
description: End-to-end playbook for implementing a new trading strategy in this quant platform (core logic, registry, tests, data patch, research notebook, API, frontend). Use whenever the user asks to add, implement, or create a strategy, signal, or factor, or to wire an existing signal into the platform.
---

# Adding a Strategy to the Quant Platform

## First: Classify the Strategy

**Cross-sectional factor** (rank a universe on a column, long top / short bottom)?
→ Follow the **Factor path**. No new API route or frontend page is needed — the
shared `POST /run-backtest` endpoint and the Factor Backtest page discover factor
columns dynamically from `GET /data/factors`.

**Anything else** (ML direction, overlay, time-series, single-asset)?
→ Follow the **Custom path** (factor path steps plus API/frontend work).

## Factor Path Checklist

Copy and track:

```
- [ ] 1. Factor function in core/data/factors/build_factors.py
- [ ] 2. Wire into build_price_factors()
- [ ] 3. Unit tests in tests/test_factor_formula.py
- [ ] 4. StrategyMetadata in core/strategies/registry.py + registry test
- [ ] 5. Patch parquet panels in place (do NOT rerun full backfill)
- [ ] 6. Research notebook in notebooks/ (MANDATORY — see below)
- [ ] 7. Verify: pytest, lint, live API smoke test
```

### Step 1–2: Factor function

- Pure function: Series in, Series out. NaN until the full lookback window exists
  (never `fill_value=0`). Docstring with formula, sign convention, and units.
- **Sign convention matters**: the pipeline ranks DESCENDING and longs the top
  tier. If the strategy longs LOW values (e.g. losers, low vol), negate the
  factor so "high = long".
- Add the column in `build_price_factors()` so future rebuilds include it.
- Check `_infer_abs_bound` in `core/backtest/portfolio.py`: new column-name
  prefixes fall through to a default bound of 10.0 — add a case if that is wrong
  for the factor's scale.

### Step 3–4: Tests and registry

- Formula tests: price-ratio identity, sign convention, NaN for short histories,
  first-valid-index position. Mirror `TestShortTermReversal` in
  `tests/test_factor_formula.py`.
- Registry entry: honest `hypothesis`, published `reference`, realistic
  `expected_sharpe_range`, and non-softened `known_limitations` (cost drag,
  crowding, regime dependence). `kind=FACTOR_CROSS_SECTION`,
  `post_path="/run-backtest"`. Add the id to
  `tests/test_strategies_registry.py::test_list_contains_expected_ids`.

### Step 5: Patch data in place

Full backfill recomputes every factor for ~800 symbols and is slow. Instead,
compute only the new column vectorized on the wide price panel and join it onto
BOTH parquet files:

```python
import pandas as pd
from core.data.factors.build_factors import my_new_factor

prices = pd.read_parquet("data/factors/prices.parquet")
new_col = prices.apply(my_new_factor).stack(future_stack=True).rename("my_col")
new_col.index.names = ["date", "symbol"]
for path in ("data/factors/factors_price.parquet", "data/factors/factors_all.parquet"):
    panel = pd.read_parquet(path)
    panel.drop(columns=["my_col"], errors="ignore").join(new_col, how="left").to_parquet(path)
```

Sanity-check the fill rate against existing columns (`panel.notna().mean()`).

### Step 6: Research notebook (mandatory)

Every registered strategy gets `notebooks/NN_strategy_<id>.ipynb` following the
`research-notebook-narrative` skill (read it before writing the notebook).
Minimum content: plain-English research question + worked example, per-variable
audit, exact portfolio-construction disclosure (lag, universe, costs), backtest
vs a reference factor and vs gross (0-cost) to isolate cost drag, yes/no decision
questions, and a risks/limitations section. Requirements:

- Kernel `quant`; resolve paths from repo root OR notebooks/ (`Path.cwd()` probe);
  guard missing data files with a clear `FileNotFoundError`.
- Import everything from `core.*` — never re-implement strategy math in the notebook.
- Execute with `/opt/anaconda3/envs/quant/bin/jupyter nbconvert --execute --inplace`
  and validate with `nbformat.validate` before finishing.

### Step 7: Verification

```bash
/opt/anaconda3/envs/quant/bin/python -m pytest tests/ -q
/opt/anaconda3/envs/quant/bin/python -m ruff check . && /opt/anaconda3/envs/quant/bin/python -m black --check <changed files>
# Boot API, confirm the factor and strategy are served, run one real backtest:
curl -s localhost:8000/data/factors
curl -s localhost:8000/strategies
curl -s -X POST localhost:8000/run-backtest -H "Content-Type: application/json" -d '{"factor_col":"<col>"}'
```

## Custom Path (non-factor strategies)

Factor-path steps 1, 3, 4, 6, 7 still apply, plus:

- Core runner in `core/` (mirror `core/strategies/factor_runner.py` thinness).
- Pydantic request/response in `api/schemas/`, thin route in `api/routes/`,
  register router in `api/main.py`. No math in routes.
- Frontend: types in `lib/types.ts`, client method in `lib/api.ts`, page in
  `pages/` using `AppLayout`, route in `App.tsx`, nav link in `TopBar.tsx`.

## Known Data Pitfalls

- ~87 delisted symbols in `prices.parquet` have corrupted quotes (spikes like
  $0.68 → $11,000), concentrated 2010–2017. Backtests starting before ~2018
  produce meaningless metrics for ANY factor. Use recent windows or wait for the
  FMP data migration (see `docs/vendor/fmp/README.md`).
- Always pass `universe_filter=sp500_universe_filter()` (import from
  `core.backtest.portfolio`) for survivorship-free results.
- Metric keys from `calculate_performance_metrics` are `annualized_return` /
  `annualized_volatility` (not `annual_*`).
