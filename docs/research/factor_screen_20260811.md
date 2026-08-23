# Factor library screen — 2026-08-11

First backtest of the 2026-08 factor library. long/short top-bottom 20%, monthly, 10bps one-way, SP500 PIT membership, T+1 execution, min 20 stocks.
Period 2000-01-03 -> 2026-08-07. **28 tests**, family-wise alpha 0.05, Šidák per-test threshold |t| ≥ 3.12.

A factor 'passing' here means its full-period net t-stat clears the
corrected threshold — necessary, nowhere near sufficient. Decade columns
expose pre-2010-only factors.

| Factor | Family | Net Sharpe | Ann ret | t | 2000s | 2010s | 2020s | MaxDD | Šidák |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| amihud_illiquidity | liquidity | 0.47 | 6.6% | 2.41 | 1.04 | 0.11 | -0.2 | -44% | fail |
| neg_net_operating_assets | earnings-quality | 0.42 | 4.6% | 2.16 | 0.27 | 0.53 | 0.65 | -28% | fail |
| sales_to_price | value | 0.42 | 6.0% | 2.16 | 0.74 | -0.01 | 0.34 | -44% | fail |
| operating_profitability | profitability | 0.37 | 4.5% | 1.89 | 0.37 | 0.39 | 0.46 | -42% | fail |
| corwin_schultz_spread | liquidity | 0.28 | 5.4% | 1.43 | 0.32 | 0.11 | 0.43 | -52% | fail |
| net_payout_yield | issuance | 0.27 | 4.0% | 1.37 | 0.34 | 0.38 | 0.11 | -46% | fail |
| book_to_market | value | 0.23 | 3.9% | 1.21 | 0.59 | -0.24 | 0.08 | -57% | fail |
| fcf_yield | value | 0.22 | 2.7% | 1.11 | 0.12 | 0.45 | 0.2 | -48% | fail |
| neg_accruals | earnings-quality | 0.20 | 1.8% | 1.04 | 0.21 | 0.15 | 0.28 | -26% | fail |
| cash_flow_to_price | value | 0.19 | 2.9% | 0.96 | 0.34 | -0.01 | 0.12 | -45% | fail |
| rd_to_market | value | 0.16 | 2.0% | 0.82 | -0.02 | 0.51 | 0.24 | -46% | fail |
| neg_net_share_issuance | issuance | 0.14 | 1.4% | 0.73 | 0.08 | 0.18 | 0.21 | -40% | fail |
| ebitda_to_ev | value | 0.10 | 1.5% | 0.53 | 0.41 | -0.26 | -0.1 | -62% | fail |
| neg_capex_intensity | investment | 0.09 | 1.2% | 0.46 | 0.05 | 0.28 | -0.02 | -50% | fail |
| rd_intensity | value | 0.08 | 1.1% | 0.43 | -0.26 | 0.5 | 0.32 | -64% | fail |
| neg_inventory_growth | investment | 0.05 | 0.5% | 0.28 | 0.07 | -0.03 | 0.11 | -34% | fail |
| gross_profitability | profitability | 0.04 | 0.6% | 0.18 | 0.03 | 0.16 | -0.07 | -53% | fail |
| neg_asset_growth | investment | 0.00 | 0.0% | 0.01 | 0.17 | -0.09 | -0.23 | -41% | fail |
| cfo_to_assets | profitability | -0.04 | -0.6% | -0.23 | -0.25 | 0.23 | 0.11 | -57% | fail |
| earnings_yield | value | -0.05 | -0.6% | -0.24 | 0.03 | -0.17 | -0.09 | -62% | fail |
| sue_price_scaled | earnings-surprise | -0.07 | -0.7% | -0.38 | -0.3 | -0.31 | 0.85 | -57% | fail |
| overnight_intraday_gap | microstructure | -0.09 | -1.3% | -0.48 | 0.03 | 0.06 | -0.46 | -66% | fail |
| roa | profitability | -0.12 | -1.9% | -0.62 | -0.3 | 0.13 | -0.03 | -66% | fail |
| altman_z | quality-score | -0.13 | -2.1% | -0.67 | -0.24 | 0.08 | -0.12 | -64% | fail |
| gross_margin | profitability | -0.17 | -1.6% | -0.90 | -0.23 | 0.3 | -0.56 | -52% | fail |
| close_location_21d | microstructure | -0.29 | -3.9% | -1.51 | -0.6 | -0.25 | 0.16 | -79% | fail |
| range_to_close_vol | microstructure | -0.33 | -3.4% | -1.69 | -0.06 | -0.23 | -0.84 | -75% | fail |
| piotroski_f | quality-score | -0.33 | -3.8% | -1.71 | -0.42 | -0.29 | -0.26 | -72% | fail |

Survivors: none.

Caveats: gross of borrow costs on the short leg; single cost assumption (10 bps) regardless of liquidity; `amihud_illiquidity` and `corwin_schultz_spread` deliberately long illiquid names, so their real costs are far above 10 bps; pre-2000 rows excluded (placeholder filing dates, ADR 0012).
