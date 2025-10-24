import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so 'src' package resolves when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.factors.build_factors import build_price_factors
from src.data.factors.fundamentals_fmp import (
    apply_asof_lag,
    compute_value_quality_factors,
    dailyize_fundamentals,
    load_bulk_ratios_range,
)
from src.data.factors.io import connect_duckdb, ensure_dirs, register_parquet, write_parquet
from src.data.factors.macro import compute_macro_zscores, load_default_macro
from src.data.factors.prices import build_prices_panel
from src.data.factors.universe import (
    fetch_sp500_from_fmp,
    fetch_sp500_from_wikipedia,
    load_sp500_static,
)


def main(years: int, universe: str, out_root: str, db_path: str, refresh_cache: bool = False):
    out_root_p = Path(out_root)
    db_path_p = Path(db_path)
    ensure_dirs(out_root_p)

    # Universe
    if universe == 'auto':
        # Prefer FMP; fallback to Wikipedia; then CSV if provided
        try:
            uni = fetch_sp500_from_fmp()
        except Exception:
            try:
                uni = fetch_sp500_from_wikipedia()
            except Exception as err2:
                raise RuntimeError(
                    "Failed to load S&P500 from FMP and Wikipedia. Provide --universe path to CSV."
                ) from err2
    else:
        uni = load_sp500_static(Path(universe))
    symbols = uni['symbol'].tolist()
    # Ensure market benchmark for beta calculation (Yahoo S&P 500 index)
    market_symbol = '^GSPC'
    if market_symbol not in symbols:
        symbols.append(market_symbol)

    # Prices panel
    prices = build_prices_panel(symbols)

    # Macro (example minimal set)
    macro = load_default_macro()
    macro_z = compute_macro_zscores(macro)

    # Price factors
    factors_price = build_price_factors(prices, market_symbol=market_symbol)

    # Fundamentals (FMP) with as-of lag and dailyization
    # Fundamentals via bulk with caching (cover the backfill range)
    end_year = pd.Timestamp.today().year
    start_year = max(end_year - years, end_year - 10)  # cap to 10y window
    fund_raw = load_bulk_ratios_range(start_year, end_year, period='quarter', refresh_cache=refresh_cache)
    factors_vq = None
    fund_daily = None
    fund_lag = apply_asof_lag(fund_raw, lag_days=60)
    if fund_lag is not None and not fund_lag.empty:
        # Align daily span to prices index
        start, end = prices.index.min(), prices.index.max()
        fund_daily = dailyize_fundamentals(fund_lag, start=start, end=end)
        if fund_daily is not None and not fund_daily.empty:
            factors_vq = compute_value_quality_factors(fund_daily)

    # Combine factors (left join on index)
    # factors_price and factors_vq share index (date,symbol)
    if factors_vq is not None and not factors_vq.empty:
        factors_all = factors_price.join(factors_vq, how='left')
    else:
        factors_all = factors_price.copy()

    # Persist to Parquet
    write_parquet(prices, out_root_p / 'prices.parquet')
    write_parquet(macro, out_root_p / 'macro.parquet')
    write_parquet(macro_z, out_root_p / 'macro_z.parquet')
    write_parquet(factors_price, out_root_p / 'factors_price.parquet')
    if fund_daily is not None and not fund_daily.empty:
        write_parquet(fund_daily, out_root_p / 'fundamentals_daily.parquet')
    if factors_vq is not None and not factors_vq.empty:
        write_parquet(factors_vq, out_root_p / 'factors_vq.parquet')
    write_parquet(factors_all, out_root_p / 'factors_all.parquet')

    # Register in DuckDB
    con = connect_duckdb(db_path_p)
    register_parquet(con, 'prices', out_root_p / 'prices.parquet')
    register_parquet(con, 'macro', out_root_p / 'macro.parquet')
    register_parquet(con, 'macro_z', out_root_p / 'macro_z.parquet')
    register_parquet(con, 'factors_price', out_root_p / 'factors_price.parquet')
    if fund_daily is not None and not fund_daily.empty:
        register_parquet(con, 'fundamentals_daily', out_root_p / 'fundamentals_daily.parquet')
    if factors_vq is not None and not factors_vq.empty:
        register_parquet(con, 'factors_vq', out_root_p / 'factors_vq.parquet')
    register_parquet(con, 'factors_all', out_root_p / 'factors_all.parquet')

    print('Backfill completed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--years', type=int, default=10)
    p.add_argument('--universe', type=str, default='auto', help="'auto' to fetch S&P500 from Wikipedia or path to CSV")
    p.add_argument('--out', type=str, default='data/factors')
    p.add_argument('--db', type=str, default='data/factors/factors.duckdb')
    p.add_argument('--refresh-cache', action='store_true', help='Force refresh of cached FMP bulk files')
    args = p.parse_args()
    main(args.years, args.universe, args.out, args.db, refresh_cache=args.refresh_cache)


