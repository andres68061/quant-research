#!/usr/bin/env python3
"""
Incremental daily update script for quantamental data.

This script:
1. Reads existing Parquet files
2. Fetches only new data since last update
3. Appends new data to Parquet files
4. Rebuilds factors for new dates
5. Refreshes sector classifications (quarterly, >90 days old)
6. Updates DuckDB views

Run daily/weekly to keep data current without full backfill.
Sector classifications are automatically refreshed every 3 months.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.factors.build_factors import build_price_factors, merge_market_cap
from core.data.factors.fama_french import update_ff5_parquet
from core.data.factors.io import connect_duckdb, register_parquet
from core.data.factors.macro import (
    RAW_MACRO_PARQUET,
    compute_macro_zscores,
    derive_macro_panel_from_raw,
    load_raw_macro_default,
)
from core.data.fmp.panel import update_panel_from_fmp
from core.data.quality import (
    load_quarantine_list,
    merge_with_existing,
    repair_isolated_bad_prints,
    scan_price_panel,
    write_quarantine_list,
)
from core.data.sector_classification import (
    add_or_update_sectors,
    get_symbols_needing_refresh,
    load_sector_classifications,
)
from core.utils.io import (
    read_parquet,
    write_parquet,
)


def update_prices(out_root: Path) -> bool:
    """
    Update prices.parquet with new data since last date.

    Returns:
        True if new data was added, False otherwise
    """
    prices_path = out_root / "prices.parquet"

    print(f"📈 Updating prices from {prices_path}...")

    # Read existing prices
    existing_prices = read_parquet(prices_path)

    if existing_prices is None or existing_prices.empty:
        print("⚠️  No existing prices found. Run backfill_all.py first.")
        return False

    last_date = existing_prices.index.max()
    print(f"   Last date in prices: {last_date.strftime('%Y-%m-%d')}")

    # Check if we need to update (skip if last date is today or in the future)
    # Handle timezone-aware timestamps
    today = pd.Timestamp.now()
    if last_date.tz is not None:
        today = today.tz_localize(last_date.tz)

    if last_date >= today.normalize():
        print("   ✅ Prices are already up to date!")
        return False

    # Update with new data from FMP (overlapping window catches vendor restatements)
    print(f"   Fetching new FMP data since {last_date.strftime('%Y-%m-%d')}...")
    updated_prices = update_panel_from_fmp(existing_prices)

    new_last_date = updated_prices.index.max()
    if new_last_date > last_date:
        new_rows = len(updated_prices) - len(existing_prices)
        print(f"   ✅ Added {new_rows} new dates")
        print(f"   New last date: {new_last_date.strftime('%Y-%m-%d')}")

        # Clean isolated bad prints in the derived panel (raw layer keeps vendor values)
        updated_prices, repair_log = repair_isolated_bad_prints(updated_prices)
        if not repair_log.empty:
            repairs_path = ROOT / "data" / "quality" / "bad_print_repairs.csv"
            previous_repairs = pd.read_csv(repairs_path) if repairs_path.exists() else pd.DataFrame()
            pd.concat([previous_repairs, repair_log.astype({"date": str})]).drop_duplicates(
                subset=["symbol", "date"]
            ).to_csv(repairs_path, index=False)
            print(f"   🩹 Repaired {len(repair_log)} isolated bad prints")

        write_parquet(updated_prices, prices_path)
        return True
    else:
        print("   ℹ️  No new data available")
        return False


def update_macro(out_root: Path, raw_path: Path = RAW_MACRO_PARQUET) -> bool:
    """
    Refresh the raw FRED panel and derive macro.parquet plus macro_z.parquet.

    The raw long-format panel at ``raw_path`` is the source of truth; the
    publication-lagged business-day panel and z-score panel are deterministic
    derivations of it.

    Returns:
        True if the latest derived ``macro.parquet`` extended further than the
        prior version, False otherwise.
    """
    macro_path = out_root / "macro.parquet"
    macro_z_path = out_root / "macro_z.parquet"

    print(f"📊 Refreshing raw macro layer at {raw_path}...")

    existing_macro = read_parquet(macro_path)
    last_date = existing_macro.index.max() if existing_macro is not None and not existing_macro.empty else None

    print("   Fetching latest raw FRED series...")
    raw_long = load_raw_macro_default()
    if raw_long.empty:
        print("⚠️  Raw FRED fetch returned no rows; skipping macro derivation.")
        return False

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(raw_long, raw_path)
    print(f"   ✅ Wrote raw panel ({len(raw_long):,} rows) to {raw_path}")

    derived_macro = derive_macro_panel_from_raw(raw_long)
    new_last_date = derived_macro.index.max()

    if last_date is not None and new_last_date <= last_date:
        print("   ℹ️  Macro data is up to date; rewriting derived files for consistency.")

    write_parquet(derived_macro, macro_path)
    write_parquet(compute_macro_zscores(derived_macro), macro_z_path)
    print(f"   ✅ Derived macro.parquet through {new_last_date.strftime('%Y-%m-%d')}")
    return last_date is None or new_last_date > last_date


def rebuild_factors(out_root: Path, prices_updated: bool) -> None:
    """
    Rebuild price factors if prices were updated.

    Args:
        out_root: Output directory
        prices_updated: Whether prices were updated
    """
    if not prices_updated:
        print("📉 Skipping factor rebuild (no new price data)")
        return

    print("📉 Rebuilding price factors...")

    prices_path = out_root / "prices.parquet"
    factors_price_path = out_root / "factors_price.parquet"
    factors_all_path = out_root / "factors_all.parquet"

    # Read updated prices
    prices = read_parquet(prices_path)

    if prices is None or prices.empty:
        print("⚠️  No prices available for factor calculation")
        return

    # Build price factors
    market_symbol = "^GSPC"
    factors_price = build_price_factors(prices, market_symbol=market_symbol)

    print(f"   ✅ Rebuilt price factors: {factors_price.shape}")

    # Write factors
    write_parquet(factors_price, factors_price_path)

    # Merge market cap if available, then write factors_all
    from pathlib import Path as _P

    mcap_path = _P("data/market_caps/historical_market_caps.parquet")
    factors_all = merge_market_cap(factors_price, mcap_path)
    write_parquet(factors_all, factors_all_path)

    print("💧 Rebuilding dollar-ADV panel...")
    from core.data.liquidity import DEFAULT_ADV_PATH, DEFAULT_RAW_DIR, build_dollar_adv_panel

    dollar_adv = build_dollar_adv_panel(ROOT / DEFAULT_RAW_DIR)
    adv_path = ROOT / DEFAULT_ADV_PATH
    write_parquet(dollar_adv, adv_path)
    print(f"   ✅ Wrote {adv_path}: {dollar_adv.shape}")


def rescan_data_quality(out_root: Path, prices_updated: bool) -> None:
    """
    Re-run the quarantine scanner after new price data lands.

    Manual ``cleared`` decisions in the existing list are preserved.
    """
    if not prices_updated:
        print("🩺 Skipping quality rescan (no new price data)")
        return

    print("🩺 Rescanning data quality...")
    prices = read_parquet(out_root / "prices.parquet")
    if prices is None or prices.empty:
        return

    findings = scan_price_panel(prices)
    merged = merge_with_existing(findings, load_quarantine_list())
    write_quarantine_list(merged)
    n_quarantined = (merged["status"] == "quarantined").sum() if not merged.empty else 0
    print(f"   ✅ Quality scan done: {len(merged)} findings, {n_quarantined} quarantined rows")


def refresh_fundamentals_if_due(max_age_days: int = 7) -> bool:
    """
    Weekly refresh of FMP statements + rebuild of the PIT fundamentals panel.

    Skips when ``factors_fundamental.parquet`` is newer than ``max_age_days``.
    Fetches only symbols whose raw income-statement file is missing or stale.
    """
    import subprocess

    factors_path = ROOT / "data" / "factors" / "factors_fundamental.parquet"
    if factors_path.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(factors_path.stat().st_mtime, unit="s")).days
        if age_days < max_age_days:
            print(f"📑 Fundamentals panel is {age_days}d old — skipping (refresh every {max_age_days}d)")
            return False

    print("📑 Refreshing FMP fundamentals (weekly)...")
    fetch = subprocess.run(
        ["/opt/anaconda3/envs/quant/bin/python", str(ROOT / "scripts" / "fetch_fmp_fundamentals.py"), "--refresh"],
        cwd=str(ROOT),
        check=False,
    )
    if fetch.returncode != 0:
        print(f"   ⚠️  Fundamentals fetch exited {fetch.returncode}; skipping panel rebuild")
        return False
    build = subprocess.run(
        ["/opt/anaconda3/envs/quant/bin/python", str(ROOT / "scripts" / "build_fundamentals_panel.py")],
        cwd=str(ROOT),
        check=False,
    )
    if build.returncode != 0:
        print(f"   ⚠️  Fundamentals panel rebuild exited {build.returncode}")
        return False
    print("   ✅ Fundamentals panel refreshed")
    return True


def _file_age_days(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    return (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days


def refresh_market_caps_if_due(max_age_days: int = 7) -> bool:
    """Weekly FMP historical market-cap refresh + panel rebuild."""
    import subprocess

    panel_path = ROOT / "data" / "market_caps" / "historical_market_caps.parquet"
    age = _file_age_days(panel_path)
    if age is not None and age < max_age_days:
        print(f"💰 Market-cap panel is {age}d old — skipping (refresh every {max_age_days}d)")
        return False

    print("💰 Refreshing FMP historical market caps (weekly)...")
    # Incremental: skip existing raw files; rebuild the stacked panel.
    result = subprocess.run(
        ["/opt/anaconda3/envs/quant/bin/python", str(ROOT / "scripts" / "fetch_fmp_market_caps.py")],
        cwd=str(ROOT),
        check=False,
    )
    if result.returncode != 0:
        print(f"   ⚠️  Market-cap refresh exited {result.returncode}")
        return False
    print("   ✅ Market-cap panel refreshed")
    return True


def refresh_sp500_membership_if_due(max_age_days: int = 7) -> bool:
    """Weekly FMP S&P membership rebuild + CSV reconciliation report."""
    import subprocess

    membership_path = ROOT / "data" / "raw" / "fmp" / "constituents" / "sp500_membership.parquet"
    age = _file_age_days(membership_path)
    if age is not None and age < max_age_days:
        print(f"🏛️  S&P membership is {age}d old — skipping (refresh every {max_age_days}d)")
        return False

    print("🏛️  Refreshing S&P 500 membership from FMP (weekly)...")
    result = subprocess.run(
        ["/opt/anaconda3/envs/quant/bin/python", str(ROOT / "scripts" / "refresh_sp500_constituents.py")],
        cwd=str(ROOT),
        check=False,
    )
    if result.returncode != 0:
        print(f"   ⚠️  S&P membership refresh exited {result.returncode}")
        return False
    print("   ✅ S&P membership refreshed (see data/quality/sp500_reconciliation.txt)")
    return True


def refresh_vix_if_due(max_age_days: int = 1) -> bool:
    """Daily VIX refresh from FMP."""
    from core.data.vix import load_vix

    vix_path = ROOT / "data" / "factors" / "vix.parquet"
    age = _file_age_days(vix_path)
    if age is not None and age < max_age_days:
        print(f"📉 VIX is {age}d old — skipping")
        return False
    print("📉 Refreshing VIX from FMP...")
    series = load_vix(force_refresh=True)
    print(f"   ✅ VIX refreshed: {len(series)} rows through {series.index.max().date() if len(series) else 'n/a'}")
    return True


def refresh_lifecycle_if_due(prices_updated: bool, max_age_days: int = 30) -> bool:
    """
    Monthly rebuild of symbol lifecycle windows; re-apply after price updates.

    Price updates can reintroduce bars outside a ticker's valid life (FMP
    restatements / recycled symbols). Re-applying the existing windows is cheap;
    refetching FMP delisted/symbol-change registries is monthly.
    """
    import subprocess

    lifecycle_path = ROOT / "data" / "quality" / "symbol_lifecycle.parquet"
    age = _file_age_days(lifecycle_path)
    windows_stale = age is None or age >= max_age_days

    if not windows_stale and not prices_updated:
        print(
            f"⏳ Lifecycle windows are {age}d old — skipping "
            f"(rebuild every {max_age_days}d; re-apply when prices update)"
        )
        return False

    script = str(ROOT / "scripts" / "build_symbol_lifecycle.py")
    if windows_stale:
        print("⏳ Refreshing symbol lifecycle windows from FMP (monthly) + applying...")
        cmd = ["/opt/anaconda3/envs/quant/bin/python", script, "--apply"]
    else:
        print("⏳ Re-applying existing lifecycle windows after price update...")
        cmd = ["/opt/anaconda3/envs/quant/bin/python", script, "--apply-only"]

    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        print(f"   ⚠️  Lifecycle refresh exited {result.returncode}")
        return False
    print("   ✅ Lifecycle applied (rebuild factors if prices changed)")
    return True


def update_sectors_if_needed(out_root: Path) -> bool:
    """
    Check if sector classifications need quarterly refresh.
    Only updates symbols that are >90 days old.

    Returns:
        True if sectors were updated, False otherwise
    """
    print("📊 Checking sector classifications...")

    # Load existing sector data
    sector_df = load_sector_classifications()

    if sector_df is None or sector_df.empty:
        print("   ℹ️  No sector data found - run fetch_sectors.py first")
        return False

    # Get symbols needing refresh (>90 days old)
    stale_symbols = get_symbols_needing_refresh(sector_df, refresh_days=90)

    if not stale_symbols:
        print("   ✅ Sector classifications are up to date")
        return False

    print(f"   Found {len(stale_symbols)} symbols needing quarterly refresh")
    print(f"   Updating sectors (this may take a few minutes)...")

    # Update stale sectors
    add_or_update_sectors(stale_symbols, force_refresh=True)

    print(f"   ✅ Updated {len(stale_symbols)} sector classifications")
    return True


def update_duckdb_views(out_root: Path, db_path: Path) -> None:
    """
    Update DuckDB views to point to updated Parquet files.

    Args:
        out_root: Directory containing Parquet files
        db_path: Path to DuckDB database
    """
    print(f"🦆 Updating DuckDB views at {db_path}...")

    con = connect_duckdb(db_path)

    # Register all views
    views = {
        "prices": out_root / "prices.parquet",
        "macro": out_root / "macro.parquet",
        "macro_z": out_root / "macro_z.parquet",
        "factors_price": out_root / "factors_price.parquet",
        "factors_all": out_root / "factors_all.parquet",
        "fama_french_5": out_root / "fama_french_5.parquet",
    }

    for name, path in views.items():
        if path.exists():
            register_parquet(con, name, path)
            print(f"   ✅ Registered view: {name}")

    con.close()


def main(out_root: str = "data/factors", db_path: str = "data/factors/factors.duckdb"):
    """
    Main incremental update workflow.

    Args:
        out_root: Output directory for Parquet files
        db_path: Path to DuckDB database
    """
    out_root_p = Path(out_root)
    db_path_p = Path(db_path)

    print("=" * 80)
    print("🔄 INCREMENTAL DATA UPDATE")
    print("=" * 80)
    print()

    # Step 1: Update prices
    prices_updated = update_prices(out_root_p)
    print()

    # Step 2: Update macro
    macro_updated = update_macro(out_root_p)
    print()

    # Step 3: Update Fama-French 5 factors
    ff5_path = out_root_p / "fama_french_5.parquet"
    try:
        update_ff5_parquet(ff5_path)
        print("   Updated Fama-French 5 factors")
    except Exception as exc:
        print(f"   Warning: FF5 update failed ({exc}); continuing.")
    print()

    # Step 4: Lifecycle before factor rebuild so truncated NaNs feed factors
    lifecycle_updated = refresh_lifecycle_if_due(prices_updated)
    print()

    # Step 4b: Rebuild price factors when the panel changed
    rebuild_factors(out_root_p, prices_updated or lifecycle_updated)
    print()

    # Step 4c: Rescan data quality on the refreshed panel
    rescan_data_quality(out_root_p, prices_updated or lifecycle_updated)
    print()
    # Step 4c: Weekly fundamentals refresh (statements change slowly)
    fundamentals_updated = refresh_fundamentals_if_due()
    print()

    # Step 4d: Weekly historical market caps
    market_caps_updated = refresh_market_caps_if_due()
    print()

    # Step 4e: Weekly S&P membership + reconciliation report
    sp500_updated = refresh_sp500_membership_if_due()
    print()

    # Step 4f: VIX (daily)
    vix_updated = refresh_vix_if_due()
    print()

    # Step 5: Update sector classifications (quarterly refresh)
    sectors_updated = update_sectors_if_needed(out_root_p)
    print()

    # Step 6: Update DuckDB views
    update_duckdb_views(out_root_p, db_path_p)
    print()

    print("=" * 80)
    if any(
        [
            prices_updated,
            macro_updated,
            sectors_updated,
            fundamentals_updated,
            market_caps_updated,
            sp500_updated,
            vix_updated,
            lifecycle_updated,
        ]
    ):
        print("✅ Incremental update completed successfully!")
        if sectors_updated:
            print("   📊 Sector classifications refreshed (quarterly update)")
        if fundamentals_updated:
            print("   📑 Fundamentals panel refreshed (weekly update)")
        if lifecycle_updated:
            print("   ⏳ Symbol lifecycle windows refreshed / re-applied")
        if market_caps_updated:
            print("   💰 Market-cap panel refreshed (weekly update)")
        if sp500_updated:
            print("   🏛️  S&P membership refreshed (weekly update)")
        if vix_updated:
            print("   📉 VIX refreshed")
    else:
        print("✅ Data is already up to date - no changes needed")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Incremental daily/weekly data update")
    p.add_argument(
        "--out", type=str, default="data/factors", help="Output directory for Parquet files"
    )
    p.add_argument(
        "--db", type=str, default="data/factors/factors.duckdb", help="Path to DuckDB database"
    )
    args = p.parse_args()

    main(args.out, args.db)
