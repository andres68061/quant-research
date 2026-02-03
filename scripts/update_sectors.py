#!/usr/bin/env python3
"""
Quarterly update of sector classifications.

This script refreshes sector classifications for symbols that are:
1. Older than 90 days (3 months)
2. Marked as 'Unknown' (retry failed fetches)

This should be run quarterly or added to a cron job.

Usage:
    python scripts/update_sectors.py
    python scripts/update_sectors.py --refresh-days 60  # Custom refresh period
    python scripts/update_sectors.py --retry-unknown    # Only retry unknowns
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sector_classification import (
    add_or_update_sectors,
    get_sector_summary,
    get_symbols_needing_refresh,
    load_sector_classifications,
    REFRESH_DAYS,
)


def main():
    """Update stale sector classifications."""
    parser = argparse.ArgumentParser(
        description='Update stale sector classifications (quarterly refresh)',
        epilog='Example: python scripts/update_sectors.py'
    )
    parser.add_argument(
        '--refresh-days',
        type=int,
        default=REFRESH_DAYS,
        help=f'Days before refresh needed (default: {REFRESH_DAYS})'
    )
    parser.add_argument(
        '--retry-unknown',
        action='store_true',
        help='Only retry symbols marked as Unknown'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔄 UPDATE SECTOR CLASSIFICATIONS")
    print("=" * 80)
    print()
    
    # Load existing classifications
    df = load_sector_classifications()
    
    if df is None or df.empty:
        print("❌ No existing sector classifications found")
        print("   Run fetch_sectors.py first to create initial data")
        return
    
    print(f"📋 Current classifications: {len(df)} symbols")
    print()
    
    # Determine symbols to update
    if args.retry_unknown:
        # Only retry Unknown symbols
        symbols_to_update = df[df['sector'] == 'Unknown']['symbol'].tolist()
        print(f"🔍 Mode: Retry Unknown only")
        print(f"   Found {len(symbols_to_update)} Unknown symbols")
    else:
        # Update stale symbols (older than refresh_days)
        symbols_to_update = get_symbols_needing_refresh(df, args.refresh_days)
        print(f"🔍 Mode: Refresh stale symbols (>{args.refresh_days} days)")
        print(f"   Found {len(symbols_to_update)} stale symbols")
    
    if not symbols_to_update:
        print()
        print("✅ All sectors are up to date - no refresh needed")
        print()
        
        # Show current summary
        print("📊 Current Sector Breakdown:")
        print()
        summary = get_sector_summary()
        for _, row in summary.head(10).iterrows():
            print(f"   {row['sector']:25s} {row['count']:4d} ({row['percentage']:5.1f}%)")
        
        return
    
    print()
    print("=" * 80)
    print("🚀 STARTING UPDATE")
    print("=" * 80)
    print()
    
    # Update sectors
    updated_df = add_or_update_sectors(symbols_to_update, force_refresh=True)
    
    print()
    print("=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print()
    
    print(f"✅ Updated {len(symbols_to_update)} symbols")
    print(f"   Total symbols in database: {len(updated_df)}")
    print()
    
    # Count unknowns
    unknown_count = (updated_df['sector'] == 'Unknown').sum()
    if unknown_count > 0:
        print(f"⚠️  Symbols still unknown: {unknown_count}")
        unknown_symbols = updated_df[
            updated_df['sector'] == 'Unknown'
        ]['symbol'].tolist()
        print(f"   {', '.join(unknown_symbols[:10])}")
        if len(unknown_symbols) > 10:
            print(f"   ... and {len(unknown_symbols) - 10} more")
        print()
    
    # Show updated sector breakdown
    print("📈 Updated Sector Breakdown:")
    print()
    summary = get_sector_summary()
    
    for _, row in summary.head(15).iterrows():
        sector = row['sector']
        count = row['count']
        pct = row['percentage']
        
        # Create bar visualization
        bar_length = int(pct / 2)
        bar = '█' * bar_length
        
        print(f"   {sector:25s} {count:4d} ({pct:5.1f}%) {bar}")
    
    if len(summary) > 15:
        print(f"   ... and {len(summary) - 15} more sectors")
    
    print()
    print("=" * 80)
    print("✅ UPDATE COMPLETE")
    print("=" * 80)
    print()
    print("Next quarterly update:")
    print(f"  python scripts/update_sectors.py")
    print()


if __name__ == '__main__':
    main()
