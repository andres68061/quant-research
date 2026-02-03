#!/usr/bin/env python3
"""
Add quoteType field to existing sector classifications.

This script updates the existing sector_classifications.parquet file
to include the quoteType field for accurate asset type detection.

Usage:
    python scripts/add_quote_type.py
    python scripts/add_quote_type.py --force  # Re-fetch all
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path BEFORE imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sector_classification import (
    add_or_update_sectors,
    load_sector_classifications,
)


def main():
    """Add quoteType to existing sector classifications."""
    parser = argparse.ArgumentParser(
        description='Add quoteType field to sector classifications',
        epilog='Example: python scripts/add_quote_type.py'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-fetch all symbols (even if quoteType exists)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("📊 ADD QUOTETYPE TO SECTOR CLASSIFICATIONS")
    print("=" * 80)
    print()

    # Load existing classifications
    df = load_sector_classifications()

    if df is None or df.empty:
        print("❌ No existing sector classifications found")
        print("   Run: python scripts/fetch_sectors.py")
        return

    print(f"📋 Current classifications: {len(df)} symbols")

    # Check if quoteType column exists
    if 'quoteType' in df.columns and not args.force:
        print("✅ quoteType column already exists")

        # Count how many have Unknown quoteType
        unknown_count = (df['quoteType'] == 'Unknown').sum()

        if unknown_count == 0:
            print("✅ All symbols have quoteType - no update needed")
            print()

            # Show breakdown
            print("📊 Asset Type Breakdown:")
            print(df['quoteType'].value_counts())
            return
        else:
            print(f"⚠️  {unknown_count} symbols have Unknown quoteType")
            print("   Will re-fetch these symbols")

            # Get symbols with Unknown quoteType
            symbols_to_update = df[df['quoteType']
                                   == 'Unknown']['symbol'].tolist()
    else:
        if args.force:
            print("🔄 Force refresh enabled - will re-fetch all symbols")
        else:
            print("📥 quoteType column not found - will fetch for all symbols")

        # Update all symbols
        symbols_to_update = df['symbol'].tolist()

    print()
    print(f"🚀 Updating {len(symbols_to_update)} symbols...")
    print("   This will take approximately {:.1f} minutes".format(
        len(symbols_to_update) * 0.5 / 60))
    print()

    # Re-fetch with quoteType
    updated_df = add_or_update_sectors(symbols_to_update, force_refresh=True)

    print()
    print("=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print()

    if 'quoteType' in updated_df.columns:
        print("✅ quoteType column added successfully")
        print()

        # Show asset type breakdown
        print("📈 Asset Type Breakdown:")
        print()
        type_counts = updated_df['quoteType'].value_counts()

        for asset_type, count in type_counts.items():
            pct = (count / len(updated_df) * 100)
            bar_length = int(pct / 2)
            bar = '█' * bar_length
            print(f"   {asset_type:20s} {count:4d} ({pct:5.1f}%) {bar}")

        print()

        # Show examples of each type
        print("📋 Examples by Type:")
        print()

        for asset_type in type_counts.index[:5]:  # Top 5 types
            examples = updated_df[updated_df['quoteType']
                                  == asset_type]['symbol'].head(5).tolist()
            print(f"   {asset_type:15s}: {', '.join(examples)}")

        print()

        # Highlight non-EQUITY types
        non_equity = updated_df[~updated_df['quoteType'].isin(
            ['EQUITY', 'Unknown'])]
        if not non_equity.empty:
            print(f"🎯 Found {len(non_equity)} non-stock assets:")
            print()
            for _, row in non_equity.iterrows():
                print(
                    f"   {row['symbol']:8s} - {row['quoteType']:15s} - {row['sector']}")

    else:
        print("⚠️  quoteType column not added - something went wrong")

    print()
    print("=" * 80)
    print("✅ UPDATE COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  - Portfolio simulator will now use accurate asset types")
    print("  - Asset type filters will work correctly")
    print("  - No more false ETF classifications")
    print()


if __name__ == '__main__':
    main()
