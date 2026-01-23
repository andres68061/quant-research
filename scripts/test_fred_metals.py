#!/usr/bin/env python3
"""
Test FRED metals data fetching.

Verifies which series IDs are valid and can fetch data.
"""

import sys
from pathlib import Path

import requests
from fredapi import Fred

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import FRED_API_KEY

if not FRED_API_KEY:
    print("❌ FRED_API_KEY not found in environment!")
    print("Add it to .env file: FRED_API_KEY=your_key")
    sys.exit(1)

print(f"✅ FRED API Key found: {FRED_API_KEY[:10]}...")
print()

# Test series IDs
METALS_TO_TEST = {
    "Gold": [
        "GOLDAMGBD228NLBM",  # London fixing
        "GOLDPMGBD228NLBM",  # London PM fixing
        "WPU10210301",       # Producer Price Index
    ],
    "Silver": [
        "SLVPRUSD",          # Handy & Harman
        "DSLVNS",            # Daily silver price
    ],
    "Copper": [
        "PCOPPUSDM",         # Global price (monthly)
        "PCOPPUSD",          # Global price (annual)
        "WPU101707",         # Producer Price Index
    ],
    "Platinum": [
        "PLATINUMPRICE",     # Suggested ID
        "WPUSI019011",       # PPI Platinum
    ],
    "Palladium": [
        "PALLADIUMPRICE",    # Suggested ID
        "WPUSI01901121",     # PPI Palladium
    ],
}

print("=" * 80)
print("Testing FRED Series IDs")
print("=" * 80)
print()

# Method 1: Using fredapi library
print("Method 1: Using fredapi library")
print("-" * 80)

fred = Fred(api_key=FRED_API_KEY)

valid_series = {}

for metal, series_ids in METALS_TO_TEST.items():
    print(f"\n{metal}:")
    for series_id in series_ids:
        try:
            data = fred.get_series(series_id, observation_start='2020-01-01')
            if data is not None and len(data) > 0:
                print(f"  ✅ {series_id}: {len(data)} observations")
                print(f"     Latest: {data.iloc[-1]:.2f} ({data.index[-1].strftime('%Y-%m-%d')})")
                if metal not in valid_series:
                    valid_series[metal] = series_id
            else:
                print(f"  ⚠️  {series_id}: No data returned")
        except Exception as e:
            print(f"  ❌ {series_id}: {str(e)}")

print()
print("=" * 80)
print("Method 2: Using raw FRED API")
print("-" * 80)

def test_series_api(series_id):
    """Test series using raw API."""
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': '2020-01-01',
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                obs = data['observations']
                latest = obs[-1]
                return True, len(obs), latest['value'], latest['date']
        return False, 0, None, None
    except Exception as e:
        return False, 0, None, str(e)

for metal, series_ids in METALS_TO_TEST.items():
    print(f"\n{metal}:")
    for series_id in series_ids:
        success, count, value, date = test_series_api(series_id)
        if success:
            print(f"  ✅ {series_id}: {count} observations")
            print(f"     Latest: {value} ({date})")
        else:
            print(f"  ❌ {series_id}: Failed")

print()
print("=" * 80)
print("RECOMMENDED SERIES IDs")
print("=" * 80)
print()

if valid_series:
    for metal, series_id in valid_series.items():
        print(f"{metal:12} -> {series_id}")
else:
    print("No valid series found!")

print()
print("=" * 80)
print("Search for alternative series")
print("=" * 80)
print()

def search_fred_series(search_term):
    """Search FRED for series."""
    url = "https://api.stlouisfed.org/fred/series/search"
    params = {
        'search_text': search_term,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'limit': 5,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'seriess' in data:
                return data['seriess']
    except Exception as e:
        print(f"Search error: {e}")
    return []

search_terms = [
    ("Gold Price", "gold price"),
    ("Silver Price", "silver price"),
    ("Copper Price", "copper price"),
    ("Platinum Price", "platinum price"),
    ("Palladium Price", "palladium price"),
]

for name, term in search_terms:
    print(f"\n{name} ('{term}'):")
    results = search_fred_series(term)
    
    if results:
        for result in results[:3]:  # Show top 3
            print(f"  {result['id']:20} - {result['title'][:60]}")
            print(f"    Freq: {result.get('frequency', 'N/A')}, Updated: {result.get('last_updated', 'N/A')}")
    else:
        print("  No results found")

print()
print("=" * 80)
print("Testing Complete!")
print("=" * 80)

