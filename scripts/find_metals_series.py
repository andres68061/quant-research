#!/usr/bin/env python3
"""
Find correct FRED series IDs for metals.
"""

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import FRED_API_KEY

def search_and_test(search_term, keywords):
    """Search for series and test which ones work."""
    print(f"\nSearching for: {search_term}")
    print("=" * 80)
    
    # Search
    url = "https://api.stlouisfed.org/fred/series/search"
    params = {
        'search_text': search_term,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'limit': 20,
    }
    
    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        print(f"Search failed: {response.status_code}")
        return []
    
    data = response.json()
    if 'seriess' not in data:
        print("No results")
        return []
    
    # Filter by keywords
    results = []
    for series in data['seriess']:
        title_lower = series['title'].lower()
        if any(kw in title_lower for kw in keywords):
            # Test if it has data
            obs_url = "https://api.stlouisfed.org/fred/series/observations"
            obs_params = {
                'series_id': series['id'],
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'limit': 1,
            }
            
            obs_response = requests.get(obs_url, params=obs_params, timeout=10)
            if obs_response.status_code == 200:
                obs_data = obs_response.json()
                if 'observations' in obs_data and len(obs_data['observations']) > 0:
                    results.append({
                        'id': series['id'],
                        'title': series['title'],
                        'frequency': series.get('frequency', 'N/A'),
                        'units': series.get('units', 'N/A'),
                        'last_updated': series.get('last_updated', 'N/A'),
                    })
    
    return results

# Search for each metal
metals = [
    ("Gold", "global price gold", ["price", "usd", "troy"]),
    ("Silver", "global price silver", ["price", "usd", "troy"]),
    ("Copper", "global price copper", ["price", "usd"]),
    ("Platinum", "global price platinum", ["price", "usd", "troy"]),
    ("Palladium", "global price palladium", ["price", "usd", "troy"]),
]

all_results = {}

for metal, search, keywords in metals:
    results = search_and_test(search, keywords)
    
    print(f"\nFound {len(results)} relevant series for {metal}:")
    for r in results[:5]:  # Show top 5
        print(f"\n  ID: {r['id']}")
        print(f"  Title: {r['title']}")
        print(f"  Frequency: {r['frequency']}")
        print(f"  Units: {r['units']}")
    
    if results:
        all_results[metal] = results[0]['id']  # Take the first one

print("\n" + "=" * 80)
print("RECOMMENDED SERIES IDs:")
print("=" * 80)

for metal, series_id in all_results.items():
    print(f"{metal:12} -> {series_id}")

# Test them all
print("\n" + "=" * 80)
print("TESTING ALL RECOMMENDED SERIES:")
print("=" * 80)

from fredapi import Fred
fred = Fred(api_key=FRED_API_KEY)

for metal, series_id in all_results.items():
    try:
        data = fred.get_series(series_id, observation_start='2020-01-01')
        print(f"\n✅ {metal} ({series_id}):")
        print(f"   Observations: {len(data)}")
        print(f"   Latest: {data.iloc[-1]:.2f} on {data.index[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"\n❌ {metal} ({series_id}): {e}")

