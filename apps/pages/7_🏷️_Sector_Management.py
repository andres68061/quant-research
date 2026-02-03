"""
Manual Sector Classification Management Page.

This page allows manual assignment of sector classifications for stocks
marked as 'Unknown' and provides trading status information.
"""

from src.data.sector_classification import (
    load_sector_classifications,
    save_sector_classifications,
)
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

# Project imports
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Constants
MANUAL_OVERRIDES_FILE = ROOT / "data" / "sectors" / "manual_overrides.json"
DELISTED_STOCKS_FILE = ROOT / "data" / "sectors" / "delisted_stocks.json"

# Yahoo Finance sectors (for dropdown)
YAHOO_SECTORS = [
    "Technology",
    "Healthcare",
    "Financial Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Industrials",
    "Energy",
    "Basic Materials",
    "Real Estate",
    "Communication Services",
    "Utilities",
    "Unknown"
]

# Common industries by sector (examples - not exhaustive)
INDUSTRIES_BY_SECTOR = {
    "Technology": ["Software", "Semiconductors", "IT Services", "Consumer Electronics", "Hardware"],
    "Healthcare": ["Pharmaceuticals", "Biotechnology", "Medical Devices", "Healthcare Plans", "Diagnostics"],
    "Financial Services": ["Banks", "Insurance", "Asset Management", "Credit Services", "Capital Markets"],
    "Consumer Cyclical": ["Retail", "Automotive", "Restaurants", "Leisure", "Apparel"],
    "Consumer Defensive": ["Food Products", "Beverages", "Household Products", "Tobacco", "Discount Stores"],
    "Industrials": ["Aerospace & Defense", "Construction", "Machinery", "Transportation", "Business Services"],
    "Energy": ["Oil & Gas", "Renewable Energy", "Oil & Gas Equipment", "Integrated Oil & Gas"],
    "Basic Materials": ["Chemicals", "Metals & Mining", "Steel", "Paper & Packaging", "Gold"],
    "Real Estate": ["REITs", "Real Estate Services", "Real Estate Development"],
    "Communication Services": ["Telecom", "Media", "Entertainment", "Publishing", "Broadcasting"],
    "Utilities": ["Electric Utilities", "Gas Utilities", "Water Utilities", "Renewable Utilities"],
    "Unknown": ["Unknown"]
}


def load_manual_overrides():
    """Load manual sector overrides from JSON."""
    if not MANUAL_OVERRIDES_FILE.exists():
        return {}

    try:
        with open(MANUAL_OVERRIDES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading manual overrides: {e}")
        return {}


def save_manual_override(symbol, sector, industry, industry_key, sector_key, reason):
    """Save a manual sector override."""
    overrides = load_manual_overrides()

    overrides[symbol] = {
        'sector': sector,
        'industry': industry,
        'industryKey': industry_key,
        'sectorKey': sector_key,
        'reason': reason,
        'updated_at': datetime.now().isoformat(),
    }

    MANUAL_OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MANUAL_OVERRIDES_FILE, 'w') as f:
        json.dump(overrides, f, indent=2)

    # Also update the main classifications file
    df = load_sector_classifications()

    if df is not None and not df.empty:
        # Update the row for this symbol
        mask = df['symbol'] == symbol
        if mask.any():
            df.loc[mask, 'sector'] = sector
            df.loc[mask, 'industry'] = industry
            df.loc[mask, 'industryKey'] = industry_key
            df.loc[mask, 'sectorKey'] = sector_key
            df.loc[mask, 'last_updated'] = datetime.now().isoformat()
        else:
            # Add new row
            new_row = pd.DataFrame([{
                'symbol': symbol,
                'sector': sector,
                'industry': industry,
                'industryKey': industry_key,
                'sectorKey': sector_key,
                'last_updated': datetime.now().isoformat(),
            }])
            df = pd.concat([df, new_row], ignore_index=True)

        save_sector_classifications(df)


def load_delisted_stocks():
    """Load delisted stocks information."""
    if not DELISTED_STOCKS_FILE.exists():
        return {}

    try:
        with open(DELISTED_STOCKS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading delisted stocks: {e}")
        return {}


def mark_as_delisted(symbol, reason):
    """Mark a stock as delisted."""
    delisted = load_delisted_stocks()

    delisted[symbol] = {
        'marked_at': datetime.now().isoformat(),
        'reason': reason,
    }

    DELISTED_STOCKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(DELISTED_STOCKS_FILE, 'w') as f:
        json.dump(delisted, f, indent=2)


def check_trading_status(symbol):
    """
    Check if a stock is currently trading.

    Returns:
        dict: {'is_trading': bool, 'last_price': float, 'last_date': str, 'error': str}
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d')

        if hist.empty:
            return {
                'is_trading': False,
                'last_price': None,
                'last_date': None,
                'error': 'No recent trading data'
            }

        last_date = hist.index[-1]
        last_price = hist['Close'].iloc[-1]

        # Check if last trade was recent (within 7 days)
        days_since = (pd.Timestamp.now() - last_date).days
        is_trading = days_since <= 7

        return {
            'is_trading': is_trading,
            'last_price': last_price,
            'last_date': last_date.strftime('%Y-%m-%d'),
            'error': None,
            'days_since': days_since
        }

    except Exception as e:
        return {
            'is_trading': False,
            'last_price': None,
            'last_date': None,
            'error': str(e)
        }


def generate_keys(sector, industry):
    """Generate sectorKey and industryKey from human-readable names."""
    sector_key = sector.lower().replace(' ', '-').replace('&', 'and')
    industry_key = industry.lower().replace(' ', '-').replace('&', 'and')
    return sector_key, industry_key


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Sector Management",
                   page_icon="🏷️", layout="wide")

st.title("🏷️ Manual Sector Classification Management")

st.markdown("""
Manually assign sector classifications for stocks marked as 'Unknown' or update existing classifications.
This page also checks if stocks are still trading.
""")

# Load data
df_sectors = load_sector_classifications()
manual_overrides = load_manual_overrides()
delisted_stocks = load_delisted_stocks()

if df_sectors is None or df_sectors.empty:
    st.error(
        "❌ No sector classifications found. Run `python scripts/fetch_sectors.py` first.")
    st.stop()

# ============================================================================
# SIDEBAR: FILTERS
# ============================================================================

st.sidebar.header("🔍 Filters")

show_filter = st.sidebar.radio(
    "Show:",
    ["Unknown Only", "All Stocks", "Manually Overridden", "Delisted"]
)

if show_filter == "Unknown Only":
    df_display = df_sectors[df_sectors['sector'] == 'Unknown'].copy()
elif show_filter == "Manually Overridden":
    df_display = df_sectors[df_sectors['symbol'].isin(
        manual_overrides.keys())].copy()
elif show_filter == "Delisted":
    df_display = df_sectors[df_sectors['symbol'].isin(
        delisted_stocks.keys())].copy()
else:
    df_display = df_sectors.copy()

# Search
search_term = st.sidebar.text_input("🔎 Search by symbol", "")
if search_term:
    df_display = df_display[df_display['symbol'].str.contains(
        search_term.upper())]

st.sidebar.markdown(f"**Showing {len(df_display)} stocks**")

# ============================================================================
# MAIN CONTENT: STATISTICS
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_stocks = len(df_sectors)
    st.metric("📊 Total Stocks", total_stocks)

with col2:
    unknown_count = (df_sectors['sector'] == 'Unknown').sum()
    unknown_pct = (unknown_count / total_stocks *
                   100) if total_stocks > 0 else 0
    st.metric("❓ Unknown Sector", f"{unknown_count} ({unknown_pct:.1f}%)")

with col3:
    override_count = len(manual_overrides)
    st.metric("✏️ Manual Overrides", override_count)

with col4:
    delisted_count = len(delisted_stocks)
    st.metric("🚫 Marked Delisted", delisted_count)

st.markdown("---")

# ============================================================================
# MAIN CONTENT: STOCK LIST WITH ACTIONS
# ============================================================================

st.subheader(f"📋 {show_filter} Stocks")

if df_display.empty:
    st.info("No stocks to display with the current filter.")
    st.stop()

# Sort by symbol
df_display = df_display.sort_values('symbol').reset_index(drop=True)

# Display with actions
for idx, row in df_display.iterrows():
    symbol = row['symbol']
    current_sector = row['sector']
    current_industry = row['industry']

    # Check if manually overridden
    is_overridden = symbol in manual_overrides
    is_delisted = symbol in delisted_stocks

    # Create expandable section for each stock
    with st.expander(f"**{symbol}** - {current_sector} / {current_industry} {'✏️' if is_overridden else ''} {'🚫' if is_delisted else ''}"):

        # Trading Status Section
        st.markdown("### 📈 Trading Status")

        if st.button(f"Check Trading Status", key=f"check_{symbol}"):
            with st.spinner(f"Checking {symbol}..."):
                status = check_trading_status(symbol)

                if status['is_trading']:
                    st.success(f"✅ **{symbol} is actively trading**")
                    st.write(f"- Last Price: ${status['last_price']:.2f}")
                    st.write(f"- Last Trade: {status['last_date']}")
                    st.write(f"- Days Since: {status['days_since']}")
                else:
                    st.warning(f"⚠️ **{symbol} is NOT actively trading**")
                    if status['last_date']:
                        st.write(
                            f"- Last Trade: {status['last_date']} ({status['days_since']} days ago)")
                        st.write(f"- Last Price: ${status['last_price']:.2f}")
                    if status['error']:
                        st.write(f"- Error: {status['error']}")

                    # Option to mark as delisted
                    if not is_delisted:
                        if st.button(f"Mark {symbol} as Delisted", key=f"mark_delisted_{symbol}"):
                            mark_as_delisted(
                                symbol, status['error'] or "No recent trading data")
                            st.success(f"Marked {symbol} as delisted")
                            st.rerun()

        # Show delisted info if marked
        if is_delisted:
            delisted_info = delisted_stocks[symbol]
            st.info(f"🚫 **Marked as Delisted**")
            st.write(f"- Marked At: {delisted_info['marked_at']}")
            st.write(f"- Reason: {delisted_info['reason']}")

        st.markdown("---")

        # Manual Classification Section
        st.markdown("### 🏷️ Manual Classification")

        col1, col2 = st.columns(2)

        with col1:
            # Sector selection
            current_sector_idx = YAHOO_SECTORS.index(
                current_sector) if current_sector in YAHOO_SECTORS else 0
            new_sector = st.selectbox(
                "Sector",
                YAHOO_SECTORS,
                index=current_sector_idx,
                key=f"sector_{symbol}"
            )

            # Industry selection (filtered by sector)
            industries = INDUSTRIES_BY_SECTOR.get(new_sector, ["Unknown"])
            current_industry_idx = industries.index(
                current_industry) if current_industry in industries else 0
            new_industry = st.selectbox(
                "Industry",
                industries + ["Other (type below)"],
                index=current_industry_idx,
                key=f"industry_{symbol}"
            )

            # Custom industry input
            if new_industry == "Other (type below)":
                new_industry = st.text_input(
                    "Custom Industry",
                    value=current_industry if current_industry != "Unknown" else "",
                    key=f"custom_industry_{symbol}"
                )

        with col2:
            # Auto-generate keys
            sector_key, industry_key = generate_keys(new_sector, new_industry)

            st.text_input("Sector Key (auto-generated)",
                          value=sector_key, disabled=True, key=f"sk_{symbol}")
            st.text_input("Industry Key (auto-generated)",
                          value=industry_key, disabled=True, key=f"ik_{symbol}")

        # Reason for override
        reason = st.text_area(
            "Reason for Manual Override",
            value=manual_overrides[symbol]['reason'] if is_overridden else "",
            placeholder="e.g., 'Yahoo Finance has incorrect sector' or 'Delisted stock, assigned based on historical data'",
            key=f"reason_{symbol}"
        )

        # Show current values
        st.markdown("**Current Values:**")
        quote_type = row.get('quoteType', 'N/A') if 'quoteType' in row.index else 'N/A'
        st.code(f"""
sector: {row['sector']}
industry: {row['industry']}
industryKey: {row['industryKey']}
sectorKey: {row['sectorKey']}
quoteType: {quote_type}
last_updated: {row['last_updated']}
""")

        # Save button
        if st.button(f"💾 Save Classification for {symbol}", key=f"save_{symbol}", type="primary"):
            if not reason.strip():
                st.error("Please provide a reason for the manual override.")
            else:
                save_manual_override(
                    symbol=symbol,
                    sector=new_sector,
                    industry=new_industry,
                    industry_key=industry_key,
                    sector_key=sector_key,
                    reason=reason
                )
                st.success(f"✅ Saved classification for {symbol}")
                st.rerun()

        # Show override history if exists
        if is_overridden:
            st.markdown("**Override History:**")
            override_info = manual_overrides[symbol]
            st.info(f"""
            - Updated At: {override_info['updated_at']}
            - Reason: {override_info['reason']}
            """)

st.markdown("---")

# ============================================================================
# BULK OPERATIONS
# ============================================================================

st.subheader("🔄 Bulk Operations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Mark Multiple as Delisted")
    delisted_symbols = st.text_area(
        "Enter symbols (one per line or comma-separated)",
        placeholder="AAPL, MSFT, GOOGL\nor\nAAPL\nMSFT\nGOOGL",
        key="bulk_delisted"
    )
    delisted_reason = st.text_input(
        "Reason", value="Bulk delisted", key="bulk_delisted_reason")

    if st.button("Mark as Delisted", key="bulk_delisted_btn"):
        symbols = [s.strip().upper() for s in delisted_symbols.replace(
            ',', '\n').split('\n') if s.strip()]
        if symbols:
            for symbol in symbols:
                mark_as_delisted(symbol, delisted_reason)
            st.success(f"Marked {len(symbols)} stocks as delisted")
            st.rerun()

with col2:
    st.markdown("### Assign Same Sector to Multiple")
    bulk_symbols = st.text_area(
        "Enter symbols (one per line or comma-separated)",
        placeholder="AAPL, MSFT, GOOGL",
        key="bulk_assign"
    )
    bulk_sector = st.selectbox("Sector", YAHOO_SECTORS, key="bulk_sector")
    bulk_industry = st.selectbox(
        "Industry",
        INDUSTRIES_BY_SECTOR.get(bulk_sector, ["Unknown"]),
        key="bulk_industry"
    )
    bulk_reason = st.text_input(
        "Reason", value="Bulk assignment", key="bulk_reason")

    if st.button("Assign to All", key="bulk_assign_btn"):
        symbols = [s.strip().upper() for s in bulk_symbols.replace(
            ',', '\n').split('\n') if s.strip()]
        if symbols:
            sector_key, industry_key = generate_keys(
                bulk_sector, bulk_industry)
            for symbol in symbols:
                save_manual_override(
                    symbol, bulk_sector, bulk_industry, industry_key, sector_key, bulk_reason)
            st.success(f"Assigned classification to {len(symbols)} stocks")
            st.rerun()

st.markdown("---")

# ============================================================================
# DOWNLOAD SECTION
# ============================================================================

st.subheader("📥 Download Data")

col1, col2, col3 = st.columns(3)

with col1:
    # Download current classifications
    csv_data = df_sectors.to_csv(index=False)
    st.download_button(
        label="📊 Download All Classifications (CSV)",
        data=csv_data,
        file_name=f"sector_classifications_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    # Download manual overrides
    if manual_overrides:
        override_json = json.dumps(manual_overrides, indent=2)
        st.download_button(
            label="✏️ Download Manual Overrides (JSON)",
            data=override_json,
            file_name=f"manual_overrides_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

with col3:
    # Download delisted list
    if delisted_stocks:
        delisted_json = json.dumps(delisted_stocks, indent=2)
        st.download_button(
            label="🚫 Download Delisted List (JSON)",
            data=delisted_json,
            file_name=f"delisted_stocks_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

st.markdown("---")
st.caption("""
💡 **Tips:**
- Use "Check Trading Status" to verify if a stock is still active
- Mark delisted stocks to keep your database clean
- Provide clear reasons for manual overrides for future reference
- Bulk operations help with large-scale corrections
""")
