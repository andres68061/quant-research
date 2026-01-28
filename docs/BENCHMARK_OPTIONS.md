# Benchmark Options Guide

## Overview

The Portfolio Simulator offers multiple benchmark options for comparing your strategy performance. This guide explains when to use each benchmark and their limitations.

## Available Benchmarks

### 1. S&P 500 (^GSPC) ✅ **Recommended for most use cases**

**What it is:**
- Official S&P 500 index data from Yahoo Finance
- Cap-weighted index of ~500 large-cap U.S. stocks
- Complete historical data from 1927 onwards

**When to use:**
- ✅ General benchmarking across any time period
- ✅ Comparing against "the market"
- ✅ Long-term backtests (10+ years)
- ✅ When you need reliable, complete data

**Advantages:**
- Complete historical coverage
- No missing data
- Industry standard benchmark
- Includes dividends (adjusted close prices)

**Limitations:**
- Survivorship bias: Only includes current/recent constituents
- Doesn't reflect actual historical S&P 500 membership changes

---

### 2. S&P 500 Reconstructed (2020+) ⚠️ **Use with caution - limited coverage**

**What it is:**
- Reconstructed S&P 500 using point-in-time historical constituents
- Based on historical S&P 500 membership data (1996-2026)
- Eliminates survivorship bias by including delisted/acquired companies

**Coverage by Period:**

| Period | Coverage | Recommendation |
|--------|----------|----------------|
| **2024-2026** | 97-99% (501/503 constituents) | ✅ **Excellent** - Use confidently |
| **2020-2023** | 93-96% | ⚠️ **Use with caution** - Some bias remains |
| **Before 2020** | <93% | ❌ **Not recommended** - Use ^GSPC instead |

**When to use:**
- ✅ Recent backtests (2024-2026) where you want to eliminate survivorship bias
- ✅ Comparing factor strategies against a "realistic" S&P 500
- ✅ When you want to include bankruptcies and delistings in benchmark

**Weighting Options:**
- **Equal Weight**: Each constituent gets 1/N weight (similar to RSP ETF)
- **Cap-Weighted**: Weighted by market capitalization (requires market cap data)

**Advantages:**
- Eliminates survivorship bias for recent periods
- Includes failed companies (Lehman Brothers, etc.)
- More realistic representation of actual S&P 500 performance

**Limitations:**
- ❌ Missing 288 historical constituents (delisted, bankruptcies, ticker changes)
- ⚠️ Coverage drops significantly before 2020
- ⚠️ 3-7% missing constituents introduce tracking error
- ⚠️ Not suitable for long-term historical analysis

**Why symbols are missing:**
- Bankruptcies (LEHMQ, AAMRQ, WAMUQ) - no longer traded
- Acquisitions (TWTR, XLNX, CELG) - merged into other companies
- Ticker changes (BRK.B, RDS.A) - class A/B shares with different symbols
- Early delistings (before 2010) - limited historical data availability

---

### 3. Equal Weight Universe

**What it is:**
- All stocks in your dataset weighted equally (1/N each)
- Similar to S&P 500 Equal Weight Index (RSP)

**When to use:**
- ✅ Removing large-cap bias
- ✅ Testing pure diversification effects
- ✅ Comparing against equal-weight strategies

**Advantages:**
- Simple and transparent
- No market cap data required
- Shows diversification benefit

**Limitations:**
- Not a standard market benchmark
- May include non-S&P 500 stocks if present in your data

---

### 4. Synthetic (Custom Mix)

**What it is:**
- Custom blend of two benchmarks
- Example: 60% S&P 500 + 40% Equal Weight

**When to use:**
- ✅ Creating custom risk profiles
- ✅ Blending growth and value exposures
- ✅ Testing against hybrid strategies

**Examples:**
- 60% S&P 500 + 40% Equal Weight Universe
- 70% S&P 500 + 30% NASDAQ (after adding ^IXIC)
- 80% Large Cap + 20% Small Cap (after adding ^RUT)

---

## Recommendation by Use Case

### Long-term backtests (10+ years)
**Use:** S&P 500 (^GSPC)
- Complete data, no gaps
- Industry standard

### Recent backtests (2024-2026)
**Use:** S&P 500 Reconstructed (2020+) with 97-99% coverage
- Eliminates survivorship bias
- More realistic performance

### Factor strategy testing
**Use:** S&P 500 Reconstructed (2020+) for 2024-2026, ^GSPC for longer periods
- Shows true factor performance
- Includes failures

### Equal-weight strategies
**Use:** Equal Weight Universe or S&P 500 Reconstructed (Equal Weight)
- Removes cap-weighting bias

---

## Data Quality Summary

### S&P 500 (^GSPC)
- ✅ Complete: 100% coverage, 1927-present
- ✅ Reliable: Official Yahoo Finance data
- ⚠️ Survivorship bias: Yes

### S&P 500 Reconstructed
- ✅ 2024-2026: 99.6% coverage (501/503)
- ✅ 2025: 98.4% coverage (495/503)
- ⚠️ 2024: 97.6% coverage (491/503)
- ⚠️ 2022: 95.6% coverage (481/503)
- ⚠️ 2020: 93.1% coverage (470/505)
- ❌ Before 2020: <93% coverage

---

## Adding More Indices

To add NASDAQ, Russell 2000, or other indices:

```bash
python scripts/add_symbol.py ^IXIC ^RUT ^DJI
```

Where:
- `^IXIC` = NASDAQ Composite (tech-heavy)
- `^RUT` = Russell 2000 (small-cap)
- `^DJI` = Dow Jones Industrial Average

---

## Technical Details

### Missing Symbols Breakdown (288 total)

| Category | Count | Status |
|----------|-------|--------|
| Early Delistings (before 2010) | 126 | ❌ No data available |
| Recent Delistings (2010-2020) | 88 | ❌ No data available |
| Recent Delistings (2020+) | 42 | ❌ No data available |
| Bankruptcies (Q suffix) | 24 | ❌ Delisted, no data |
| Class A Shares | 6 | ⚠️ Ticker format issues |
| Class B Shares | 2 | ⚠️ Ticker format issues |

### Notable Missing Symbols
- LEHMQ: Lehman Brothers (bankruptcy, 2008)
- AAMRQ: American Airlines (bankruptcy, 2011)
- WAMUQ: Washington Mutual (bankruptcy, 2008)
- TWTR: Twitter (acquired by Elon Musk, 2022)
- XLNX: Xilinx (acquired by AMD, 2022)
- CELG: Celgene (acquired by Bristol-Myers Squibb, 2019)

---

## Best Practices

1. **For production/publication:** Use S&P 500 (^GSPC) for consistency and completeness

2. **For research (2024-2026):** Use S&P 500 Reconstructed to eliminate survivorship bias

3. **For long-term analysis:** Always use ^GSPC - reconstructed data has insufficient coverage

4. **Document your choice:** Always note which benchmark you used and its limitations

5. **Compare multiple benchmarks:** Run your strategy against both ^GSPC and Reconstructed to see the impact of survivorship bias

---

## References

- S&P 500 Historical Components: `data/S&P 500 Historical Components & Changes(01-17-2026).csv`
- Failed symbols analysis: `python scripts/analyze_failed_symbols.py`
- Coverage analysis: See "Data Quality Summary" above
