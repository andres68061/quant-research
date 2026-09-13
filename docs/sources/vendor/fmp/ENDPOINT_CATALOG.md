# FMP endpoint catalog

Single-file catalog of the **Financial Modeling Prep (FMP) stable API**: every
documented endpoint, including paths unavailable on this repo’s subscription.

- **Vendor:** Financial Modeling Prep
- **Base URL:** `https://financialmodelingprep.com/stable/`
- **Auth:** `?apikey=` query param or `apikey:` header. The key lives in `.env` as `FMP_API_KEY` (never commit it).
- **Live docs:** [FMP stable API](https://site.financialmodelingprep.com/developer/docs/stable)
- **Endpoint list fetched and tested:** 2026-08-18 from the live FMP page (263 documented examples covering **230 unique paths**).
- **Test method:** every documented example was called with this repo’s Premium key; transient 429/5xx responses were retried. The one documentation example missing a required parameter was retested with `symbol=AAPL`.

This repo is on **Premium** (750 calls/min advertised; the client throttles to ~500/min).
The sweep made 264 calls: all 263 documented examples plus one corrected retest. A path is
marked **works** only when its own documented request returned HTTP 200; inferred family
or sibling labels are no longer used.

**Observed result: 176 of 230 unique paths work; 54 return HTTP 402.**

Restricted paths include selected transcript/directory, batch quote, TTM statement,
market-snapshot, ETF/fund, Form 13F, ESG, and all bulk endpoints. The complete result is in
each row below.

Common query params: `symbol` / `symbols`, `from` / `to` (YYYY-MM-DD), `page` / `limit`,
`period` (`annual` / `quarter` / `FY` / `Q1`…), `cik`, `exchange`. Per-endpoint parameter
tables are JS-rendered on FMP’s site and are not copied here.

Shorter per-category notes, including traps this repo has already hit, live in
[README.md](README.md) and the sibling files in this folder.

## Access legend

| Access | Meaning |
|---|---|
| works — HTTP 200 | This exact documented path was called successfully on 2026-08-18 |
| blocked — HTTP 402 | This exact documented path returned Payment Required on 2026-08-18 |

## Counts

| Result | Unique paths |
|---|---:|
| works — HTTP 200 | 176 |
| blocked — HTTP 402 | 54 |
| **total tested** | **230** |

## Company Search

| Path | What it returns | Access |
|---|---|---|
| `/search-symbol` | **Stock Symbol Search.** Easily find the ticker symbol of any stock with the FMP Stock Symbol Search API. | works — HTTP 200 |
| `/search-name` | **Name Search.** Search for ticker symbols, company names, and exchange details for equity securities and ETFs listed on various exchanges with the FMP Name Search API. | works — HTTP 200 |
| `/search-cik` | **CIK.** Easily retrieve the Central Index Key (CIK) for publicly traded companies with the FMP CIK API. | works — HTTP 200 |
| `/search-cusip` | **CUSIP.** Easily search and retrieve financial securities information by CUSIP number using the FMP CUSIP API. | works — HTTP 200 |
| `/search-isin` | **ISIN.** Easily search and retrieve the International Securities Identification Number (ISIN) for financial securities using the FMP ISIN API. | works — HTTP 200 |
| `/company-screener` | **Stock Screener.** Discover stocks that align with your investment strategy using the FMP Stock Screener API. | works — HTTP 200 |
| `/search-exchange-variants` | **Exchange Variants.** Search across multiple public exchanges to find where a given stock symbol is listed using the FMP Exchange Variants API. | works — HTTP 200 |

## Stock Directory

| Path | What it returns | Access |
|---|---|---|
| `/stock-list` | **Company Symbols List.** Easily retrieve a comprehensive list of financial symbols with the FMP Company Symbols List API. | works — HTTP 200 |
| `/financial-statement-symbol-list` | **Financial Statement Symbols List.** Access a comprehensive list of companies with available financial statements through the FMP Financial Statement Symbols List API. | works — HTTP 200 |
| `/cik-list` | **CIK List.** Access a comprehensive database of CIK (Central Index Key) numbers for SEC-registered entities with the FMP CIK List API. | works — HTTP 200 |
| `/symbol-change` | **Stock Symbol Changes.** Stay informed about the latest stock symbol changes with the FMP Stock Symbol Changes API. | works — HTTP 200 |
| `/etf-list` | **ETF Symbol Search.** Quickly find ticker symbols and company names for Exchange Traded Funds (ETFs) using the FMP ETF Symbol Search API. | works — HTTP 200 |
| `/actively-trading-list` | **Actively Trading List.** List all actively trading companies and financial instruments with the FMP Actively Trading List API. | works — HTTP 200 |
| `/earnings-transcript-list` | **Earnings Transcript List.** Access available earnings transcripts for companies with the FMP Earnings Transcript List API. | blocked — HTTP 402 |
| `/available-exchanges` | **Available Exchanges.** Access a complete list of supported stock exchanges using the FMP Available Exchanges API. | works — HTTP 200 |
| `/available-sectors` | **Available Sectors.** Access a complete list of industry sectors using the FMP Available Sectors API. | works — HTTP 200 |
| `/available-industries` | **Available Industries.** Access a comprehensive list of industries where stock symbols are available using the FMP Available Industries API. | works — HTTP 200 |
| `/available-countries` | **Available Countries.** Access a comprehensive list of countries where stock symbols are available with the FMP Available Countries API. | works — HTTP 200 |

## Company Information

| Path | What it returns | Access |
|---|---|---|
| `/profile` | **Company Profile Data.** Access detailed company profile data with the FMP Company Profile Data API. | works — HTTP 200 |
| `/profile-cik` | **Company Profile by CIK.** Retrieve detailed company profile data by CIK (Central Index Key) with the FMP Company Profile by CIK API. | works — HTTP 200 |
| `/company-notes` | **Company Notes.** Retrieve detailed information about company-issued notes with the FMP Company Notes API. | works — HTTP 200 |
| `/stock-peers` | **Stock Peer Comparison.** Identify and compare companies within the same sector and market capitalization range using the FMP Stock Peer Comparison API. | works — HTTP 200 |
| `/delisted-companies` | **Delisted Companies.** Stay informed with the FMP Delisted Companies API. | works — HTTP 200 |
| `/employee-count` | **Company Employee Count.** Retrieve detailed workforce information for companies, including employee count, reporting period, and filing date. | works — HTTP 200 |
| `/historical-employee-count` | **Company Historical Employee Count.** Access historical employee count data for a company based on specific reporting periods. | works — HTTP 200 |
| `/market-capitalization` | **Company Market Capitalization.** Retrieve the market capitalization for a specific company on any given date using the FMP Company Market Capitalization API. | works — HTTP 200 |
| `/market-capitalization-batch` | **Batch Market Capitalization.** Retrieve market capitalization data for multiple companies in a single request with the FMP Batch Market Capitalization API. | works — HTTP 200 |
| `/historical-market-capitalization` | **Historical Market Capitalization.** Access historical market capitalization data for a company using the FMP Historical Market Capitalization API. | works — HTTP 200 |
| `/shares-float` | **Company Share Float and Liquidity.** Understand the liquidity and volatility of a stock with the FMP Company Share Float and Liquidity API. | works — HTTP 200 |
| `/shares-float-all` | **All Shares Float.** Access comprehensive shares float data for all available companies with the FMP All Shares Float API. | works — HTTP 200 |
| `/mergers-acquisitions-latest` | **Latest Mergers and Acquisitions.** Access real-time data on the latest mergers and acquisitions with the FMP Latest Mergers and Acquisitions API. | works — HTTP 200 |
| `/mergers-acquisitions-search` | **Search Mergers and Acquisitions.** Search for specific mergers and acquisitions data with the FMP Search Mergers and Acquisitions API. | works — HTTP 200 |
| `/key-executives` | **Company Executives.** Retrieve detailed information on company executives with the FMP Company Executives API. | works — HTTP 200 |
| `/governance-executive-compensation` | **Executive Compensation.** Retrieve comprehensive compensation data for company executives with the FMP Executive Compensation API. | works — HTTP 200 |
| `/executive-compensation-benchmark` | **Executive Compensation Benchmark.** Gain access to average executive compensation data across various industries with the FMP Executive Compensation Benchmark API. | works — HTTP 200 |

## Quote

| Path | What it returns | Access |
|---|---|---|
| `/quote` | **Stock Quote.** Access real-time stock quotes with the FMP Stock Quote API. | works — HTTP 200 |
| `/quote-short` | **Stock Quote Short.** Get quick snapshots of real-time stock quotes with the FMP Stock Quote Short API. | works — HTTP 200 |
| `/aftermarket-trade` | **Aftermarket Trade.** Track real-time trading activity occurring after regular market hours with the FMP Aftermarket Trade API. | works — HTTP 200 |
| `/aftermarket-quote` | **Aftermarket Quote.** Access real-time aftermarket quotes for stocks with the FMP Aftermarket Quote API. | works — HTTP 200 |
| `/stock-price-change` | **Stock Price Change.** Track stock price fluctuations in real-time with the FMP Stock Price Change API. | works — HTTP 200 |
| `/batch-quote` | **Stock Batch Quote.** Retrieve multiple real-time stock quotes in a single request with the FMP Stock Batch Quote API. | works — HTTP 200 |
| `/batch-quote-short` | **Stock Batch Quote Short.** Access real-time, short-form quotes for multiple stocks with the FMP Stock Batch Quote Short API. | works — HTTP 200 |
| `/batch-aftermarket-trade` | **Batch Aftermarket Trade.** Retrieve real-time aftermarket trading data for multiple stocks with the FMP Batch Aftermarket Trade API. | works — HTTP 200 |
| `/batch-aftermarket-quote` | **Batch Aftermarket Quote.** Retrieve real-time aftermarket quotes for multiple stocks with the FMP Batch Aftermarket Quote API. | works — HTTP 200 |
| `/batch-exchange-quote` | **Exchange Stock Quotes.** Retrieve real-time stock quotes for all listed stocks on a specific exchange with the FMP Exchange Stock Quotes API. | blocked — HTTP 402 |
| `/batch-mutualfund-quotes` | **Mutual Fund Price Quotes.** Access real-time quotes for mutual funds with the FMP Mutual Fund Price Quotes API. | blocked — HTTP 402 |
| `/batch-etf-quotes` | **ETF Price Quotes.** Get real-time price quotes for exchange-traded funds (ETFs) with the FMP ETF Price Quotes API. | blocked — HTTP 402 |
| `/batch-commodity-quotes` | **Real-Time Commodities Quotes.** Get up-to-the-minute quotes for commodities with the FMP Real-Time Commodities Quotes API. | blocked — HTTP 402 |
| `/batch-crypto-quotes` | **Full Cryptocurrency Quotes.** Access real-time cryptocurrency quotes with the FMP Full Cryptocurrency Quotes API. | blocked — HTTP 402 |
| `/batch-forex-quotes` | **Batch Forex Quote.** Retrieve real-time quotes for multiple forex currency pairs with the FMP Batch Forex Quote API. | blocked — HTTP 402 |
| `/batch-index-quotes` | **Stock Market Index Quotes.** Track real-time movements of major stock market indexes with the FMP Stock Market Index Quotes API. | blocked — HTTP 402 |

## Statements

| Path | What it returns | Access |
|---|---|---|
| `/income-statement` | **Income Statements.** Access detailed income statement data for publicly traded companies with the Income Statements API. | works — HTTP 200 |
| `/balance-sheet-statement` | **Balance Sheet Data.** Access detailed balance sheet statements for publicly traded companies with the Balance Sheet Data API. | works — HTTP 200 |
| `/cash-flow-statement` | **Cash Flow Statements.** Gain insights into a company's cash flow activities with the Cash Flow Statements API. | works — HTTP 200 |
| `/latest-financial-statements` | **Latest Financial Statements.** | blocked — HTTP 402 |
| `/income-statement-ttm` | **Income Statements TTM.** | blocked — HTTP 402 |
| `/balance-sheet-statement-ttm` | **Balance Sheet Statements TTM.** | blocked — HTTP 402 |
| `/cash-flow-statement-ttm` | **Cashflow Statements TTM.** | blocked — HTTP 402 |
| `/key-metrics` | **Financial Key Metrics.** Access essential financial metrics for a company with the FMP Financial Key Metrics API. | works — HTTP 200 |
| `/ratios` | **Financial Ratios.** Analyze a company's financial performance using the Financial Ratios API. | works — HTTP 200 |
| `/key-metrics-ttm` | **TTM Key Metrics.** Retrieve a comprehensive set of trailing twelve-month (TTM) key performance metrics with the TTM Key Metrics API. | works — HTTP 200 |
| `/ratios-ttm` | **TTM Ratios.** Gain access to trailing twelve-month (TTM) financial ratios with the TTM Ratios API. | works — HTTP 200 |
| `/financial-scores` | **Financial Health Scores.** Assess a company's financial strength using the Financial Health Scores API. | works — HTTP 200 |
| `/owner-earnings` | **Owner Earnings.** Retrieve a company's owner earnings with the Owner Earnings API, which provides a more accurate representation of cash available to shareholders by adjusting net income. | works — HTTP 200 |
| `/enterprise-values` | **Enterprise Values.** Access a company's enterprise value using the Enterprise Values API. | works — HTTP 200 |
| `/income-statement-growth` | **Income Statement Growth.** Track key financial growth metrics with the Income Statement Growth API. | works — HTTP 200 |
| `/balance-sheet-statement-growth` | **Balance Sheet Statement Growth.** Analyze the growth of key balance sheet items over time with the Balance Sheet Statement Growth API. | works — HTTP 200 |
| `/cash-flow-statement-growth` | **Cashflow Statement Growth.** Measure the growth rate of a company’s cash flow with the FMP Cashflow Statement Growth API. | works — HTTP 200 |
| `/financial-growth` | **Financial Statement Growth.** Analyze the growth of key financial statement items across income, balance sheet, and cash flow statements with the Financial Statement Growth API. | works — HTTP 200 |
| `/financial-reports-dates` | **Financial Reports Dates.** | works — HTTP 200 |
| `/financial-reports-json` | **Annual Reports on Form 10-K.** Access comprehensive annual reports with the FMP Annual Reports on Form 10-K API. | works — HTTP 200 |
| `/financial-reports-xlsx` | **Financial Reports Form 10-K XLSX.** Download detailed 10-K reports in XLSX format with the Financial Reports Form 10-K XLSX API. | works — HTTP 200 |
| `/revenue-product-segmentation` | **Revenue Product Segmentation.** Access detailed revenue breakdowns by product line with the Revenue Product Segmentation API. | works — HTTP 200 |
| `/revenue-geographic-segmentation` | **Revenue Geographic Segments.** Access detailed revenue breakdowns by geographic region with the Revenue Geographic Segments API. | works — HTTP 200 |
| `/income-statement-as-reported` | **As Reported Income Statements.** Retrieve income statements as they were reported by the company. | works — HTTP 200 |
| `/balance-sheet-statement-as-reported` | **As Reported Balance Statements.** Access balance sheets as reported by the company. | works — HTTP 200 |
| `/cash-flow-statement-as-reported` | **As Reported Cash Flow Statements.** View cash flow statements as reported by the company. | works — HTTP 200 |
| `/financial-statement-full-as-reported` | **As Reported Financial Statements.** Retrieve comprehensive financial statements as reported by companies with FMP As Reported Financial Statements API. | works — HTTP 200 |

## Charts

| Path | What it returns | Access |
|---|---|---|
| `/historical-price-eod/light` | **Basic Stock Chart.** Access simplified stock chart data using the FMP Basic Stock Chart API. | works — HTTP 200 |
| `/historical-price-eod/full` | **Comprehensive Stock Price and Volume Data.** Access full price and volume data for any stock symbol using the FMP Comprehensive Stock Price and Volume Data API. | works — HTTP 200 |
| `/historical-price-eod/non-split-adjusted` | **Unadjusted Stock Price Chart.** Access stock price and volume data without adjustments for stock splits with the FMP Unadjusted Stock Price Chart API. | works — HTTP 200 |
| `/historical-price-eod/dividend-adjusted` | **Dividend-Adjusted Price Chart.** Analyze stock performance with dividend adjustments using the FMP Dividend-Adjusted Price Chart API. | works — HTTP 200 |
| `/historical-chart/1min` | **1-Minute Interval Stock Chart.** Access precise intraday stock price and volume data with the FMP 1-Minute Interval Stock Chart API. | works — HTTP 200 |
| `/historical-chart/5min` | **5-Minute Interval Stock Chart.** Access stock price and volume data with the FMP 5-Minute Interval Stock Chart API. | works — HTTP 200 |
| `/historical-chart/15min` | **15-Minute Interval Stock Chart.** Access stock price and volume data with the FMP 15-Minute Interval Stock Chart API. | works — HTTP 200 |
| `/historical-chart/30min` | **30-Minute Interval Stock Chart.** Access stock price and volume data with the FMP 30-Minute Interval Stock Chart API. | works — HTTP 200 |
| `/historical-chart/1hour` | **1-Hour Interval Stock Chart.** Track stock price movements over hourly intervals with the FMP 1-Hour Interval Stock Chart API. | works — HTTP 200 |
| `/historical-chart/4hour` | **4-Hour Interval Stock Chart.** Analyze stock price movements over extended intraday periods with the FMP 4-Hour Interval Stock Chart API. | works — HTTP 200 |

## Economics

| Path | What it returns | Access |
|---|---|---|
| `/treasury-rates` | **Treasury Rates.** Access latest and historical Treasury rates for all maturities with the FMP Treasury Rates API. | works — HTTP 200 |
| `/economic-indicators` | **Economic Indicators.** Access real-time and historical economic data for key indicators like GDP, unemployment, and inflation with the FMP Economic Indicators API. | works — HTTP 200 |
| `/economic-calendar` | **Economic Data Releases Calendar.** Stay informed with the FMP Economic Data Releases Calendar API. | works — HTTP 200 |
| `/market-risk-premium` | **Market Risk Premium.** Access the market risk premium for specific dates with the FMP Market Risk Premium API. | works — HTTP 200 |

## Earnings, Dividends, Splits

| Path | What it returns | Access |
|---|---|---|
| `/dividends` | **Dividends Company.** Stay informed about upcoming dividend payments with the FMP Dividends Company API. | works — HTTP 200 |
| `/dividends-calendar` | **Dividend Events Calendar.** Stay informed on upcoming dividend events with the Dividend Events Calendar API. | works — HTTP 200 |
| `/earnings` | **Earnings Report.** Retrieve in-depth earnings information with the FMP Earnings Report API. | works — HTTP 200 |
| `/earnings-calendar` | **Earnings Calendar.** Stay informed on upcoming and past earnings announcements with the FMP Earnings Calendar API. | works — HTTP 200 |
| `/ipos-calendar` | **IPO Calendar.** Access a comprehensive list of all upcoming initial public offerings (IPOs) with the FMP IPO Calendar API. | works — HTTP 200 |
| `/ipos-disclosure` | **IPO Disclosures.** Access a comprehensive list of disclosure filings for upcoming initial public offerings (IPOs) with the FMP IPO Disclosures API. | works — HTTP 200 |
| `/ipos-prospectus` | **IPO Prospectus.** Access comprehensive information on IPO prospectuses with the FMP IPO Prospectus API. | works — HTTP 200 |
| `/splits` | **Stock Split Details.** Access detailed information on stock splits for a specific company using the FMP Stock Split Details API. | works — HTTP 200 |
| `/splits-calendar` | **Stock Splits Calendar.** Stay informed about upcoming stock splits with the FMP Stock Splits Calendar API. | works — HTTP 200 |

## Earnings Transcript

| Path | What it returns | Access |
|---|---|---|
| `/earning-call-transcript-latest` | **Latest Earning Transcripts.** Access available earnings transcripts for companies with the FMP Latest Earning Transcripts API. | blocked — HTTP 402 |
| `/earning-call-transcript` | **Earnings Transcript.** Access the full transcript of a company’s earnings call with the FMP Earnings Transcript API. | blocked — HTTP 402 |
| `/earning-call-transcript-dates` | **Transcripts Dates By Symbol.** Access earnings call transcript dates for specific companies with the FMP Transcripts Dates By Symbol API. | blocked — HTTP 402 |

## News

| Path | What it returns | Access |
|---|---|---|
| `/fmp-articles` | **Articles.** Access the latest articles from Financial Modeling Prep with the FMP Articles API. | works — HTTP 200 |
| `/news/general-latest` | **General News.** Access the latest general news articles from a variety of sources with the FMP General News API. | works — HTTP 200 |
| `/news/press-releases-latest` | **Press Releases.** Access official company press releases with the FMP Press Releases API. | works — HTTP 200 |
| `/news/stock-latest` | **Stock News Feed.** Stay informed with the latest stock market news using the FMP Stock News Feed API. | works — HTTP 200 |
| `/news/crypto-latest` | **Crypto News.** Stay informed with the latest cryptocurrency news using the FMP Crypto News API. | works — HTTP 200 |
| `/news/forex-latest` | **Forex News.** Stay updated with the latest forex news articles from various sources using the FMP Forex News API. | works — HTTP 200 |
| `/news/press-releases` | **Search Press Releases.** Search for company press releases with the FMP Search Press Releases API. | works — HTTP 200 |
| `/news/stock` | **Search Stock News.** Search for stock-related news using the FMP Search Stock News API. | works — HTTP 200 |
| `/news/crypto` | **Search Crypto News.** Search for cryptocurrency news using the FMP Search Crypto News API. | works — HTTP 200 |
| `/news/forex` | **Search Forex News.** Search for foreign exchange news using the FMP Search Forex News API. | works — HTTP 200 |

## Form 13F

| Path | What it returns | Access |
|---|---|---|
| `/institutional-ownership/latest` | **Institutional Ownership Filings.** Stay up to date with the most recent SEC filings related to institutional ownership using the Institutional Ownership Filings API. | blocked — HTTP 402 |
| `/institutional-ownership/extract` | **Institutional Ownership Filings.** The SEC Filings Extract API allows users to extract detailed data directly from official SEC filings. | blocked — HTTP 402 |
| `/institutional-ownership/dates` | **Form 13F Filings Dates.** The Form 13F Filings Dates API allows you to retrieve dates associated with Form 13F filings by institutional investors. | blocked — HTTP 402 |
| `/institutional-ownership/extract-analytics/holder` | **Filings Extract With Analytics By Holder.** The Filings Extract With Analytics By Holder API provides an analytical breakdown of institutional filings. | blocked — HTTP 402 |
| `/institutional-ownership/holder-performance-summary` | **Holder Performance Summary.** The Holder Performance Summary API provides insights into the performance of institutional investors based on their stock holdings. | blocked — HTTP 402 |
| `/institutional-ownership/holder-industry-breakdown` | **Holders Industry Breakdown.** Overview of the sectors and industries that institutional holders are investing in. | blocked — HTTP 402 |
| `/institutional-ownership/symbol-positions-summary` | **Positions Summary.** Snapshot of institutional holdings for a specific stock symbol. | blocked — HTTP 402 |
| `/institutional-ownership/industry-summary` | **Industry Performance Summary.** Overview of how various industries are performing financially. | blocked — HTTP 402 |

## Analyst

| Path | What it returns | Access |
|---|---|---|
| `/analyst-estimates` | **Financial Estimates.** Retrieve analyst financial estimates for stock symbols with the FMP Financial Estimates API. | works — HTTP 200 |
| `/ratings-snapshot` | **Ratings Snapshot.** Quickly assess the financial health and performance of companies with the FMP Ratings Snapshot API. | works — HTTP 200 |
| `/ratings-historical` | **Historical Ratings.** Track changes in financial performance over time with the FMP Historical Ratings API. | works — HTTP 200 |
| `/price-target-summary` | **Price Target Summary.** Gain insights into analysts' expectations for stock prices with the FMP Price Target Summary API. | works — HTTP 200 |
| `/price-target-consensus` | **Price Target Consensus.** Access analysts' consensus price targets with the FMP Price Target Consensus API. | works — HTTP 200 |
| `/grades` | **Grades.** Access the latest stock grades from top analysts and financial institutions with the FMP Grades API. | works — HTTP 200 |
| `/grades-historical` | **Historical Grades.** Access a comprehensive record of analyst grades with the FMP Historical Grades API. | works — HTTP 200 |
| `/grades-consensus` | **Grades Summary.** Quickly access an overall view of analyst ratings with the FMP Grades Summary API. | works — HTTP 200 |

## Market Performance

| Path | What it returns | Access |
|---|---|---|
| `/sector-performance-snapshot` | **Market Sector Performance Snapshot.** Get a snapshot of sector performance using the Market Sector Performance Snapshot API. | blocked — HTTP 402 |
| `/industry-performance-snapshot` | **Industry Performance Snapshot.** Access detailed performance data by industry using the Industry Performance Snapshot API. | blocked — HTTP 402 |
| `/historical-sector-performance` | **Historical Market Sector Performance.** Access historical sector performance data using the Historical Market Sector Performance API. | works — HTTP 200 |
| `/historical-industry-performance` | **Historical Industry Performance.** Access historical performance data for industries using the Historical Industry Performance API. | works — HTTP 200 |
| `/sector-pe-snapshot` | **Sector PE Snapshot.** Retrieve the price-to-earnings (P/E) ratios for various sectors using the Sector P/E Snapshot API. | blocked — HTTP 402 |
| `/industry-pe-snapshot` | **Industry P/E Snapshot.** View price-to-earnings (P/E) ratios for different industries using the Industry P/E Snapshot API. | blocked — HTTP 402 |
| `/historical-sector-pe` | **Historical Sector P/E.** Access historical price-to-earnings (P/E) ratios for various sectors using the Historical Sector P/E API. | works — HTTP 200 |
| `/historical-industry-pe` | **Historical Industry P/E.** Access historical price-to-earnings (P/E) ratios by industry using the Historical Industry P/E API. | works — HTTP 200 |
| `/biggest-gainers` | **Top Stock Gainers.** Stocks with the largest price increases. | works — HTTP 200 |
| `/biggest-losers` | **Biggest Stock Losers.** Stocks with the largest price drops. | works — HTTP 200 |
| `/most-actives` | **Top Traded Stocks.** Most actively traded stocks by volume. | works — HTTP 200 |

## Technical Indicators

| Path | What it returns | Access |
|---|---|---|
| `/technical-indicators/sma` | **Simple Moving Average.** | works — HTTP 200 |
| `/technical-indicators/ema` | **Exponential Moving Average.** | works — HTTP 200 |
| `/technical-indicators/wma` | **Weighted Moving Average.** | works — HTTP 200 |
| `/technical-indicators/dema` | **Double Exponential Moving Average.** | works — HTTP 200 |
| `/technical-indicators/tema` | **Triple Exponential Moving Average.** | works — HTTP 200 |
| `/technical-indicators/rsi` | **Relative Strength Index.** | works — HTTP 200 |
| `/technical-indicators/standarddeviation` | **Relative Strength Index.** Parameters | works — HTTP 200 |
| `/technical-indicators/williams` | **Relative Strength Index.** Parameters | works — HTTP 200 |
| `/technical-indicators/adx` | **Average Directional Index.** | works — HTTP 200 |

## Etf And Mutual Funds

| Path | What it returns | Access |
|---|---|---|
| `/etf/holdings` | **ETF & Fund Holdings.** Get a detailed breakdown of the assets held within ETFs and mutual funds using the FMP ETF & Fund Holdings API. | blocked — HTTP 402 |
| `/etf/info` | **ETF & Mutual Fund Information.** Access comprehensive data on ETFs and mutual funds with the FMP ETF & Mutual Fund Information API. | works — HTTP 200 |
| `/etf/country-weightings` | **ETF & Fund Country Allocation.** Gain insight into how ETFs and mutual funds distribute assets across different countries with the FMP ETF & Fund Country Allocation API. | works — HTTP 200 |
| `/etf/asset-exposure` | **ETF Asset Exposure.** Discover which ETFs hold specific stocks with the FMP ETF Asset Exposure API. | blocked — HTTP 402 |
| `/etf/sector-weightings` | **ETF Sector Weighting.** The FMP ETF Sector Weighting API provides a breakdown of the percentage of an ETF's assets that are invested in each sector. | works — HTTP 200 |
| `/funds/disclosure-holders-latest` | **Mutual Fund & ETF Disclosure.** Access the latest disclosures from mutual funds and ETFs with the FMP Mutual Fund & ETF Disclosure API. | blocked — HTTP 402 |
| `/funds/disclosure` | **Mutual Fund Disclosures.** Access comprehensive disclosure data for mutual funds with the FMP Mutual Fund Disclosures API. | blocked — HTTP 402 |
| `/funds/disclosure-holders-search` | **Mutual Fund & ETF Disclosure Name Search.** Easily search for mutual fund and ETF disclosures by name using the Mutual Fund & ETF Disclosure Name Search API. | blocked — HTTP 402 |
| `/funds/disclosure-dates` | **Fund & ETF Disclosures by Date.** Retrieve detailed disclosures for mutual funds and ETFs based on filing dates with the FMP Fund & ETF Disclosures by Date API. | blocked — HTTP 402 |

## Sec Filings

| Path | What it returns | Access |
|---|---|---|
| `/sec-filings-8k` | **Latest 8-K SEC Filings.** Stay up-to-date with the most recent 8-K filings from publicly traded companies using the FMP Latest 8-K SEC Filings API. | works — HTTP 200 |
| `/sec-filings-financials` | **Latest SEC Filings.** Stay updated with the most recent SEC filings from publicly traded companies using the FMP Latest SEC Filings API. | works — HTTP 200 |
| `/sec-filings-search/form-type` | **SEC Filings By Form Type.** Search for specific SEC filings by form type with the FMP SEC Filings By Form Type API. | works — HTTP 200 |
| `/sec-filings-search/symbol` | **SEC Filings By Symbol.** Search and retrieve SEC filings by company symbol using the FMP SEC Filings By Symbol API. | works — HTTP 200 |
| `/sec-filings-search/cik` | **SEC Filings By CIK.** Search for SEC filings using the FMP SEC Filings By CIK API. | works — HTTP 200 |
| `/sec-filings-company-search/name` | **SEC Filings By Name.** Search for SEC filings by company or entity name using the FMP SEC Filings By Name API. | works — HTTP 200 |
| `/sec-filings-company-search/symbol` | **SEC Filings Company Search By Symbol.** Find company information and regulatory filings using a stock symbol with the FMP SEC Filings Company Search By Symbol API. | works — HTTP 200 |
| `/sec-filings-company-search/cik` | **SEC Filings Company Search By CIK.** Easily find company information using a CIK (Central Index Key) with the FMP SEC Filings Company Search By CIK API. | works — HTTP 200 |
| `/sec-profile` | **SEC Company Full Profile.** Retrieve detailed company profiles, including business descriptions, executive details, contact information, and financial data with the FMP SEC Company Full Profile API. | works — HTTP 200 |
| `/standard-industrial-classification-list` | **Industry Classification List.** Retrieve a comprehensive list of industry classifications, including Standard Industrial Classification (SIC) codes and industry titles with the FMP Industry Classification List API. | works — HTTP 200 |
| `/industry-classification-search` | **Industry Classification Search.** Search and retrieve industry classification details for companies, including SIC codes, industry titles, and business information, with the FMP Industry Classification Search API. | works — HTTP 200 |
| `/all-industry-classification` | **All Industry Classification.** Access comprehensive industry classification data for companies across all sectors with the FMP All Industry Classification API. | works — HTTP 200 |

## Insider Trades

| Path | What it returns | Access |
|---|---|---|
| `/insider-trading/latest` | **Latest Insider Trading.** Access the latest insider trading activity. | works — HTTP 200 |
| `/insider-trading/search` | **Search Insider Trades.** Search insider trading activity by company or symbol using the Search Insider Trades API. | works — HTTP 200 |
| `/insider-trading/reporting-name` | **Search Insider Trades by Reporting Name.** Search for insider trading activity by reporting name using the Search Insider Trades by Reporting Name API. | works — HTTP 200 |
| `/insider-trading-transaction-type` | **All Insider Transaction Types.** Access a comprehensive list of insider transaction types with the All Insider Transaction Types API. | works — HTTP 200 |
| `/insider-trading/statistics` | **Insider Trade Statistics.** Analyze insider trading activity with the Insider Trade Statistics API. | works — HTTP 200 |
| `/acquisition-of-beneficial-ownership` | **Acquisition Ownership.** Track changes in stock ownership during acquisitions using the Acquisition Ownership API. | works — HTTP 200 |

## Indexes

Index symbols (e.g. `^GSPC`) also work on the shared `/quote`, `/historical-price-eod/*`, and `/historical-chart/*` paths listed under Quotes and Charts. `/historical-price-eod/dividend-adjusted` can 402 on index symbols; use `/full` instead.

| Path | What it returns | Access |
|---|---|---|
| `/index-list` | **Stock Market Indexes List.** Retrieve a comprehensive list of stock market indexes across global exchanges using the FMP Stock Market Indexes List API. | works — HTTP 200 |
| `/sp500-constituent` | **S&P 500 index using the S&P 500 Index.** Access detailed data on the S&P 500 index using the S&P 500 Index API. | works — HTTP 200 |
| `/nasdaq-constituent` | **Nasdaq index with the Nasdaq Index.** Access comprehensive data for the Nasdaq index with the Nasdaq Index API. | works — HTTP 200 |
| `/dowjones-constituent` | **Dow Jones Industrial Average using the Dow Jones.** Access data on the Dow Jones Industrial Average using the Dow Jones API. | works — HTTP 200 |
| `/historical-sp500-constituent` | **S&P 500 index using the Historical S&P 500.** Retrieve historical data for the S&P 500 index using the Historical S&P 500 API. | works — HTTP 200 |
| `/historical-nasdaq-constituent` | **Nasdaq index using the Historical Nasdaq.** Access historical data for the Nasdaq index using the Historical Nasdaq API. | works — HTTP 200 |
| `/historical-dowjones-constituent` | **Dow Jones Industrial Average using the Historical Dow Jones.** Access historical data for the Dow Jones Industrial Average using the Historical Dow Jones API. | works — HTTP 200 |

## Market Hours

| Path | What it returns | Access |
|---|---|---|
| `/exchange-market-hours` | **Global Exchange Market Hours.** Retrieve trading hours for specific stock exchanges using the Global Exchange Market Hours API. | works — HTTP 200 |
| `/holidays-by-exchange` | **Holidays By Exchange.** | works — HTTP 200 |
| `/all-exchange-market-hours` | **All Exchange Market Hours.** View the market hours for all exchanges. | works — HTTP 200 |

## Commodity

Commodity symbols (e.g. `GCUSD`) use the same quote, EOD, and intraday paths as equities, listed under Quotes and Charts. This section keeps the commodity-specific list endpoint.

| Path | What it returns | Access |
|---|---|---|
| `/commodities-list` | **Commodities List.** Access an extensive list of tracked commodities across various sectors, including energy, metals, and agricultural products. | works — HTTP 200 |

## Discounted Cash Flow

| Path | What it returns | Access |
|---|---|---|
| `/discounted-cash-flow` | **Discounted Cash Flow Valuation.** Estimate the intrinsic value of a company with the FMP Discounted Cash Flow Valuation API. | works — HTTP 200 |
| `/levered-discounted-cash-flow` | **Levered Discounted Cash Flow (DCF).** Analyze a company’s value with the FMP Levered Discounted Cash Flow (DCF) API, which incorporates the impact of debt. | works — HTTP 200 |
| `/custom-discounted-cash-flow` | **Custom DCF Advanced.** Run a tailored Discounted Cash Flow (DCF) analysis using the FMP Custom DCF Advanced API. | works — HTTP 200 |
| `/custom-levered-discounted-cash-flow` | **Custom DCF Advanced.** Run a tailored Discounted Cash Flow (DCF) analysis using the FMP Custom DCF Advanced API. | works — HTTP 200 |

## Forex

FX pairs (e.g. `EURUSD`) use the same quote, EOD, and intraday paths as equities. This section keeps the pair-list endpoint.

| Path | What it returns | Access |
|---|---|---|
| `/forex-list` | **Forex Currency Pairs.** Access a comprehensive list of all currency pairs traded on the forex market with the FMP Forex Currency Pairs API. | works — HTTP 200 |

## Crypto

Crypto pairs (e.g. `BTCUSD`) use the same quote, EOD, and intraday paths as equities. This section keeps the coin-list endpoint.

| Path | What it returns | Access |
|---|---|---|
| `/cryptocurrency-list` | **Cryptocurrencies Overview.** Access a comprehensive list of all cryptocurrencies traded on exchanges worldwide with the FMP Cryptocurrencies Overview API. | works — HTTP 200 |

## Senate

| Path | What it returns | Access |
|---|---|---|
| `/senate-latest` | **Latest Senate Financial Disclosures.** Access the latest financial disclosures from U.S. | works — HTTP 200 |
| `/house-latest` | **Latest House Financial Disclosures.** Access real-time financial disclosures from U.S. | works — HTTP 200 |
| `/senate-trades` | **Senate Trading Activity.** Monitor the trading activity of US Senators with the FMP Senate Trading Activity API. | works — HTTP 200 |
| `/senate-trades-by-name` | **Senate Trades By Name.** | works — HTTP 200 |
| `/house-trades` | **U.S. House Trades.** Track the financial trades made by U.S. House members and their families with the FMP U.S. House Trades API. Access real-time information on stock sales, purchases, and other investment activities to gain insight into their financial decisions. | works — HTTP 200 |
| `/house-trades-by-name` | **House Trades By Name.** | works — HTTP 200 |

## ESG

| Path | What it returns | Access |
|---|---|---|
| `/esg-disclosures` | **ESG Investment Search.** Align your investments with your values using the FMP ESG Investment Search API. | blocked — HTTP 402 |
| `/esg-ratings` | **ESG Ratings.** Access comprehensive ESG ratings for companies and funds with the FMP ESG Ratings API. | blocked — HTTP 402 |
| `/esg-benchmark` | **ESG Benchmark Comparison.** Evaluate the ESG performance of companies and funds with the FMP ESG Benchmark Comparison API. | blocked — HTTP 402 |

## Commitment Of Traders

| Path | What it returns | Access |
|---|---|---|
| `/commitment-of-traders-report` | **COT Report.** Access comprehensive Commitment of Traders (COT) reports with the FMP COT Report API. | works — HTTP 200 |
| `/commitment-of-traders-analysis` | **COT Report Analysis.** Gain in-depth insights into market sentiment with the FMP COT Report Analysis API. | works — HTTP 200 |
| `/commitment-of-traders-list` | **COT Report List.** Access a comprehensive list of available Commitment of Traders (COT) reports by commodity or futures contract using the FMP COT Report List API. | works — HTTP 200 |

## Fundraisers

| Path | What it returns | Access |
|---|---|---|
| `/crowdfunding-offerings-latest` | **Latest Crowdfunding Campaigns.** Discover the most recent crowdfunding campaigns with the FMP Latest Crowdfunding Campaigns API. | works — HTTP 200 |
| `/crowdfunding-offerings-search` | **Crowdfunding Campaign Search.** Search for crowdfunding campaigns by company name, campaign name, or platform with the FMP Crowdfunding Campaign Search API. | works — HTTP 200 |
| `/crowdfunding-offerings` | **Crowdfunding By CIK.** Access detailed information on all crowdfunding campaigns launched by a specific company with the FMP Crowdfunding By CIK API. | works — HTTP 200 |
| `/fundraising-latest` | **Equity Offering Updates.** Stay informed about the latest equity offerings with the FMP Equity Offering Updates API. | works — HTTP 200 |
| `/fundraising-search` | **Equity Offering Search.** Easily search for equity offerings by company name or stock symbol with the FMP Equity Offering Search API. | works — HTTP 200 |
| `/fundraising` | **Company Equity Offerings by CIK.** Access detailed information on equity offerings announced by specific companies with the FMP Company Equity Offerings by CIK API. | works — HTTP 200 |

## Bulk

| Path | What it returns | Access |
|---|---|---|
| `/profile-bulk` | **Profile Bulk.** The FMP Profile Bulk API allows users to retrieve comprehensive company profile data in bulk. | blocked — HTTP 402 |
| `/rating-bulk` | **Rating Bulk.** The FMP Rating Bulk API provides users with comprehensive rating data for multiple stocks in a single request. | blocked — HTTP 402 |
| `/dcf-bulk` | **DCF Bulk.** The FMP DCF Bulk API enables users to quickly retrieve discounted cash flow (DCF) valuations for multiple symbols in one request. | blocked — HTTP 402 |
| `/scores-bulk` | **Scores Bulk.** The FMP Scores Bulk API allows users to quickly retrieve a wide range of key financial scores and metrics for multiple symbols. | blocked — HTTP 402 |
| `/price-target-summary-bulk` | **Price Target Summary Bulk.** The Price Target Summary Bulk API provides a comprehensive overview of price targets for all listed symbols over multiple timeframes. | blocked — HTTP 402 |
| `/etf-holder-bulk` | **ETF Holder Bulk.** Holdings (assets and shares) for ETFs, whole-universe dump. | blocked — HTTP 402 |
| `/upgrades-downgrades-consensus-bulk` | **Upgrades Downgrades Consensus Bulk.** The Upgrades Downgrades Consensus Bulk API provides a comprehensive view of analyst ratings across all symbols. | blocked — HTTP 402 |
| `/key-metrics-ttm-bulk` | **Key Metrics TTM Bulk.** Trailing-twelve-month key metrics for all companies. | blocked — HTTP 402 |
| `/ratios-ttm-bulk` | **Ratios TTM Bulk.** The Ratios TTM Bulk API offers an efficient way to retrieve trailing twelve months (TTM) financial ratios for stocks. | blocked — HTTP 402 |
| `/peers-bulk` | **Stock Peers Bulk.** The Stock Peers Bulk API allows you to quickly retrieve a comprehensive list of peer companies for all stocks in the database. | blocked — HTTP 402 |
| `/earnings-surprises-bulk` | **Earnings Surprises Bulk.** The Earnings Surprises Bulk API allows users to retrieve bulk data on annual earnings surprises, enabling quick analysis of which companies have beaten, missed, or met their earnings estimates. | blocked — HTTP 402 |
| `/income-statement-bulk` | **Income Statement Bulk.** The Bulk Income Statement API allows users to retrieve detailed income statement data in bulk. | blocked — HTTP 402 |
| `/income-statement-growth-bulk` | **Income Statement Growth Bulk.** The Bulk Income Statement Growth API provides access to growth data for income statements across multiple companies. | blocked — HTTP 402 |
| `/balance-sheet-statement-bulk` | **Balance Sheet Statement Bulk.** The Bulk Balance Sheet Statement API provides comprehensive access to balance sheet data across multiple companies. | blocked — HTTP 402 |
| `/balance-sheet-statement-growth-bulk` | **Balance Sheet Statement Growth Bulk.** The Balance Sheet Growth Bulk API allows users to retrieve growth data across multiple companies’ balance sheets, enabling detailed analysis of how financial positions have changed over time. | blocked — HTTP 402 |
| `/cash-flow-statement-bulk` | **Cash Flow Statement Bulk.** The Cash Flow Statement Bulk API provides access to detailed cash flow reports for a wide range of companies. | blocked — HTTP 402 |
| `/cash-flow-statement-growth-bulk` | **Cash Flow Statement Growth Bulk.** The Cash Flow Statement Growth Bulk API allows you to retrieve bulk growth data for cash flow statements, enabling you to track changes in cash flows over time. | blocked — HTTP 402 |
| `/eod-bulk` | **Cash Flow Statement Growth Bulk.** The EOD Bulk API allows users to retrieve end-of-day stock price data for multiple symbols in bulk. | blocked — HTTP 402 |

## How to re-check entitlements

```bash
/opt/anaconda3/envs/quant/bin/python scripts/ingest/probe_fmp_entitlements.py
/opt/anaconda3/envs/quant/bin/python scripts/ingest/probe_fmp_entitlements.py --restricted-only
```

Paste a changed restricted list into `docs/data/DATA_INVENTORY.md` §6. After adding a new
per-symbol dataset, register it in `core/data/vendors/fmp/datasets.py` rather than writing a
one-off fetcher.
