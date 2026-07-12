"""Named strategy registry: metadata for catalogs and documentation.

Each entry answers "what are we betting on, why should it work, what published
evidence supports it, and what are the known failure modes" — so the UI and
docs can present the strategy honestly (with caveats) rather than as a black
box.  See ``docs/FACTOR_BACKTEST_AUDIT.md`` §4 for the motivation.
"""

from __future__ import annotations

from core.strategies.types import StrategyKind, StrategyMetadata

STRATEGIES: dict[str, StrategyMetadata] = {
    "factor_cross_section": StrategyMetadata(
        id="factor_cross_section",
        title="Factor long/short (cross-sectional)",
        description=(
            "Rank universe on a factor column each period; long top tier, "
            "short bottom tier (optional long-only). Uses in-memory factor and "
            "price panels."
        ),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
        hypothesis=(
            "The cross-section of expected stock returns is predictable from "
            "observable factor loadings. Ranking on a factor and buying winners / "
            "selling losers should earn a positive spread on average, net of "
            "transaction costs, if the factor is economically meaningful and not "
            "fully arbitraged away."
        ),
        reference=(
            "Fama & French (1993) 'Common risk factors in the returns on stocks "
            "and bonds', JFE 33(1); see also Harvey, Liu & Zhu (2016) '...and the "
            "Cross-Section of Expected Returns' for multiple-testing warnings."
        ),
        expected_sharpe_range=(0.3, 0.8),
        known_limitations=(
            "Long-short returns are simulated gross of borrow cost and short "
            "availability; real shorts can be expensive or impossible for small caps.",
            "Default execution is T+1 close (signal_lag_days=1); the legacy 0-lag "
            "MOC execution is unrealistic and retained only for reproducing old runs.",
            "Delisted names are realised at -100% on the long leg (bankruptcy "
            "convention). Real deals often include a merger payout or partial recovery.",
            "Sharpe ratios over short sub-periods are noisy; prefer a 10+ year window "
            "and check sub-period stability before drawing conclusions.",
            "Factor crowding: once a factor becomes well-known and heavily traded, "
            "its published premium typically shrinks (McLean & Pontiff 2016).",
        ),
    ),
    "momentum_12_1": StrategyMetadata(
        id="momentum_12_1",
        title="12-1 price momentum",
        description=(
            "Jegadeesh-Titman (m-1) momentum: long stocks with the highest "
            "geometric return over the past 12 months excluding the most recent "
            "month; short the lowest. Standard monthly rebalance."
        ),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
        hypothesis=(
            "Under-reaction to news (behavioral) and slow information diffusion "
            "cause past winners to continue winning and past losers to continue "
            "losing over horizons of 3-12 months. The 1-month skip removes the "
            "short-term reversal effect driven by microstructure and liquidity."
        ),
        reference=(
            "Jegadeesh & Titman (1993) 'Returns to Buying Winners and Selling "
            "Losers: Implications for Stock Market Efficiency', Journal of Finance "
            "48(1); Carhart (1997) for the factor formalization."
        ),
        expected_sharpe_range=(0.3, 0.7),
        known_limitations=(
            "Severe drawdowns in recovery months after major bear markets "
            "('momentum crashes', Daniel & Moskowitz 2016).",
            "Highly turnover-intensive; sensitive to transaction costs.",
            "Empirically weak in post-2010 US large-cap data; better in small/mid "
            "caps and international markets.",
        ),
    ),
    "low_volatility": StrategyMetadata(
        id="low_volatility",
        title="Low volatility (beta anomaly)",
        description=(
            "Long lowest-volatility names, short highest. Equivalent to the "
            "'betting-against-beta' / 'betting-against-vol' trade."
        ),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
        hypothesis=(
            "Leverage-constrained investors overbid high-beta stocks, causing "
            "low-beta names to deliver higher risk-adjusted returns. The CAPM "
            "security market line is flatter than theory predicts."
        ),
        reference=(
            "Frazzini & Pedersen (2014) 'Betting Against Beta', JFE 111(1); "
            "Ang, Hodrick, Xing & Zhang (2006) for the idiosyncratic-vol variant."
        ),
        expected_sharpe_range=(0.3, 0.7),
        known_limitations=(
            "Concentrates in defensive sectors (utilities, staples); sector-neutral "
            "construction is usually required before live deployment.",
            "Crowded trade since ~2012; factor ETF flows may have compressed the premium.",
            "Returns are positively correlated with bond yields falling — may "
            "underperform in rate-rising regimes.",
        ),
    ),
    "size_small_minus_big": StrategyMetadata(
        id="size_small_minus_big",
        title="Size (small-minus-big)",
        description=("Long small-cap, short large-cap. Ranked on log market capitalization."),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
        hypothesis=(
            "Small caps earn a premium as compensation for illiquidity and higher "
            "distress risk. Originally documented by Banz (1981) and formalized as "
            "SMB in the Fama-French 3-factor model."
        ),
        reference=(
            "Fama & French (1993); Banz (1981) 'The relationship between return and "
            "market value of common stocks', JFE 9(1)."
        ),
        expected_sharpe_range=(0.1, 0.4),
        known_limitations=(
            "Premium has largely disappeared in the US since the 1980s; much of the "
            "SMB return is concentrated in January and micro-caps.",
            "Small caps incur significantly higher transaction costs and have "
            "sparser factor panels — min_stocks filter is critical.",
            "The premium is stronger when combined with other factors (quality, "
            "profitability) than in isolation.",
        ),
    ),
    "short_term_reversal": StrategyMetadata(
        id="short_term_reversal",
        title="Short-term reversal (1-month)",
        description=(
            "Long the biggest losers of the past 21 trading days, short the "
            "biggest winners. Ranked on the negative trailing 1-month return "
            "(rev_21d) so the top tier is the losers."
        ),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
        hypothesis=(
            "Over 1-week to 1-month horizons stock returns partially reverse: "
            "liquidity providers demand compensation for absorbing order-flow "
            "imbalances, and prices overshoot on non-fundamental demand. Buying "
            "recent losers and selling recent winners harvests that liquidity "
            "premium."
        ),
        reference=(
            "Jegadeesh (1990) 'Evidence of Predictable Behavior of Security "
            "Returns', Journal of Finance 45(3); Lehmann (1990) 'Fads, "
            "Martingales, and Market Efficiency', QJE 105(1); Nagel (2012) "
            "'Evaporating Liquidity', RFS 25(7)."
        ),
        expected_sharpe_range=(0.2, 0.9),
        known_limitations=(
            "Extremely turnover-intensive (full position refresh every month or "
            "faster); gross premium is largely consumed by transaction costs at "
            "realistic bps — always compare gross vs net.",
            "The premium decayed sharply after the 1990s as electronic market "
            "making compressed liquidity-provision profits (Nagel 2012).",
            "Loser stocks are often losers for fundamental reasons (earnings "
            "misses, litigation); without news filtering the strategy buys some "
            "genuinely deteriorating names.",
            "Best results historically come from weekly rebalancing, which this "
            "platform approximates with monthly rebalance — expect weaker "
            "performance than published weekly numbers.",
        ),
    ),
    "earnings_yield": StrategyMetadata(
        id="earnings_yield",
        title="Earnings yield (E/P value)",
        description=(
            "Long highest trailing-twelve-month earnings yield (net income / "
            "market cap), short lowest. Point-in-time: earnings become visible "
            "only after the SEC filing acceptance date."
        ),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
        hypothesis=(
            "Cheap stocks (high E/P) earn a premium as compensation for distress "
            "risk and as a correction of investor over-extrapolation of growth. "
            "Earnings yield is the income-statement twin of book-to-market."
        ),
        reference=(
            "Basu (1977) 'Investment Performance of Common Stocks in Relation to "
            "Their Price-Earnings Ratios', Journal of Finance 32(3); Fama & French "
            "(1992) for the value premium in the cross-section."
        ),
        expected_sharpe_range=(-0.2, 0.5),
        known_limitations=(
            "US large-cap value premium has been weak to flat since ~2007; expect "
            "near-zero Sharpe in S&P 500 samples after 2005.",
            "Negative earners sort into the short leg; that is intentional but "
            "concentrates the short book in loss-making names.",
            "Pre-2015 S&P coverage is incomplete for delisted members (FMP gap); "
            "prefer 2015+ windows or treat earlier results as biased toward survivors.",
        ),
    ),
    "book_to_market": StrategyMetadata(
        id="book_to_market",
        title="Book-to-market (HML value)",
        description=(
            "Long highest book-equity / market-cap, short lowest. Classic "
            "Fama-French HML construction on a point-in-time fundamentals panel."
        ),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
        hypothesis=(
            "High book-to-market stocks are underpriced relative to accounting "
            "value, either as compensation for financial distress risk (rational) "
            "or because investors overpay for growth glamour (behavioral)."
        ),
        reference=(
            "Fama & French (1993) 'Common risk factors in the returns on stocks "
            "and bonds', JFE 33(1); Rosenberg, Reid & Lanstein (1985) for the "
            "original book-to-market evidence."
        ),
        expected_sharpe_range=(-0.2, 0.5),
        known_limitations=(
            "Same post-2007 value decay as earnings yield in US large caps.",
            "Negative book equity is excluded (NaN), which drops some financials "
            "and distressed names from the ranking.",
            "Intangible-heavy businesses make book equity a poorer measure of "
            "economic capital than it was in the 1960-1990 sample.",
            "Pre-2015 S&P coverage gap applies (see earnings_yield limitations).",
        ),
    ),
    "roe_quality": StrategyMetadata(
        id="roe_quality",
        title="ROE quality (profitability)",
        description=(
            "Long highest trailing-twelve-month ROE (net income / book equity), "
            "short lowest. Closest single-ratio proxy for Fama-French RMW."
        ),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
        hypothesis=(
            "Highly profitable firms earn a premium because the market under-reacts "
            "to the persistence of profitability, and because low-ROE firms carry "
            "uncompensated distress risk that is not fully priced."
        ),
        reference=(
            "Novy-Marx (2013) 'The other side of value: The gross profitability "
            "premium', JFE 108(1); Fama & French (2015) for RMW in the 5-factor model."
        ),
        expected_sharpe_range=(0.0, 0.6),
        known_limitations=(
            "ROE uses book equity in the denominator — same negative-book exclusion "
            "as book_to_market, and sensitive to buybacks that shrink the book.",
            "Gross profitability (Novy-Marx) is usually preferred to ROE; ROE can "
            "be mechanically high for levered firms.",
            "Quality and value often hedge each other; a combined portfolio is "
            "more informative than either leg alone.",
        ),
    ),
    "low_asset_growth": StrategyMetadata(
        id="low_asset_growth",
        title="Low asset growth (investment)",
        description=(
            "Long lowest year-over-year total-asset growth, short highest. Ranked "
            "on ``neg_asset_growth`` so the shared descending ranker longs the "
            "low-investment side (Fama-French CMA)."
        ),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
        hypothesis=(
            "Firms that grow assets aggressively earn lower subsequent returns — "
            "empire-building, overinvestment, and the market's under-reaction to "
            "the diminishing returns of expansion. Low-investment firms are the "
            "premium side."
        ),
        reference=(
            "Cooper, Gulen & Schill (2008) 'Asset Growth and the Cross-Section of "
            "Stock Returns', Journal of Finance 63(4); Fama & French (2015) CMA."
        ),
        expected_sharpe_range=(-0.1, 0.5),
        known_limitations=(
            "M&A and large capex programs dominate the high-growth short leg; "
            "those events are public and may be partially priced already.",
            "Asset growth is noisy for financials (balance-sheet composition "
            "differs); consider excluding banks in a follow-up.",
            "Post-2005 US large-cap realization has been weak, similar to value.",
        ),
    ),
    "ml_commodity_direction": StrategyMetadata(
        id="ml_commodity_direction",
        title="ML commodity direction (walk-forward)",
        description=(
            "Walk-forward validation for directional prediction on a single symbol "
            "using engineered features (XGBoost, Random Forest, Logistic, LSTM)."
        ),
        kind=StrategyKind.ML_DIRECTION,
        post_path="/run-ml-strategy",
        hypothesis=(
            "Short-horizon directional predictability exists in commodity futures "
            "due to slow-moving fundamentals (inventories, seasonality) and "
            "systematic positioning biases. An ML model trained on engineered "
            "technical and fundamental features can exploit these patterns."
        ),
        reference=(
            "Moskowitz, Ooi & Pedersen (2012) 'Time series momentum', JFE 104(2); "
            "Gu, Kelly & Xiu (2020) 'Empirical Asset Pricing via Machine Learning', RFS."
        ),
        expected_sharpe_range=(0.2, 0.8),
        known_limitations=(
            "Walk-forward accuracy rarely translates 1:1 into trading Sharpe once "
            "execution costs, slippage, and position sizing are included.",
            "Feature importance is unstable across folds; do not interpret a single "
            "fold's ranking as causal.",
            "Overfitting risk is high with few splits; prefer max_splits >= 20.",
        ),
    ),
}


def list_strategies() -> list[StrategyMetadata]:
    """Return all registered strategies in stable order (by id)."""
    return [STRATEGIES[k] for k in sorted(STRATEGIES.keys())]


def get_strategy(strategy_id: str) -> StrategyMetadata:
    """
    Look up metadata by strategy id.

    Raises:
        KeyError: If ``strategy_id`` is not registered.
    """
    return STRATEGIES[strategy_id]
