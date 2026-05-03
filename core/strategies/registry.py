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
        description=(
            "Long small-cap, short large-cap. Ranked on log market capitalization."
        ),
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
