"""Named strategy registry: metadata for catalogs and documentation."""

from __future__ import annotations

from core.strategies.types import StrategyKind, StrategyMetadata

STRATEGIES: dict[str, StrategyMetadata] = {
    "factor_cross_section": StrategyMetadata(
        id="factor_cross_section",
        title="Factor long/short (cross-sectional)",
        description=(
            "Rank universe on a factor column each period; long top tier, "
            "short bottom tier (optional long-only). Uses in-memory factor and price panels."
        ),
        kind=StrategyKind.FACTOR_CROSS_SECTION,
        post_path="/run-backtest",
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
