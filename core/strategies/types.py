"""Types for the named strategy registry (metadata only; logic lives elsewhere)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StrategyKind(str, Enum):
    """High-level family of strategy implementation."""

    FACTOR_CROSS_SECTION = "factor_cross_section"
    ML_DIRECTION = "ml_direction"
    PAIRS_COINTEGRATION = "pairs_cointegration"
    PAIRS_INDEX = "pairs_index"


@dataclass(frozen=True)
class StrategyMetadata:
    """
    Public description of a registered strategy for API catalogs and UIs.

    Attributes:
        id: Stable slug (e.g. factor_cross_section).
        title: Short human-readable name.
        description: One or two sentences on behavior and data requirements.
        kind: Strategy family.
        post_path: If set, HTTP POST path for running this strategy (FastAPI route path).
        hypothesis: Academic or economic claim the strategy is predicated on
            ("what we are betting on, and why it should work"). Written in plain
            English so non-quants can read and pushback.
        reference: Canonical published reference (author, year, journal/URL).
            None if the strategy is proprietary or heuristic.
        expected_sharpe_range: Order-of-magnitude Sharpe expectations from the
            published literature or internal live experience, as (lo, hi).
            Intentionally wide — do NOT treat as a target.
        known_limitations: Bullet-point caveats specific to this strategy
            (execution, crowding, regime dependence, data quality, etc.).
            Shown in the UI so users can make informed decisions.
    """

    id: str
    title: str
    description: str
    kind: StrategyKind
    post_path: Optional[str] = None
    hypothesis: Optional[str] = None
    reference: Optional[str] = None
    expected_sharpe_range: Optional[tuple[float, float]] = None
    known_limitations: tuple[str, ...] = field(default_factory=tuple)
