"""Types for the named strategy registry (metadata only; logic lives elsewhere)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StrategyKind(str, Enum):
    """High-level family of strategy implementation."""

    FACTOR_CROSS_SECTION = "factor_cross_section"
    ML_DIRECTION = "ml_direction"


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
    """

    id: str
    title: str
    description: str
    kind: StrategyKind
    post_path: Optional[str] = None
