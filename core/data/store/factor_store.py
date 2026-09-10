"""Lazy, per-column access to the factor panels.

The API used to hold every factor panel fully in memory. After the ADR-0013
cutover that costs ~7 GB (measured), because the fundamentals factor panel alone
is ~21M rows x 37 columns.

It is also unnecessary. Every consumer needs one of two things:

1. **The list of available factor names** — schema only, free to read.
2. **One factor column** for a backtest — a single column is ~85 MB at full
   universe, and a small LRU cache covers repeated requests for the same factor.

So this store reads parquet metadata at startup and column data on demand.
Nothing else changes: :meth:`load_factor` returns a ``(date, symbol)`` frame with
one column, exactly the shape the cross-section runner already expects.

Column names are unique across panels by construction; when two panels do share a
name, the first panel in :data:`PANEL_PRIORITY` wins and the collision is logged
rather than silently resolved.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Searched in order; earlier files win a name collision.
PANEL_PRIORITY: tuple[str, ...] = (
    "factors_all.parquet",
    "factors_price.parquet",
    "factors_fundamental.parquet",
    "factors_composites.parquet",
    "factors_microstructure.parquet",
    "factors_earnings_surprise.parquet",
    "factors_vendor_metrics.parquet",
    "factors_technical.parquet",
)

_INDEX_LEVELS = ("date", "symbol")

# factors_all is factors_price plus log_market_cap, so every price factor appears
# in both by construction. That overlap is expected and resolved by priority; only
# unexpected collisions deserve a warning.
_EXPECTED_OVERLAPS: frozenset[tuple[str, str]] = frozenset(
    {("factors_all.parquet", "factors_price.parquet")}
)
_CACHE_SIZE = 8


@lru_cache(maxsize=_CACHE_SIZE)
def _read_factor_column(path_str: str, column: str) -> pd.DataFrame:
    """
    Cached single-column parquet read.

    Module-level rather than a cached method on purpose: an ``lru_cache`` on a
    method keys on ``self`` and keeps every store instance alive for the life of
    the cache, which is a leak in a long-running API process.

    Args:
        path_str: Panel path as a string (hashable cache key).
        column: Factor column to read; index levels come along automatically.

    Returns:
        One-column ``(date, symbol)`` DataFrame.
    """
    frame = pd.read_parquet(Path(path_str), columns=[column])
    logger.info("Loaded factor %r from %s: %s rows", column, Path(path_str).name, f"{len(frame):,}")
    return frame


class FactorStore:
    """
    Read-on-demand access to factor columns spread across several panel files.

    Attributes:
        factors_dir: Directory holding the panel parquet files.
        column_to_panel: Resolved map of factor name -> panel path.
    """

    def __init__(
        self,
        factors_dir: Path,
        symbols: Optional[set[str]] = None,
        panel_priority: Optional[tuple[str, ...]] = None,
    ) -> None:
        """
        Args:
            factors_dir: Directory containing the factor panels.
            symbols: Optional universe restriction applied to every loaded column,
                so the store never returns symbols the API did not load prices for.
            panel_priority: Optional override of :data:`PANEL_PRIORITY` — the panels
                to scan, highest priority first. Use it to scope a store to specific
                panels, e.g. so a research script never touches a panel that a
                concurrent rebuild is rewriting.
        """
        self.factors_dir = Path(factors_dir)
        self.symbols = symbols
        self.panel_priority = PANEL_PRIORITY if panel_priority is None else tuple(panel_priority)
        self.column_to_panel: dict[str, Path] = {}
        self._index_cache: dict[Path, pd.MultiIndex] = {}
        self._scan()

    def _scan(self) -> None:
        """Map every factor column to the panel that owns it, using metadata only."""
        for name in self.panel_priority:
            path = self.factors_dir / name
            if not path.exists():
                continue
            try:
                schema = pq.ParquetFile(path).schema_arrow
            except Exception:
                logger.exception("Unreadable factor panel %s; skipping", path.name)
                continue
            for column in schema.names:
                if column in _INDEX_LEVELS or column.startswith("__"):
                    continue
                if column in self.column_to_panel:
                    owner = self.column_to_panel[column].name
                    expected = (owner, path.name) in _EXPECTED_OVERLAPS
                    logger.log(
                        logging.DEBUG if expected else logging.WARNING,
                        "Factor %r appears in both %s and %s; keeping %s",
                        column,
                        owner,
                        path.name,
                        owner,
                    )
                    continue
                self.column_to_panel[column] = path

        logger.info(
            "FactorStore: %d factors across %d panels",
            len(self.column_to_panel),
            len({p.name for p in self.column_to_panel.values()}),
        )

    @property
    def available_factors(self) -> list[str]:
        """Sorted factor names, without reading any column data."""
        return sorted(self.column_to_panel)

    def panel_of(self, factor: str) -> Optional[str]:
        """Name of the panel file a factor lives in, or None if unknown."""
        path = self.column_to_panel.get(factor)
        return path.name if path else None

    def load_factor(self, factor: str) -> pd.DataFrame:
        """
        Load one factor as a ``(date, symbol)`` single-column DataFrame.

        Args:
            factor: Factor column name.

        Returns:
            One-column DataFrame, restricted to the store's universe when set.

        Raises:
            KeyError: If the factor is not present in any panel.
        """
        if factor not in self.column_to_panel:
            raise KeyError(f"Unknown factor {factor!r}; {len(self.column_to_panel)} available")
        frame = self._read_column(self.column_to_panel[factor], factor)
        if self.symbols is not None:
            level = frame.index.get_level_values("symbol")
            frame = frame[level.isin(self.symbols)]
        return frame

    def _read_column(self, path: Path, column: str) -> pd.DataFrame:
        return _read_factor_column(str(path), column)

    def load_factors(self, factors: list[str]) -> pd.DataFrame:
        """
        Load several factors and align them on ``(date, symbol)``.

        Columns may live in different panels with different row coverage, so the
        result is an outer join: a row present in one panel but not another gets
        NaN rather than being dropped, and the caller decides how to handle it.

        Args:
            factors: Factor names.

        Returns:
            DataFrame with one column per requested factor.

        Raises:
            KeyError: If any factor is unknown.
        """
        unknown = [f for f in factors if f not in self.column_to_panel]
        if unknown:
            raise KeyError(f"Unknown factors: {unknown}")

        frames = [self.load_factor(factor) for factor in factors]
        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.join(frame, how="outer")
        return combined[factors]

    @staticmethod
    def cache_clear() -> None:
        """Drop cached columns (used by tests and after a panel rebuild)."""
        _read_factor_column.cache_clear()


def describe_factor_source(store: FactorStore) -> list[dict[str, str]]:
    """
    Factor -> owning panel, for the ``GET /data/factors`` disclosure.

    Knowing which panel a factor came from tells a reader what universe and
    build date it inherits, which is not otherwise visible from the name.
    """
    return [
        {"factor": factor, "panel": store.panel_of(factor) or "unknown"}
        for factor in store.available_factors
    ]
