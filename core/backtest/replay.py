"""
Frame precomputation for strategy replay.

Generates a sequence of ReplayFrame objects that capture the state of a
strategy at each point in time: signal, position, PnL, cumulative PnL,
and rolling metrics.  Frames can be served from memory or serialised to
Parquet / JSON for caching.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.metrics.coverage import position_label_from_counts
from core.metrics.performance import (
    calculate_cumulative_returns,
    calculate_drawdown,
    calculate_sortino_ratio,
)

logger = logging.getLogger(__name__)


def precompute_backtest_frames(
    net_returns: pd.Series,
    signals: Optional[pd.Series] = None,
    rolling_window: int = 60,
    n_long: Optional[pd.Series] = None,
    n_short: Optional[pd.Series] = None,
) -> List[Dict[str, Any]]:
    """
    Build frame-by-frame replay data from a backtest result.

    Each frame captures:
    - date
    - daily PnL (net return)
    - cumulative PnL: cumulative **return** since start (wealth index minus 1)
    - drawdown
    - rolling Sortino (if enough history; downside deviation, annualised)
    - signal value (if provided) or net headcount ``n_long - n_short``
    - position label (long / short / long_short / flat)
    - ``n_long`` / ``n_short`` when provided

    Prefer ``n_long`` / ``n_short`` for cross-sectional portfolios: a scalar
    ``signals`` series alone cannot represent multi-name books, and omitting
    both leaves every frame labelled ``flat``.

    Args:
        net_returns: Series of daily net returns (date-indexed)
        signals: Optional Series of signal values aligned to same dates
        rolling_window: Window for rolling Sortino (default 60 trading days)
        n_long: Optional daily long headcount from the portfolio simulator
        n_short: Optional daily short headcount

    Returns:
        List of frame dictionaries ready for JSON serialisation
    """
    cum = calculate_cumulative_returns(net_returns)
    dd = calculate_drawdown(net_returns)

    rolling_sortino = pd.Series(np.nan, index=net_returns.index)
    if len(net_returns) > rolling_window:
        for i in range(rolling_window, len(net_returns)):
            window = net_returns.iloc[i - rolling_window : i]
            rolling_sortino.iloc[i] = calculate_sortino_ratio(window)

    long_aligned = n_long.reindex(net_returns.index) if n_long is not None else None
    short_aligned = n_short.reindex(net_returns.index) if n_short is not None else None

    frames: List[Dict[str, Any]] = []
    for i, date in enumerate(net_returns.index):
        signal_val = None
        position = "flat"
        n_long_i: Optional[int] = None
        n_short_i: Optional[int] = None

        if long_aligned is not None and short_aligned is not None:
            n_long_i = int(long_aligned.iloc[i]) if pd.notna(long_aligned.iloc[i]) else 0
            n_short_i = int(short_aligned.iloc[i]) if pd.notna(short_aligned.iloc[i]) else 0
            position = position_label_from_counts(n_long_i, n_short_i)
            signal_val = float(n_long_i - n_short_i)
        elif signals is not None and date in signals.index:
            sv = signals.loc[date]
            signal_val = float(sv) if pd.notna(sv) else None
            if signal_val is not None:
                if signal_val > 0:
                    position = "long"
                elif signal_val < 0:
                    position = "short"

        frame = {
            "date": str(date.date()) if hasattr(date, "date") else str(date),
            "pnl_today": float(net_returns.iloc[i]),
            "cumulative_pnl": float(cum.iloc[i] - 1.0),
            "drawdown": float(dd.iloc[i]),
            "rolling_sortino": (
                float(rolling_sortino.iloc[i]) if pd.notna(rolling_sortino.iloc[i]) else None
            ),
            "signal": signal_val,
            "position": position,
            "n_long": n_long_i,
            "n_short": n_short_i,
        }
        frames.append(frame)

    logger.info("Precomputed %d replay frames", len(frames))
    return frames


def frames_to_parquet(frames: List[Dict], path: str) -> None:
    """Persist precomputed frames to a Parquet file."""
    df = pd.DataFrame(frames)
    df.to_parquet(path, index=False)
    logger.info("Saved %d frames to %s", len(frames), path)


def frames_from_parquet(path: str) -> List[Dict[str, Any]]:
    """Load precomputed frames from a Parquet file."""
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")
