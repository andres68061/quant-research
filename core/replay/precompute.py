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

from core.metrics.performance import (
    calculate_cumulative_returns,
    calculate_drawdown,
    calculate_sharpe_ratio,
)

logger = logging.getLogger(__name__)


def precompute_backtest_frames(
    net_returns: pd.Series,
    signals: Optional[pd.Series] = None,
    rolling_window: int = 60,
) -> List[Dict[str, Any]]:
    """
    Build frame-by-frame replay data from a backtest result.

    Each frame captures:
    - date
    - daily PnL (net return)
    - cumulative PnL
    - drawdown
    - rolling Sharpe (if enough history)
    - signal value (if provided)
    - position label (long / short / flat)

    Args:
        net_returns: Series of daily net returns (date-indexed)
        signals: Optional Series of signal values aligned to same dates
        rolling_window: Window for rolling Sharpe (default 60 trading days)

    Returns:
        List of frame dictionaries ready for JSON serialisation

    Example:
        >>> frames = precompute_backtest_frames(results["net_return"])
        >>> len(frames)
        500
    """
    cum = calculate_cumulative_returns(net_returns)
    dd = calculate_drawdown(net_returns)

    rolling_sharpe = pd.Series(np.nan, index=net_returns.index)
    if len(net_returns) > rolling_window:
        for i in range(rolling_window, len(net_returns)):
            window = net_returns.iloc[i - rolling_window : i]
            rolling_sharpe.iloc[i] = calculate_sharpe_ratio(window)

    frames: List[Dict[str, Any]] = []
    for i, date in enumerate(net_returns.index):
        signal_val = None
        position = "flat"
        if signals is not None and date in signals.index:
            sv = signals.loc[date]
            signal_val = float(sv) if pd.notna(sv) else None
            if signal_val is not None:
                if signal_val > 0:
                    position = "long"
                elif signal_val < 0:
                    position = "short"

        frame = {
            "date": str(date.date()),
            "pnl_today": float(net_returns.iloc[i]),
            "cumulative_pnl": float(cum.iloc[i]),
            "drawdown": float(dd.iloc[i]),
            "rolling_sharpe": (
                float(rolling_sharpe.iloc[i])
                if pd.notna(rolling_sharpe.iloc[i])
                else None
            ),
            "signal": signal_val,
            "position": position,
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
