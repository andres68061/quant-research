"""
Walk-forward validation for time-series models.

Provides an expanding-window splitter that guarantees no data leakage:
train always precedes test in calendar time, and training rows whose
forward-looking labels overlap the test window are purged (Bailey &
López de Prado, *Advances in Financial Machine Learning*, ch. 7).
"""

import logging
from typing import List, Tuple

import pandas as pd

from core.exceptions import ConfigError

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation with expanding window for time series.

    Design:
    - Expanding window (not rolling) -- uses all historical data
    - Fixed test period (5 days default)
    - No data leakage: train precedes test, and the last
      ``label_horizon_days - 1 + embargo_days`` training rows are purged
      because their forward-looking labels are computed from returns that
      fall inside the test window. Without purging, a model trained on a
      label built with ``prices.shift(-horizon)`` has partially seen the
      outcome it is then tested on.
    """

    def __init__(
        self,
        initial_train_days: int = 63,
        test_days: int = 5,
        max_splits: int = 50,
        label_horizon_days: int = 1,
        embargo_days: int = 0,
    ):
        """
        Args:
            initial_train_days: Initial training period (default 63 ~ 3 months).
                Note: 63 days is thin for anything beyond a simple model —
                prefer 252+ when data allows.
            test_days: Test period length (default 5 = 1 week)
            max_splits: Maximum number of splits to prevent runaway training
            label_horizon_days: Forward horizon of the label in trading days
                (1 = next-day direction). ``horizon - 1`` rows are purged
                from the end of every training window.
            embargo_days: Extra rows dropped from the end of every training
                window on top of the purge, as a safety margin against
                serial correlation between train and test.
        """
        if label_horizon_days < 1:
            raise ConfigError(f"label_horizon_days must be >= 1, got {label_horizon_days}")
        if embargo_days < 0:
            raise ConfigError(f"embargo_days must be >= 0, got {embargo_days}")
        purge_days = (label_horizon_days - 1) + embargo_days
        if initial_train_days <= purge_days:
            raise ConfigError(
                f"initial_train_days ({initial_train_days}) must exceed purge + embargo "
                f"({purge_days}) or every training window would be empty"
            )
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        self.max_splits = max_splits
        self.label_horizon_days = label_horizon_days
        self.embargo_days = embargo_days
        self.purge_days = purge_days

    def create_splits(self, df: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Create walk-forward splits with purged training windows.

        Args:
            df: Feature DataFrame (rows = dates)

        Returns:
            List of (train_indices, test_indices) tuples. Training indices
            stop ``purge_days`` rows before the test window starts.
        """
        splits: List[Tuple[pd.Index, pd.Index]] = []
        total_rows = len(df)

        if total_rows < self.initial_train_days + self.test_days:
            logger.warning(
                "Insufficient data: %d rows (need %d)",
                total_rows,
                self.initial_train_days + self.test_days,
            )
            return splits

        train_end = self.initial_train_days

        while train_end + self.test_days <= total_rows:
            train_indices = df.index[: train_end - self.purge_days]
            test_indices = df.index[train_end : train_end + self.test_days]
            splits.append((train_indices, test_indices))

            if len(splits) >= self.max_splits:
                logger.info("Reached max_splits limit (%d). Stopping.", self.max_splits)
                break

            train_end += self.test_days

        logger.info(
            "Created %d walk-forward splits (purged %d rows per train window)",
            len(splits),
            self.purge_days,
        )
        return splits
