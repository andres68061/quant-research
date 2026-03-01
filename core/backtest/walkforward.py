"""
Walk-forward validation for time-series models.

Provides an expanding-window splitter that guarantees no data leakage:
train always precedes test in calendar time.
"""

import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation with expanding window for time series.

    Design:
    - Expanding window (not rolling) -- uses all historical data
    - Fixed test period (5 days default)
    - No data leakage (train on past, test on future)
    """

    def __init__(
        self,
        initial_train_days: int = 63,
        test_days: int = 5,
        max_splits: int = 50,
    ):
        """
        Args:
            initial_train_days: Initial training period (default 63 ~ 3 months)
            test_days: Test period length (default 5 = 1 week)
            max_splits: Maximum number of splits to prevent runaway training
        """
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        self.max_splits = max_splits

    def create_splits(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Create walk-forward splits.

        Args:
            df: Feature DataFrame (rows = dates)

        Returns:
            List of (train_indices, test_indices) tuples
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
            train_indices = df.index[:train_end]
            test_indices = df.index[train_end : train_end + self.test_days]
            splits.append((train_indices, test_indices))

            if len(splits) >= self.max_splits:
                logger.info(
                    "Reached max_splits limit (%d). Stopping.", self.max_splits
                )
                break

            train_end += self.test_days

        logger.info("Created %d walk-forward splits", len(splits))
        return splits
