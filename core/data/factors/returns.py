"""Return computation with bad-print rejection — the one place returns are made.

The ADR-0013 universe expansion brought in the full micro-cap tail, and with it
vendor defects that a 774-name S&P panel never contained. Measured on the
canonical panel (2026-08-13):

- max daily "return": **101,599,900%** (a 1,000,000x one-day move)
- 7,691 daily returns above +100%; 3,142 above +500%
- 40,009 sub-penny prices across 358 symbols

Almost all of these are un-adjusted reverse splits and quote errors on delisted
micro-caps. They are not returns, and they are not merely noisy: a mean, a
cross-sectional average, or a cumulative sum that includes one of them is
destroyed. A single infinite return in a cross-sectional mean turns *every*
symbol's abnormal return that day into infinity.

Two separable controls, because they answer different questions:

- ``max_abs_return`` — **defect rejection.** A stock can double on takeover news;
  it cannot rise 900,000% in a session. Beyond the bound the observation is
  discarded as a bad print, not winsorized, because winsorizing implies the
  magnitude meant something.
- ``min_price`` — **research eligibility.** Sub-dollar stocks are dominated by
  bid-ask bounce (a one-cent spread on a $0.05 stock is a 20% round trip), which
  is why the cross-sectional literature conventionally screens them out. This is
  a *choice about the universe*, so it defaults to off and must be stated
  wherever it is used.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A legitimate single-session gain above this is vanishingly rare (biotech
# approvals and takeovers cluster below +200%); above it, defects dominate.
DEFAULT_MAX_ABS_RETURN = 3.0

# Conventional research floor. $1 is permissive; $5 is the stricter convention.
DEFAULT_MIN_PRICE = 1.0


def compute_clean_returns(
    prices: pd.DataFrame,
    max_abs_return: float | None = DEFAULT_MAX_ABS_RETURN,
    min_price: float | None = None,
) -> pd.DataFrame:
    """
    Daily simple returns with non-finite values and bad prints removed.

    Args:
        prices: Wide date x symbol adjusted-close panel.
        max_abs_return: Discard observations whose absolute return exceeds this
            (3.0 = +300%). None disables defect rejection.
        min_price: Require this close AND the prior close to be at least this
            price for the return to count. None keeps every price level.

    Returns:
        Return panel shaped like ``prices``; rejected observations are NaN.

    Example:
        >>> returns = compute_clean_returns(prices, min_price=1.0)  # doctest: +SKIP
    """
    returns = prices.pct_change(fill_method=None)
    returns = returns.where(np.isfinite(returns))

    if min_price is not None:
        tradable = prices >= min_price
        # Both ends of the return must clear the floor: a jump from $0.02 to $3
        # is exactly the bad print the floor exists to exclude.
        returns = returns.where(tradable & tradable.shift(1))

    if max_abs_return is not None:
        extreme = returns.abs() > max_abs_return
        n_extreme = int(extreme.to_numpy().sum())
        if n_extreme:
            affected = int(extreme.any().sum())
            logger.warning(
                "Rejected %d returns beyond +/-%.0f%% as bad prints, across %d symbols",
                n_extreme,
                max_abs_return * 100,
                affected,
            )
            returns = returns.where(~extreme)

    return returns


def compute_abnormal_returns(
    prices: pd.DataFrame,
    max_abs_return: float | None = DEFAULT_MAX_ABS_RETURN,
    min_price: float | None = None,
    min_names_per_date: int = 20,
) -> pd.DataFrame:
    """
    Returns in excess of the equal-weighted cross-sectional mean of the same day.

    A simple market adjustment with no beta fit. The cross-sectional mean is
    computed **after** cleaning, so one defect cannot shift the benchmark for
    every other symbol — the failure mode that produced infinite CAR paths in the
    first expanded-universe PEAD run.

    Args:
        prices: Wide date x symbol adjusted-close panel.
        max_abs_return: Bad-print bound, per :func:`compute_clean_returns`.
        min_price: Optional research price floor.
        min_names_per_date: Dates with fewer surviving names are blanked
            entirely; a "market return" from three stocks is not a benchmark.

    Returns:
        Abnormal return panel shaped like ``prices``.
    """
    returns = compute_clean_returns(prices, max_abs_return, min_price)
    counts = returns.notna().sum(axis=1)
    market = returns.mean(axis=1).where(counts >= min_names_per_date)
    thin_dates = int((counts < min_names_per_date).sum())
    if thin_dates:
        logger.info(
            "Blanked %d dates with fewer than %d priced names", thin_dates, min_names_per_date
        )
    return returns.sub(market, axis=0)
