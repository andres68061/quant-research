"""Earnings surprise (SUE) factors from announcement-dated actual vs. estimated EPS.

This is the one dataset in the FMP footprint that is **announcement dated** rather
than period dated: a row stamped 2024-01-29 was public that day. That makes it the
only fundamental input usable without a filing-date join, and it is what post-
earnings announcement drift (PEAD) is built on.

Three surprise definitions are emitted because the literature does not agree on
one, and they disagree most exactly where it matters (small earnings, loss-making
firms):

- ``sue_price_scaled`` — surprise divided by the pre-announcement share price.
  Scale-free, stable through zero earnings, and the Livnat & Mendenhall (2006)
  preference. **The default.**
- ``sue_std_scaled`` — surprise divided by the standard deviation of the firm's
  own past surprises. The classical SUE; expresses the surprise in units of that
  firm's normal forecast error, but explodes for firms with a history of exact
  hits.
- ``eps_surprise_pct`` — surprise as a fraction of the estimate. Intuitive and
  useless near zero; provided for reconciliation with vendor "surprise %" columns,
  not for ranking.

Leakage rule: every scaling input is drawn strictly from **before** the
announcement — the price is the prior trading day's close and the surprise
standard deviation is computed on past surprises only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

# Quarters of past surprises used for the standard-deviation scaling.
SURPRISE_HISTORY_QUARTERS = 8
MIN_SURPRISE_HISTORY = 6

# Trading days the drift is held. Bernard & Thomas (1989) find drift persisting
# roughly one quarter after the announcement.
DRIFT_WINDOW_TRADING_DAYS = 60

SURPRISE_COLUMNS: tuple[str, ...] = (
    "sue_price_scaled",
    "sue_std_scaled",
    "eps_surprise_pct",
    "revenue_surprise_pct",
    "eps_growth_yoy",
    "announcement_ordinal",
)

_REQUIRED_FIELDS = {"date", "epsActual", "epsEstimated"}


_EPOCH = pd.Timestamp("1970-01-01")


def _epoch_day_ordinal(dates: pd.DatetimeIndex) -> pd.Index:
    """
    Whole days from the epoch, independent of the index's datetime resolution.

    ``DatetimeIndex.astype("int64")`` exposes the underlying integer
    representation, whose unit is nanoseconds on pandas 2.x but microseconds on
    pandas 3.0 — so dividing it by a nanoseconds-per-day constant silently
    collapses every date onto the same ordinal. Differencing against a fixed
    timestamp yields days under either resolution.

    Args:
        dates: Tz-aware or tz-naive datetime index.

    Returns:
        Integer index of whole days since 1970-01-01.
    """
    naive = dates.tz_localize(None) if dates.tz is not None else dates
    return (naive.normalize() - _EPOCH).days


def compute_announcement_surprises(
    earnings: pd.DataFrame,
    close_prices: pd.Series,
) -> pd.DataFrame:
    """
    Build one symbol's announcement-dated surprise metrics.

    Args:
        earnings: Raw rows from the ``earnings`` dataset for one symbol, with
            ``date``, ``epsActual``, ``epsEstimated`` and optionally
            ``revenueActual``/``revenueEstimated``. Rows without an actual are
            future scheduled announcements and are dropped.
        close_prices: That symbol's daily adjusted closes, tz-aware ascending.
            Used only for the price scaling; the price taken is the last close
            **strictly before** the announcement.

    Returns:
        DataFrame indexed by announcement date (tz-aware, matching
        ``close_prices``) with :data:`SURPRISE_COLUMNS`. Empty when the symbol has
        no announcements carrying both an actual and an estimate.

    Raises:
        DataSchemaError: If required vendor fields are missing.

    Example:
        >>> surprises = compute_announcement_surprises(aapl_earnings, aapl_close)
        ... # doctest: +SKIP
    """
    empty = pd.DataFrame(columns=list(SURPRISE_COLUMNS))
    if earnings.empty:
        return empty
    if missing := _REQUIRED_FIELDS - set(earnings.columns):
        raise DataSchemaError(f"earnings payload missing fields: {missing}")

    announced = earnings.dropna(subset=["epsActual"]).sort_values("date").copy()
    if announced.empty:
        return empty

    announcement_date = pd.to_datetime(announced["date"])
    if close_prices.index.tz is not None and announcement_date.dt.tz is None:
        announcement_date = announcement_date.dt.tz_localize(close_prices.index.tz)
    announced = announced.set_index(pd.DatetimeIndex(announcement_date, name="date"))
    announced = announced[~announced.index.duplicated(keep="last")]

    actual = pd.to_numeric(announced["epsActual"], errors="coerce")
    estimated = pd.to_numeric(announced["epsEstimated"], errors="coerce")
    surprise = actual - estimated

    # Last close STRICTLY before the announcement — an announcement can move the
    # same day's close, so using it would scale the surprise by its own effect.
    prices = close_prices.dropna().sort_index()
    positions = prices.index.searchsorted(announced.index, side="left") - 1
    prior_close = pd.Series(
        np.where(positions >= 0, prices.to_numpy()[positions.clip(min=0)], np.nan),
        index=announced.index,
        dtype="float64",
    )

    # Standard deviation of PAST surprises only (shift before rolling).
    past_surprise_std = (
        surprise.shift(1).rolling(SURPRISE_HISTORY_QUARTERS, min_periods=MIN_SURPRISE_HISTORY).std()
    )

    result = pd.DataFrame(index=announced.index)
    result["sue_price_scaled"] = surprise / prior_close.where(prior_close > 0)
    result["sue_std_scaled"] = surprise / past_surprise_std.where(past_surprise_std > 0)
    result["eps_surprise_pct"] = surprise / estimated.abs().where(estimated.abs() > 0)

    if {"revenueActual", "revenueEstimated"}.issubset(announced.columns):
        revenue_actual = pd.to_numeric(announced["revenueActual"], errors="coerce")
        revenue_estimated = pd.to_numeric(announced["revenueEstimated"], errors="coerce")
        result["revenue_surprise_pct"] = (revenue_actual - revenue_estimated) / (
            revenue_estimated.abs().where(revenue_estimated.abs() > 0)
        )
    else:
        result["revenue_surprise_pct"] = np.nan

    prior_year_eps = actual.shift(4)
    result["eps_growth_yoy"] = (actual - prior_year_eps) / prior_year_eps.abs().where(
        prior_year_eps.abs() > 0
    )

    # Days since the epoch, so the daily panel can derive days-since-announcement
    # after forward-filling (needed to slice PEAD event windows).
    result["announcement_ordinal"] = _epoch_day_ordinal(announced.index).astype("float64")

    return result.replace([np.inf, -np.inf], np.nan)[list(SURPRISE_COLUMNS)]


def add_days_since_announcement(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ``days_since_earnings`` from the forward-filled announcement ordinal.

    PEAD is an event-window effect, so a backtest needs to know how far into the
    drift window each row sits — a surprise 55 days old is not the same signal as
    one from yesterday.

    Args:
        panel: MultiIndex (date, symbol) panel carrying ``announcement_ordinal``.

    Returns:
        Copy of ``panel`` with an added ``days_since_earnings`` column (calendar
        days) and ``announcement_ordinal`` dropped.
    """
    if "announcement_ordinal" not in panel.columns:
        return panel

    dates = panel.index.get_level_values("date")
    date_ordinal = _epoch_day_ordinal(dates)
    result = panel.copy()
    result["days_since_earnings"] = (
        date_ordinal - result["announcement_ordinal"].to_numpy()
    ).astype("float32")
    return result.drop(columns=["announcement_ordinal"])
