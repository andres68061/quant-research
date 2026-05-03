"""Hidden Markov Model regime detection signal.

Infers latent market regimes (risk-on, neutral, risk-off) from a joint
feature set and converts filtered state probabilities into portfolio
exposure signals.

References:
    Hamilton (1989), "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle."
    Ang & Bekaert (2002), "International Asset Allocation with Regime Shifts."
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

_EPS = 1e-10


def build_regime_features(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    macro_z: pd.DataFrame,
    vix: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build daily feature matrix for HMM regime detection.

    All inputs are aligned to a common business-day index and forward-filled.
    The output has no NaN rows (leading NaN period is dropped).

    Args:
        prices: Wide DataFrame (date x symbols) of adjusted close prices.
        factors: MultiIndex (date, symbol) DataFrame with ``beta_60d``
            and ``vol_60d`` columns.
        macro_z: DataFrame with macro z-score columns (e.g. from
            ``macro_z.parquet``).
        vix: Optional daily VIX close series.

    Returns:
        DataFrame with columns: ``return_dispersion``, ``beta_spread``,
        ``vol_median``, macro z-score columns, and optionally ``vix``.

    Example:
        >>> features = build_regime_features(prices, factors, macro_z, vix)
        >>> features.columns.tolist()
        ['return_dispersion', 'beta_spread', 'vol_median', ..., 'vix']
    """
    prices_naive = _strip_tz(prices)
    factors_naive = _strip_tz_multiindex(factors, level="date")
    macro_z_naive = _strip_tz(macro_z)

    daily_returns = prices_naive.pct_change()
    return_dispersion = daily_returns.std(axis=1).rename("return_dispersion")

    beta_spread = _cross_sectional_spread(factors_naive, "beta_60d")
    vol_median = _cross_sectional_median(factors_naive, "vol_60d")

    macro_cols = _select_macro_columns(macro_z_naive)

    parts: list[pd.Series | pd.DataFrame] = [
        return_dispersion,
        beta_spread,
        vol_median,
        macro_cols,
    ]
    if vix is not None:
        vix_clean = _strip_tz(vix.copy()).rename("vix")
        parts.append(vix_clean)

    combined = pd.concat(parts, axis=1).sort_index()
    combined = combined.asfreq("B").ffill()
    combined = combined.dropna()

    logger.info(
        "Regime features: %d rows, %d columns, range %s .. %s",
        len(combined),
        combined.shape[1],
        combined.index[0].date(),
        combined.index[-1].date(),
    )
    return combined


def fit_regime_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    train_window: int = 252 * 5,
    step: int = 21,
    market_returns: Optional[pd.Series] = None,
    covariance_type: str = "diag",
    predict_mode: Literal["filtered", "smoothed"] = "filtered",
) -> pd.DataFrame:
    """Walk-forward HMM regime estimation.

    On each step date, the model is fit on the trailing ``train_window``
    days (or all available data if shorter). Features are standardized
    using only the train window to prevent data leakage. State labels are
    assigned by sorting on the mean market return observed during each
    state in-sample.

    Args:
        features: Output of :func:`build_regime_features` (no NaN).
        n_states: Number of latent states (default 3).
        train_window: Rolling training window in business days.
        step: Number of days between re-estimations.
        market_returns: Daily market return series used for state labelling.
            If None, ``return_dispersion`` is used as a proxy (lower
            dispersion ≈ risk-on).
        covariance_type: hmmlearn covariance parameterisation. ``"diag"``
            (default) is stable with correlated financial features;
            ``"full"`` can overfit on rolling windows.
        predict_mode: ``"filtered"`` uses only observations through each
            prediction date. ``"smoothed"`` preserves the legacy behavior of
            using the full prediction block in the posterior calculation.

    Returns:
        DataFrame indexed by date with columns ``p_risk_on``,
        ``p_neutral``, ``p_risk_off`` (state probabilities) and
        ``regime`` (argmax label as string).

    Raises:
        ValueError: If ``features`` has fewer rows than ``n_states * 10``.

    Example:
        >>> probs = fit_regime_hmm(features, n_states=3)
        >>> probs[["p_risk_on", "p_risk_off"]].plot()
    """
    min_rows = n_states * 10
    if len(features) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} rows to fit {n_states}-state HMM, " f"got {len(features)}"
        )

    features = _strip_tz(features)
    if market_returns is not None:
        market_returns = _strip_tz(market_returns)

    dates = features.index
    n = len(dates)
    prob_records: list[dict] = []

    cursor = max(train_window, min_rows)
    while cursor < n:
        end_train = cursor
        start_train = max(0, end_train - train_window)
        end_pred = min(cursor + step, n)

        train_data = features.iloc[start_train:end_train].values
        pred_data = features.iloc[cursor:end_pred].values
        pred_dates = dates[cursor:end_pred]

        mu = train_data.mean(axis=0)
        sigma = train_data.std(axis=0) + _EPS
        train_scaled = (train_data - mu) / sigma
        pred_scaled = (pred_data - mu) / sigma

        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=200,
            tol=1e-3,
            random_state=42,
            verbose=False,
        )
        try:
            model.fit(train_scaled)
            _fix_degenerate_transmat(model)
            probs = _predict_state_probabilities(
                model,
                train_scaled,
                pred_scaled,
                predict_mode=predict_mode,
            )
        except Exception as exc:
            logger.warning("HMM fit/predict failed at cursor=%d: %s", cursor, exc)
            cursor += step
            continue

        label_order = _label_states(
            model, train_scaled, market_returns, dates[start_train:end_train], n_states
        )

        if n_states >= 3:
            regime_map = {0: "risk_on", 1: "neutral", 2: "risk_off"}
        else:
            regime_map = {0: "risk_on", 1: "risk_off"}

        for i, date in enumerate(pred_dates):
            row = {"date": date}
            ordered_probs = probs[i, label_order]
            row["p_risk_on"] = ordered_probs[0]
            if n_states >= 3:
                row["p_neutral"] = ordered_probs[1]
                row["p_risk_off"] = ordered_probs[2]
            else:
                row["p_risk_off"] = ordered_probs[1]
            row["regime"] = regime_map[np.argmax(ordered_probs)]
            prob_records.append(row)

        cursor += step

    if not prob_records:
        logger.warning("No regime probabilities produced; check data length")
        return pd.DataFrame()

    result = pd.DataFrame(prob_records).set_index("date")
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()

    logger.info(
        "Regime HMM: %d dates, regime distribution: %s",
        len(result),
        result["regime"].value_counts().to_dict(),
    )
    return result


def regime_signal(
    regime_probs: pd.DataFrame,
    mode: Literal["exposure_scale", "long_short"] = "exposure_scale",
) -> pd.Series:
    """Convert regime probabilities into a trading signal.

    Args:
        regime_probs: Output of :func:`fit_regime_hmm`.
        mode: Signal mode:
            - ``"exposure_scale"``: continuous [0, 1] from ``p_risk_on``,
              suitable for scaling gross exposure.
            - ``"long_short"``: +1 (risk_on), 0 (neutral), -1 (risk_off)
              from the argmax regime label.

    Returns:
        Series named ``"regime_signal"`` aligned to the input index.

    Example:
        >>> sig = regime_signal(probs, mode="exposure_scale")
        >>> sig.describe()
    """
    if mode == "exposure_scale":
        signal = regime_probs["p_risk_on"].clip(0.0, 1.0)
    elif mode == "long_short":
        mapping = {"risk_on": 1, "neutral": 0, "risk_off": -1}
        signal = regime_probs["regime"].map(mapping).astype(float)
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'exposure_scale' or 'long_short'")

    signal.name = "regime_signal"
    return signal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_tz(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Return *obj* with a tz-naive DatetimeIndex (no-op if already naive)."""
    if getattr(obj.index, "tz", None) is not None:
        obj = obj.copy()
        obj.index = obj.index.tz_localize(None)
    return obj


def _strip_tz_multiindex(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Drop tz from a named level of a MultiIndex DataFrame."""
    if level not in df.index.names:
        return df
    idx = df.index
    pos = idx.names.index(level)
    lvl_values = idx.levels[pos]
    if getattr(lvl_values, "tz", None) is None:
        return df
    df = df.copy()
    df.index = idx.set_levels(lvl_values.tz_localize(None), level=level)
    return df


def _fix_degenerate_transmat(model: GaussianHMM) -> None:
    """Repair transition matrix rows that sum to zero (unvisited states).

    hmmlearn raises ValueError on predict if any transmat row sums to 0.
    We assign uniform transition probability to such rows so prediction
    can proceed.
    """
    row_sums = model.transmat_.sum(axis=1)
    degenerate = row_sums < 1e-10
    if degenerate.any():
        n = model.n_components
        model.transmat_[degenerate] = 1.0 / n
        model.transmat_ /= model.transmat_.sum(axis=1, keepdims=True)


def _predict_state_probabilities(
    model: GaussianHMM,
    train_scaled: np.ndarray,
    pred_scaled: np.ndarray,
    *,
    predict_mode: Literal["filtered", "smoothed"],
) -> np.ndarray:
    """Predict state probabilities without leaking future prediction rows."""
    if predict_mode == "smoothed":
        return model.predict_proba(pred_scaled)

    if predict_mode != "filtered":
        raise ValueError("predict_mode must be either 'filtered' or 'smoothed'")

    rows = []
    for i in range(len(pred_scaled)):
        observed_through_date = np.vstack([train_scaled, pred_scaled[: i + 1]])
        rows.append(model.predict_proba(observed_through_date)[-1])
    return np.vstack(rows)


def _cross_sectional_spread(factors: pd.DataFrame, col: str, quantile: float = 0.1) -> pd.Series:
    """Top-decile minus bottom-decile mean of *col* per date."""
    if col not in factors.columns:
        return pd.Series(dtype=float, name=f"{col}_spread")

    grouped = factors[col].dropna().groupby(level="date")
    top = grouped.quantile(1.0 - quantile)
    bottom = grouped.quantile(quantile)
    spread = (top - bottom).rename(f"{col}_spread")
    return spread


def _cross_sectional_median(factors: pd.DataFrame, col: str) -> pd.Series:
    """Median of *col* across symbols per date."""
    if col not in factors.columns:
        return pd.Series(dtype=float, name=f"{col}_median")

    return factors[col].dropna().groupby(level="date").median().rename(f"{col}_median")


def _select_macro_columns(macro_z: pd.DataFrame) -> pd.DataFrame:
    """Pick macro z-score columns, preferring yield-curve and rates features."""
    preferred = [
        "macro_z_t10y2y",
        "macro_z_fed_funds",
        "macro_z_unrate",
        "macro_z_dgs10",
        "macro_z_cpi_yoy",
    ]
    available = [c for c in preferred if c in macro_z.columns]
    if not available:
        available = list(macro_z.columns[:3])
    return macro_z[available]


def _label_states(
    model: GaussianHMM,
    train_scaled: np.ndarray,
    market_returns: Optional[pd.Series],
    train_dates: pd.DatetimeIndex,
    n_states: int,
) -> np.ndarray:
    """Return state index ordering: risk-on first, risk-off last.

    If market_returns is available, states are sorted by their mean
    market return (highest = risk-on). Otherwise, states are sorted
    by mean return_dispersion (lowest dispersion ≈ risk-on).
    """
    hidden = model.predict(train_scaled)

    if market_returns is not None:
        aligned = market_returns.reindex(train_dates).fillna(0.0).values
        state_means = [aligned[hidden == s].mean() for s in range(n_states)]
        return np.argsort(state_means)[::-1]

    state_means = np.array(
        [
            train_scaled[hidden == s, 0].mean() if (hidden == s).any() else 0.0
            for s in range(n_states)
        ]
    )
    return np.argsort(state_means)
