"""Strategy execution endpoints: backtest and ML strategies."""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import get_factors, get_prices
from api.schemas.metrics import EquityCurvePoint, PerformanceMetrics
from api.schemas.strategy import BacktestRequest, MLStrategyRequest
from api.schemas.walkforward import ConfusionMatrixResult, FoldResult, WalkForwardResult
from core.backtest.portfolio import sp500_universe_filter
from core.strategies import run_factor_cross_section_backtest
from core.metrics.performance import (
    calculate_cumulative_returns,
    calculate_performance_metrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["strategy"])

_last_backtest_returns = None  # cached for GET /equity-curve


@router.post("/run-backtest")
def run_backtest(req: BacktestRequest) -> dict:
    """Run a factor-based backtest and return metrics + equity curve."""
    global _last_backtest_returns

    factors = get_factors()
    prices = get_prices()
    if factors is None or prices is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    factor_col = req.factor_col or factors.columns[0]
    if factor_col not in factors.columns:
        raise HTTPException(status_code=400, detail=f"Factor '{factor_col}' not found")

    start = (
        pd.Timestamp(req.start_date)
        if req.start_date
        else pd.Timestamp(date.today() - timedelta(days=5 * 365))
    )
    end = pd.Timestamp(req.end_date) if req.end_date else pd.Timestamp(date.today())

    uf = sp500_universe_filter() if req.survivorship_free else None

    net_returns = run_factor_cross_section_backtest(
        factors,
        prices,
        factor_col=factor_col,
        start=start,
        end=end,
        top_pct=req.top_pct,
        bottom_pct=req.bottom_pct,
        long_only=req.long_only,
        rebalance_freq=req.rebalance_freq,
        transaction_cost=req.transaction_cost_bps / 10_000,
        universe_filter=uf,
    )
    _last_backtest_returns = net_returns

    metrics = calculate_performance_metrics(net_returns)

    cum = calculate_cumulative_returns(net_returns)
    equity = [
        EquityCurvePoint(date=str(d.date()), cumulative_return=float(v)) for d, v in cum.items()
    ]

    return {
        "metrics": PerformanceMetrics(**metrics).model_dump(),
        "equity_curve": [e.model_dump() for e in equity[-500:]],
        "total_days": len(net_returns),
    }


@router.get("/equity-curve")
def get_equity_curve(tail: int = 500) -> dict:
    """Return the equity curve from the last backtest run."""
    if _last_backtest_returns is None:
        raise HTTPException(status_code=404, detail="No backtest has been run yet")

    cum = calculate_cumulative_returns(_last_backtest_returns)
    points = [
        {"date": str(d.date()), "cumulative_return": float(v)} for d, v in cum.tail(tail).items()
    ]
    return {"count": len(points), "equity_curve": points}


@router.post("/run-ml-strategy")
def run_ml_strategy(req: MLStrategyRequest) -> dict:
    """Run an ML direction-prediction strategy via walk-forward validation."""
    try:
        from core.data.ml_features import create_ml_features_with_transparency
        from core.models.commodity_direction import run_walk_forward_validation

        prices = get_prices()
        if prices is None:
            raise HTTPException(status_code=503, detail="Price data not loaded")

        if req.symbol not in prices.columns:
            raise HTTPException(status_code=404, detail=f"Symbol '{req.symbol}' not found")

        price_series = prices[req.symbol].dropna()
        features, metadata = create_ml_features_with_transparency(price_series, symbol=req.symbol)

        wf_results = run_walk_forward_validation(
            features,
            model_type=req.model_type,
            initial_train_days=req.initial_train_days,
            test_days=req.test_days,
            max_splits=req.max_splits,
            verbose=False,
        )

        if "error" in wf_results:
            raise HTTPException(status_code=400, detail=wf_results["error"])

        overall = wf_results["overall_metrics"]

        folds = [
            FoldResult(
                fold=s["split"],
                train_size=s["train_size"],
                test_size=s["test_size"],
                accuracy=s["accuracy"],
                train_start=(
                    str(s["train_start"].date())
                    if hasattr(s["train_start"], "date")
                    else str(s["train_start"])
                ),
                train_end=(
                    str(s["train_end"].date())
                    if hasattr(s["train_end"], "date")
                    else str(s["train_end"])
                ),
                test_start=(
                    str(s["test_start"].date())
                    if hasattr(s["test_start"], "date")
                    else str(s["test_start"])
                ),
                test_end=(
                    str(s["test_end"].date())
                    if hasattr(s["test_end"], "date")
                    else str(s["test_end"])
                ),
            )
            for s in wf_results["split_metrics"]
        ]

        fi = wf_results.get("feature_importance")
        feature_importance = (
            [{"feature": k, "importance": float(v)} for k, v in fi.items()]
            if fi is not None
            else None
        )

        cm = None
        if all(
            k in overall
            for k in ("true_negatives", "false_positives", "false_negatives", "true_positives")
        ):
            cm = ConfusionMatrixResult(
                true_negatives=overall["true_negatives"],
                false_positives=overall["false_positives"],
                false_negatives=overall["false_negatives"],
                true_positives=overall["true_positives"],
            )

        return {
            "walkforward": WalkForwardResult(
                model_type=wf_results["model_type"],
                n_splits=wf_results["n_splits"],
                overall_accuracy=overall.get("accuracy", 0),
                overall_precision=overall.get("precision"),
                overall_recall=overall.get("recall"),
                overall_f1=overall.get("f1_score"),
                overall_roc_auc=overall.get("roc_auc"),
                confusion_matrix=cm,
                folds=folds,
            ).model_dump(),
            "feature_importance": feature_importance,
            "metadata": {
                "symbol": req.symbol,
                "total_features": metadata.get("total_features"),
                "final_rows": metadata.get("final_rows"),
            },
        }

    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"ML dependencies not installed: {exc}")
