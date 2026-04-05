"""Walk-forward validation results endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/walkforward", tags=["walkforward"])


class WalkForwardUsageResponse(BaseModel):
    """Explains how to obtain walk-forward results."""

    endpoint: str = Field(..., description="The endpoint to call")
    method: str = Field(..., description="HTTP method")
    description: str = Field(..., description="How to use it")
    example_body: dict = Field(..., description="Example request body")


@router.get("/results", response_model=WalkForwardUsageResponse)
def walkforward_results() -> WalkForwardUsageResponse:
    """Walk-forward results are returned inline from POST /run-ml-strategy.

    This endpoint documents the correct way to obtain walk-forward
    validation results including fold-by-fold metrics, feature importance,
    and confusion matrix data.
    """
    return WalkForwardUsageResponse(
        endpoint="/run-ml-strategy",
        method="POST",
        description=(
            "Walk-forward validation results are computed and returned inline "
            "when you run an ML strategy. The response includes: overall accuracy, "
            "precision, recall, F1, ROC-AUC, confusion matrix, per-fold metrics "
            "(train/test dates, accuracy), and feature importance."
        ),
        example_body={
            "symbol": "GLD",
            "model_type": "xgboost",
            "initial_train_days": 63,
            "test_days": 5,
            "max_splits": 50,
        },
    )
