"""Walk-forward validation results endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/walkforward", tags=["walkforward"])


@router.get("/results")
def walkforward_results() -> dict:
    """Placeholder -- walk-forward results are returned inline from /run-ml-strategy."""
    return {
        "message": "Use POST /run-ml-strategy to run walk-forward validation. "
        "Results are returned inline in the response."
    }
