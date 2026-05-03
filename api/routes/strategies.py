"""Strategy catalog: registered strategy metadata for UIs and clients."""

from fastapi import APIRouter

from core.strategies import list_strategies

router = APIRouter(tags=["strategies"])


@router.get("/strategies")
def get_strategies_catalog() -> dict:
    """
    Return all registered strategies, including the hypothesis, reference,
    expected Sharpe range, and known limitations used to populate the
    Strategy Brief and Methodology pages in the frontend.

    Execution remains on the referenced POST routes (e.g. /run-backtest).
    """
    strategies = []
    for meta in list_strategies():
        strategies.append(
            {
                "id": meta.id,
                "title": meta.title,
                "description": meta.description,
                "kind": meta.kind.value,
                "post_path": meta.post_path,
                "hypothesis": meta.hypothesis,
                "reference": meta.reference,
                "expected_sharpe_range": (
                    list(meta.expected_sharpe_range)
                    if meta.expected_sharpe_range is not None
                    else None
                ),
                "known_limitations": list(meta.known_limitations),
            }
        )
    return {"strategies": strategies}
