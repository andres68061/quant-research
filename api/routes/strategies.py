"""Strategy catalog: registered strategy metadata for UIs and clients."""

from fastapi import APIRouter

from core.strategies import list_strategies

router = APIRouter(tags=["strategies"])


@router.get("/strategies")
def get_strategies_catalog() -> dict:
    """
    Return all registered strategies (id, title, description, kind, post_path).

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
            }
        )
    return {"strategies": strategies}
