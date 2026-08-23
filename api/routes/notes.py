"""Research note endpoints.

Notes live in :mod:`core.research.notes` — the numbers in them are asserted
against the persisted experiment JSON by the test suite, so what the page shows
cannot drift from what the experiment produced.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from core.research.notes import VERDICT_LABELS, as_dicts, get_note

router = APIRouter(prefix="/research-notes", tags=["research"])


@router.get("")
def list_notes() -> dict[str, Any]:
    """Index of notes, newest first, without the full prose."""
    notes = sorted(as_dicts(full=False), key=lambda n: str(n["run_date"]), reverse=True)
    return {
        "notes": notes,
        "verdicts": [{"id": key, "label": label} for key, label in VERDICT_LABELS.items()],
    }


@router.get("/{note_id}")
def read_note(note_id: str) -> dict[str, Any]:
    """One note in full."""
    if get_note(note_id) is None:
        raise HTTPException(status_code=404, detail=f"No research note '{note_id}'")
    return next(n for n in as_dicts(full=True) if n["id"] == note_id)
