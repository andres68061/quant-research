"""Glossary endpoint — the single definition registry, served to any page.

Definitions live in :mod:`core.research.glossary` and nowhere else, so a term
defined on the glossary page, the data-health page and inside a research note is
guaranteed to read identically.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from core.research.glossary import CATEGORIES, as_dicts

router = APIRouter(prefix="/glossary", tags=["glossary"])


@router.get("")
def get_glossary() -> dict[str, Any]:
    """Every term, with its category and cross-references."""
    return {
        "categories": [{"id": key, "label": label} for key, label in CATEGORIES],
        "terms": as_dicts(),
    }
