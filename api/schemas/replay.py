"""Response models for replay / animation frames."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ReplayFrame(BaseModel):
    date: str
    signal: Optional[float] = None
    position: Optional[str] = None
    pnl_today: Optional[float] = None
    cumulative_pnl: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    surface_data: Optional[Dict[str, Any]] = None


class ReplayResponse(BaseModel):
    strategy_id: str
    total_frames: int
    frames: List[ReplayFrame]
