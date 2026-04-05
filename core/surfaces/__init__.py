"""
Volatility surface utilities: Black–Scholes pricing, implied volatility, and IV grids.

Dupire, SABR/SVI, and API routes are not implemented here; see roadmap.
"""

from core.exceptions import ImpliedVolatilityError
from core.surfaces.black_scholes import (
    OptionKind,
    black_scholes_delta,
    black_scholes_price,
    d1,
    d2,
    implied_volatility,
)
from core.surfaces.grid import implied_vol_surface_grid

__all__ = [
    "OptionKind",
    "ImpliedVolatilityError",
    "black_scholes_delta",
    "black_scholes_price",
    "d1",
    "d2",
    "implied_volatility",
    "implied_vol_surface_grid",
]
