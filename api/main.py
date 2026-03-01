"""
FastAPI application factory.

Run with:
    uvicorn api.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import ALLOWED_ORIGINS
from api.dependencies import load_data
from api.routes import (
    banxico,
    commodities,
    data,
    exclusions,
    fred,
    health,
    metrics,
    momentum,
    portfolio,
    replay,
    sectors,
    simulation,
    strategy,
    walkforward,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load shared datasets on startup."""
    logger.info("Loading data into memory...")
    load_data()
    logger.info("Data loading complete. API ready.")
    yield


app = FastAPI(
    title="Quant Analytics API",
    description=(
        "REST API for the Quant Analytics Platform. "
        "Exposes backtesting, ML strategies, performance metrics, "
        "and replay endpoints."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(data.router)
app.include_router(strategy.router)
app.include_router(metrics.router)
app.include_router(walkforward.router)
app.include_router(replay.router)
app.include_router(momentum.router)
app.include_router(portfolio.router)
app.include_router(banxico.router)
app.include_router(commodities.router)
app.include_router(fred.router)
app.include_router(sectors.router)
app.include_router(simulation.router)
app.include_router(exclusions.router)
