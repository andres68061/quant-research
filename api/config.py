"""
API-specific configuration.

Reads from environment / .env with sensible defaults for local development.
"""

import os

from dotenv import load_dotenv

load_dotenv()

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

ALLOWED_ORIGINS: list[str] = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8050,http://localhost:3000"
).split(",")
