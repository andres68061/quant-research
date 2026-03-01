"""
Tests for database path configuration.

Skipped until core.data.database.StockDatabase is implemented.
"""

import pytest

try:
    from core.data.database import StockDatabase

    _HAS_DATABASE = True
except ImportError:
    _HAS_DATABASE = False


@pytest.mark.skipif(not _HAS_DATABASE, reason="StockDatabase module not yet implemented")
def test_default_db_path_matches_config():
    from pathlib import Path

    from config.settings import get_database_path

    db = StockDatabase()
    try:
        assert Path(db.db_path) == get_database_path()
    finally:
        db.close()
