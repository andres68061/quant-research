from pathlib import Path

from config.settings import get_database_path
from src.data.database import StockDatabase


def test_default_db_path_matches_config(tmp_path):
    # Ensure default StockDatabase path equals config.get_database_path()
    db = StockDatabase()
    try:
        assert Path(db.db_path) == get_database_path()
    finally:
        db.close()


