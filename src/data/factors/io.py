from pathlib import Path

import duckdb
import pandas as pd

from src.utils.io import (
    connect_duckdb as _connect_duckdb,
)

# Re-export shared IO helpers from utils to centralize functionality
from src.utils.io import (
    ensure_dirs as _ensure_dirs,
)
from src.utils.io import (
    register_parquet as _register_parquet,
)
from src.utils.io import (
    write_parquet as _write_parquet,
)


def ensure_dirs(root: Path) -> None:
    _ensure_dirs(root)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    _write_parquet(df, path)


def connect_duckdb(db_path: Path) -> duckdb.DuckDBPyConnection:
    return _connect_duckdb(db_path)


def register_parquet(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    _register_parquet(con, name, path)


