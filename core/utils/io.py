from pathlib import Path
from typing import Optional, Union

import duckdb
import pandas as pd


def ensure_dirs(root: Path) -> None:
    """Ensure directory exists."""
    root.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to Parquet file."""
    ensure_dirs(path.parent)
    df.to_parquet(path, index=True)


def read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """
    Read Parquet file if it exists.
    
    Args:
        path: Path to Parquet file
        
    Returns:
        DataFrame or None if file doesn't exist
    """
    if not path.exists():
        return None
    return pd.read_parquet(path)


def get_last_date_from_parquet(path: Path) -> Optional[pd.Timestamp]:
    """
    Get the most recent date from a Parquet file.
    
    Args:
        path: Path to Parquet file
        
    Returns:
        Most recent date as Timestamp or None if file doesn't exist
    """
    df = read_parquet(path)
    if df is None or df.empty:
        return None
    
    # Handle both regular index and MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        # Assume first level is date
        return df.index.get_level_values(0).max()
    else:
        return df.index.max()


def get_existing_symbols_from_parquet(path: Path) -> list:
    """
    Get list of existing symbols from a Parquet file.
    
    For wide format (date × symbols): returns column names
    For long format (date, symbol): returns unique symbols from index
    
    Args:
        path: Path to Parquet file
        
    Returns:
        List of symbols
    """
    df = read_parquet(path)
    if df is None or df.empty:
        return []
    
    # Wide format: symbols are columns
    if not isinstance(df.index, pd.MultiIndex):
        return df.columns.tolist()
    
    # Long format: symbols in MultiIndex
    if 'symbol' in df.index.names:
        return df.index.get_level_values('symbol').unique().tolist()
    
    return []


def append_rows_to_parquet(path: Path, new_rows: pd.DataFrame) -> None:
    """
    Append new rows (dates) to existing Parquet file.
    
    Args:
        path: Path to Parquet file
        new_rows: DataFrame with new rows to append (same columns/structure)
    """
    existing = read_parquet(path)
    
    if existing is None or existing.empty:
        # No existing data, just write new rows
        write_parquet(new_rows, path)
        return
    
    # Concatenate and remove duplicates
    combined = pd.concat([existing, new_rows])
    
    # Drop duplicate index values (keep last/newest)
    if isinstance(combined.index, pd.MultiIndex):
        combined = combined[~combined.index.duplicated(keep='last')]
    else:
        combined = combined[~combined.index.duplicated(keep='last')]
    
    combined = combined.sort_index()
    write_parquet(combined, path)


def add_columns_to_parquet(path: Path, new_columns: Union[pd.DataFrame, pd.Series]) -> None:
    """
    Add new columns (symbols or factors) to existing Parquet file.
    
    Args:
        path: Path to Parquet file
        new_columns: DataFrame or Series with new columns to add
    """
    existing = read_parquet(path)
    
    if existing is None or existing.empty:
        # No existing data, just write new columns
        if isinstance(new_columns, pd.Series):
            new_columns = new_columns.to_frame()
        write_parquet(new_columns, path)
        return
    
    # Convert Series to DataFrame if needed
    if isinstance(new_columns, pd.Series):
        new_columns = new_columns.to_frame()
    
    # Merge/join based on index
    combined = existing.join(new_columns, how='outer')
    combined = combined.sort_index()
    
    write_parquet(combined, path)


def merge_parquet_files(path1: Path, path2: Path, output_path: Path, how: str = 'outer') -> None:
    """
    Merge two Parquet files on their index.
    
    Args:
        path1: Path to first Parquet file
        path2: Path to second Parquet file
        output_path: Path for output merged file
        how: Join type ('inner', 'outer', 'left', 'right')
    """
    df1 = read_parquet(path1)
    df2 = read_parquet(path2)
    
    if df1 is None and df2 is None:
        return
    
    if df1 is None:
        write_parquet(df2, output_path)
        return
    
    if df2 is None:
        write_parquet(df1, output_path)
        return
    
    merged = df1.join(df2, how=how)
    merged = merged.sort_index()
    write_parquet(merged, output_path)


def connect_duckdb(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB database."""
    ensure_dirs(db_path.parent)
    con = duckdb.connect(str(db_path))
    return con


def register_parquet(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    """Register Parquet file as DuckDB view."""
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")


