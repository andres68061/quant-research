"""
ML Results Caching Utilities

Provides functions to save and load pre-computed ML results for Quick View mode.
"""
import pickle
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Cache directory relative to project root
CACHE_DIR = Path(__file__).parent.parent.parent / "outputs" / "ml_results"


def save_ml_results(
    symbol: str,
    freq: str,
    model_type: str,
    results: Dict,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Save ML results to cache.
    
    Args:
        symbol: Commodity symbol (e.g., "GLD", "SLV")
        freq: Data frequency ("Daily", "Weekly", "Monthly")
        model_type: "xgboost", "lstm", or "compare"
        results: Results dictionary from walk-forward validation
        metadata: Optional metadata (params, date, etc.)
        
    Returns:
        Path to saved file
        
    Example:
        >>> results = run_walk_forward_validation(...)
        >>> path = save_ml_results("GLD", "Daily", "xgboost", results)
    """
    # Create cache directory if needed
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Construct filename
    filename = f"{symbol}_{freq}_{model_type}.pkl"
    path = CACHE_DIR / filename
    
    # Package data
    cache_data = {
        'results': results,
        'metadata': metadata or {},
        'symbol': symbol,
        'freq': freq,
        'model_type': model_type,
    }
    
    # Save
    with open(path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    logger.info(f"Saved ML results to {path}")
    return path


def load_ml_results(
    symbol: str,
    freq: str,
    model_type: str,
) -> Optional[Dict]:
    """
    Load ML results from cache.
    
    Args:
        symbol: Commodity symbol (e.g., "GLD", "SLV")
        freq: Data frequency ("Daily", "Weekly", "Monthly")
        model_type: "xgboost", "lstm", or "compare"
        
    Returns:
        Dictionary with 'results' and 'metadata' keys, or None if not found
        
    Example:
        >>> cached = load_ml_results("GLD", "Daily", "xgboost")
        >>> if cached:
        >>>     results = cached['results']
        >>>     metadata = cached['metadata']
    """
    filename = f"{symbol}_{freq}_{model_type}.pkl"
    path = CACHE_DIR / filename
    
    if not path.exists():
        logger.debug(f"No cached results found: {path}")
        return None
    
    try:
        with open(path, 'rb') as f:
            cache_data = pickle.load(f)
        
        logger.info(f"Loaded ML results from {path}")
        return cache_data
    
    except Exception as e:
        logger.error(f"Failed to load cache from {path}: {e}")
        return None


def list_cached_results() -> list:
    """
    List all cached ML results.
    
    Returns:
        List of tuples (symbol, freq, model_type, path)
        
    Example:
        >>> for symbol, freq, model_type, path in list_cached_results():
        >>>     print(f"{symbol} {freq} {model_type}: {path.stat().st_mtime}")
    """
    if not CACHE_DIR.exists():
        return []
    
    results = []
    for path in CACHE_DIR.glob("*.pkl"):
        try:
            # Parse filename: symbol_freq_modeltype.pkl
            parts = path.stem.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                freq = parts[1]
                model_type = '_'.join(parts[2:])  # Handle "compare_both"
                results.append((symbol, freq, model_type, path))
        except Exception as e:
            logger.warning(f"Failed to parse cache filename {path}: {e}")
    
    return results


def clear_cache(symbol: Optional[str] = None, freq: Optional[str] = None):
    """
    Clear cached results.
    
    Args:
        symbol: Optional symbol filter (e.g., "GLD")
        freq: Optional frequency filter (e.g., "Daily")
        
    If no filters provided, clears all cache.
    
    Example:
        >>> clear_cache(symbol="GLD")  # Clear all GLD results
        >>> clear_cache(freq="Weekly")  # Clear all Weekly results
        >>> clear_cache()  # Clear everything
    """
    if not CACHE_DIR.exists():
        return
    
    pattern = f"{symbol or '*'}_{freq or '*}_*.pkl"
    count = 0
    
    for path in CACHE_DIR.glob(pattern):
        try:
            path.unlink()
            count += 1
            logger.info(f"Deleted cache: {path}")
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
    
    logger.info(f"Cleared {count} cached result(s)")


def get_cache_info(symbol: str, freq: str, model_type: str) -> Optional[Dict]:
    """
    Get metadata about cached results without loading full data.
    
    Args:
        symbol: Commodity symbol
        freq: Data frequency
        model_type: Model type
        
    Returns:
        Dictionary with cache metadata (size, modified time, etc.) or None
    """
    filename = f"{symbol}_{freq}_{model_type}.pkl"
    path = CACHE_DIR / filename
    
    if not path.exists():
        return None
    
    stat = path.stat()
    return {
        'path': str(path),
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': stat.st_mtime,
        'exists': True,
    }
