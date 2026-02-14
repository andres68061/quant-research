# ML Results Cache

Auto-generated cached results for Quick View mode.

## Files

Format: `{symbol}_{frequency}_{model_type}.pkl`

Examples:
- `GLD_Daily_xgboost.pkl`
- `SLV_Weekly_lstm.pkl`
- `GDX_Monthly_compare.pkl`

## Usage

Results are automatically cached when training completes in Interactive mode.

Access cached results via Quick View mode in the Streamlit UI.

## Clear Cache

```python
from src.utils.ml_cache import clear_cache

# Clear specific commodity
clear_cache(symbol="GLD")

# Clear specific frequency
clear_cache(freq="Daily")

# Clear everything
clear_cache()
```
