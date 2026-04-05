"""Domain-specific exceptions for the quant engine."""


class DataSchemaError(ValueError):
    """Raised when time-series or tabular inputs violate schema (tz, monotonicity, keys)."""


class LeakageError(ValueError):
    """Raised when a pipeline would leak future information into the past."""


class ConfigError(ValueError):
    """Raised when configuration or environment is invalid."""


class ImpliedVolatilityError(ValueError):
    """Raised when implied volatility cannot be found (price out of model range or no bracket)."""
