"""FMP (Financial Modeling Prep) data source: HTTP client, prices, fundamentals, market caps."""

from core.data.fmp.client import fmp_get
from core.data.fmp.fundamentals import fetch_quarterly_statement, parse_statement_rows
from core.data.fmp.market_caps import fetch_historical_market_cap, parse_market_cap_rows
from core.data.fmp.panel import update_panel_from_fmp
from core.data.fmp.prices import (
    fetch_dividend_adjusted_history,
    generate_date_chunks,
    parse_dividend_adjusted_rows,
)

__all__ = [
    "fmp_get",
    "fetch_dividend_adjusted_history",
    "fetch_historical_market_cap",
    "fetch_quarterly_statement",
    "generate_date_chunks",
    "parse_dividend_adjusted_rows",
    "parse_market_cap_rows",
    "parse_statement_rows",
    "update_panel_from_fmp",
]
