"""
Configuration settings for the quant project.

This module contains all configuration settings including API keys,
database paths, and other project settings.
"""

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from a local .env file if present
load_dotenv()

# Database settings
DATABASE_PATH = PROJECT_ROOT / "data" / "stock_data.db"
DATABASE_BACKUP_PATH = PROJECT_ROOT / "data" / "backups"

# API Keys (store these securely in environment variables)
# Core market data providers
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
FMP_API_KEY = os.getenv('FMP_API_KEY') or os.getenv('FINANCIAL_MODELING_PREP_API_KEY')

# AI / LLM providers
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Macro / Economics
FRED_API_KEY = os.getenv('FRED_API_KEY')
BEA_API_KEY = os.getenv('BEA_API_KEY')

# Reddit API (script/app auth)
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
REDDIT_USERNAME = os.getenv('REDDIT_USERNAME')
REDDIT_PASSWORD = os.getenv('REDDIT_PASSWORD')

if not FINNHUB_API_KEY:
    print("⚠️  WARNING: FINNHUB_API_KEY not found in environment variables")
    print("   Set it with: export FINNHUB_API_KEY='your_api_key_here'")
    FINNHUB_API_KEY = None

# Data sources configuration
DATA_SOURCES = {
    'yfinance': {
        'enabled': True,
        'rate_limit': 100,  # requests per minute
        'timeout': 30
    },
    'finnhub': {
        'enabled': bool(FINNHUB_API_KEY),
        'rate_limit': 60,  # requests per minute (free tier)
        'timeout': 30
    }
}

# Default data fetching settings
DEFAULT_PERIOD = "1y"
DEFAULT_SOURCE = "auto"
DEFAULT_RESOLUTION = "1d"

# Database retention settings
DATA_RETENTION_DAYS = 3650  # 10 years
CLEANUP_INTERVAL_DAYS = 30  # Clean up every 30 days

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "logs" / "quant.log"

# Machine learning settings
ML_MODELS_PATH = PROJECT_ROOT / "models"
ML_DATA_PATH = PROJECT_ROOT / "data" / "ml"
ML_RESULTS_PATH = PROJECT_ROOT / "results"

# Visualization settings
PLOT_STYLE = "seaborn-v0_8"
FIGURE_SIZE = (12, 8)
DPI = 300

# Technical analysis settings
TECHNICAL_INDICATORS = {
    'sma_periods': [20, 50, 200],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2
}

# Risk management settings
RISK_SETTINGS = {
    'max_position_size': 0.1,  # 10% of portfolio
    'stop_loss_pct': 0.05,     # 5% stop loss
    'take_profit_pct': 0.15,   # 15% take profit
    'max_drawdown': 0.20       # 20% max drawdown
}

# Performance metrics
PERFORMANCE_METRICS = [
    'total_return',
    'annualized_return',
    'volatility',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'information_ratio'
]

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "backups",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "data" / "ml",
        PROJECT_ROOT / "results"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
create_directories()

# Validation functions
def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that all required API keys are available.
    
    Returns:
        Dict[str, bool]: Dictionary with API key validation status
    """
    validation = {}

    # Market data providers
    validation['finnhub'] = bool(FINNHUB_API_KEY and len(FINNHUB_API_KEY) > 0)
    validation['alphavantage'] = bool(ALPHAVANTAGE_API_KEY)
    validation['fmp'] = bool(FMP_API_KEY)

    # AI / LLM
    validation['openai'] = bool(OPENAI_API_KEY)

    # Macro / Economics
    validation['fred'] = bool(FRED_API_KEY)
    validation['bea'] = bool(BEA_API_KEY)

    # Reddit bundle: require all fields
    validation['reddit'] = all([
        REDDIT_CLIENT_ID,
        REDDIT_CLIENT_SECRET,
        REDDIT_USER_AGENT,
        REDDIT_USERNAME,
        REDDIT_PASSWORD
    ])

    return validation

def get_database_path() -> Path:
    """
    Get the database path, creating the directory if needed.
    
    Returns:
        Path: Database file path
    """
    db_dir = DATABASE_PATH.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    return DATABASE_PATH

def get_log_file_path() -> Path:
    """
    Get the log file path, creating the directory if needed.
    
    Returns:
        Path: Log file path
    """
    log_dir = LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    return LOG_FILE

# Environment-specific settings
class DevelopmentConfig:
    """Development environment configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    DATABASE_PATH = PROJECT_ROOT / "data" / "stock_data_dev.db"

class ProductionConfig:
    """Production environment configuration."""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    DATABASE_PATH = PROJECT_ROOT / "data" / "stock_data_prod.db"

class TestingConfig:
    """Testing environment configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    DATABASE_PATH = PROJECT_ROOT / "data" / "stock_data_test.db"

# Configuration mapping
configs = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

def get_config(environment: str = 'development'):
    """
    Get configuration for the specified environment.
    
    Args:
        environment (str): Environment name ('development', 'production', 'testing')
        
    Returns:
        Config class: Configuration object
    """
    return configs.get(environment, DevelopmentConfig)

# Current environment (can be set via environment variable)
CURRENT_ENV = os.getenv('QUANT_ENV', 'development')
CURRENT_CONFIG = get_config(CURRENT_ENV)

if __name__ == "__main__":
    # Test configuration
    print("🔧 Quant Project Configuration")
    print("=" * 40)
    
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Database Path: {get_database_path()}")
    print(f"Log File: {get_log_file_path()}")
    
    print(f"\nAPI Keys:")
    validation = validate_api_keys()
    for api, valid in validation.items():
        status = "✅" if valid else "❌"
        print(f"  {api}: {status}")
    
    print(f"\nData Sources:")
    for source, config in DATA_SOURCES.items():
        status = "✅" if config['enabled'] else "❌"
        print(f"  {source}: {status}")
    
    print(f"\nCurrent Environment: {CURRENT_ENV}")
    print(f"Debug Mode: {CURRENT_CONFIG.DEBUG}")
    print(f"Log Level: {CURRENT_CONFIG.LOG_LEVEL}")
    
    print("\n✅ Configuration loaded successfully!")
