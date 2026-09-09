"""
database package: Relational models, rules engines, sector mappings, and configurations.
"""
from database.stocks_config import (
    STOCK_SYMBOLS,
    STOCK_UNIVERSE,
    DEFAULT_CAPITAL,
    MIN_NET_PROFIT_PCT,
    MAX_GROSS_LOSS_PCT,
    POSITION_SIZING,
    SCAN_INTERVAL_SECONDS,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
)
from database.rules import TradingRulesEngine
from database.relations import STOCK_METADATA, SECTORS, get_stock_sector, get_sector_stocks, get_correlated_peers
from database.db_manager import DatabaseManager

__all__ = [
    'STOCK_SYMBOLS',
    'STOCK_UNIVERSE',
    'DEFAULT_CAPITAL',
    'MIN_NET_PROFIT_PCT',
    'MAX_GROSS_LOSS_PCT',
    'POSITION_SIZING',
    'TradingRulesEngine',
    'DatabaseManager',
    'STOCK_METADATA',
    'SECTORS',
    'get_stock_sector',
    'get_sector_stocks',
    'get_correlated_peers',
]
