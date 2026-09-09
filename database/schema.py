"""
database.schema: Relational database schema definitions for the trading system.
Supports SQLite storage for accounts, positions, trades, signals, and audit events.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT UNIQUE NOT NULL,
    total_capital REAL NOT NULL,
    available_capital REAL NOT NULL,
    invested_capital REAL NOT NULL DEFAULT 0.0,
    starting_capital REAL NOT NULL,
    total_net_pnl REAL NOT NULL DEFAULT 0.0,
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0.0,
    peak_capital REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL NOT NULL,
    invested REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    confidence REAL NOT NULL,
    predicted_return REAL NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    company TEXT,
    action TEXT NOT NULL, -- BUY, SELL
    qty REAL NOT NULL,
    price REAL NOT NULL,
    invested REAL NOT NULL,
    gross_return REAL DEFAULT 0.0,
    gross_profit REAL DEFAULT 0.0,
    total_cost REAL DEFAULT 0.0,
    brokerage REAL DEFAULT 0.0,
    stt REAL DEFAULT 0.0,
    gst REAL DEFAULT 0.0,
    tax REAL DEFAULT 0.0,
    net_profit REAL DEFAULT 0.0,
    net_return_pct REAL DEFAULT 0.0,
    capital_after REAL,
    hold_duration TEXT,
    reason TEXT,
    timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL, -- BUY, SELL, SKIP
    probability REAL NOT NULL,
    expected_return REAL NOT NULL,
    regime TEXT,
    price REAL NOT NULL,
    reason TEXT,
    timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL, -- DIAGNOSTIC, ERROR, WARNING, INFO, TRADE, RETRAIN
    source TEXT NOT NULL,
    message TEXT NOT NULL,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trading_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_key TEXT UNIQUE NOT NULL,
    rule_name TEXT NOT NULL,
    rule_type TEXT NOT NULL, -- RISK, EXECUTION, TIMING, TAX
    value REAL NOT NULL,
    description TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
