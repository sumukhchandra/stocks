"""
database.db_manager: Database manager layer providing thread-safe SQLite operations.
"""
import os
import sqlite3
import json
from datetime import datetime
from database.schema import SCHEMA_SQL

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_system.db")

class DatabaseManager:
    """Handles all persistence, relational queries, and event audits."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.initialize_db()

    def get_connection(self):
        """Returns a connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_db(self):
        """Execute schema setup if tables do not exist."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    # -- Account Operations ---------------------------------------------------

    def sync_account(self, account_name="main", total_capital=100000.0, available_capital=100000.0,
                     starting_capital=100000.0, total_net_pnl=0.0, total_trades=0,
                     winning_trades=0, losing_trades=0, peak_capital=100000.0):
        """Insert or update account equity snapshot."""
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        invested = max(0.0, total_capital - available_capital)
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO accounts (
                    account_name, total_capital, available_capital, invested_capital,
                    starting_capital, total_net_pnl, total_trades, winning_trades,
                    losing_trades, win_rate, peak_capital, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(account_name) DO UPDATE SET
                    total_capital=excluded.total_capital,
                    available_capital=excluded.available_capital,
                    invested_capital=excluded.invested_capital,
                    total_net_pnl=excluded.total_net_pnl,
                    total_trades=excluded.total_trades,
                    winning_trades=excluded.winning_trades,
                    losing_trades=excluded.losing_trades,
                    win_rate=excluded.win_rate,
                    peak_capital=MAX(accounts.peak_capital, excluded.total_capital),
                    updated_at=CURRENT_TIMESTAMP
            """, (account_name, total_capital, available_capital, invested,
                  starting_capital, total_net_pnl, total_trades, winning_trades,
                  losing_trades, win_rate, peak_capital))
            conn.commit()

    def get_account(self, account_name="main"):
        """Fetch current account record."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM accounts WHERE account_name = ?", (account_name,))
            row = cur.fetchone()
            return dict(row) if row else None

    # -- Trade & Signal Operations --------------------------------------------

    def record_trade(self, trade_dict):
        """Insert trade record."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO trades (
                    symbol, company, action, qty, price, invested, gross_return,
                    gross_profit, total_cost, brokerage, stt, gst, tax,
                    net_profit, net_return_pct, capital_after, hold_duration,
                    reason, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_dict.get("symbol"),
                trade_dict.get("company", ""),
                trade_dict.get("action"),
                trade_dict.get("qty", 0.0),
                trade_dict.get("price", 0.0),
                trade_dict.get("invested", 0.0),
                trade_dict.get("gross_return", 0.0),
                trade_dict.get("gross_profit", 0.0),
                trade_dict.get("total_cost", 0.0),
                trade_dict.get("brokerage", 0.0),
                trade_dict.get("stt", 0.0),
                trade_dict.get("gst", 0.0),
                trade_dict.get("tax", 0.0),
                trade_dict.get("net_profit", 0.0),
                trade_dict.get("net_return_pct", 0.0),
                trade_dict.get("capital_after", 0.0),
                trade_dict.get("hold_duration", ""),
                trade_dict.get("reason", ""),
                trade_dict.get("timestamp", datetime.now().isoformat())
            ))
            conn.commit()

    def record_signal(self, symbol, action, probability, expected_return, price, regime="", reason="", timestamp=None):
        """Insert model prediction signal."""
        ts = timestamp or datetime.now().isoformat()
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO signals (symbol, action, probability, expected_return, regime, price, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, probability, expected_return, regime, price, reason, ts))
            conn.commit()

    def get_recent_trades(self, limit=50):
        """Retrieve recent trade history."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # -- System Events --------------------------------------------------------

    def log_event(self, event_type, source, message, details=None):
        """Record a diagnostic or operational event."""
        detail_str = json.dumps(details) if isinstance(details, (dict, list)) else (details or "")
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO system_events (event_type, source, message, details)
                VALUES (?, ?, ?, ?)
            """, (event_type, source, message, detail_str))
            conn.commit()

    def get_recent_events(self, limit=50):
        """Fetch system events."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM system_events ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]
