"""
Portfolio Manager: Manages capital, active positions, realized/unrealized P&L, and equity curve.
"""

from typing import Dict, Any, List
from datetime import datetime


class PortfolioManager:
    def __init__(self, starting_capital: float = 100000.0):
        self.starting_capital = starting_capital
        self.available_capital = starting_capital
        self.total_capital = starting_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []

    def get_state(self) -> Dict[str, Any]:
        invested = sum(p.get("invested", 0.0) for p in self.positions.values())
        return {
            "starting_capital": self.starting_capital,
            "total_capital": round(self.available_capital + invested, 2),
            "available_capital": round(self.available_capital, 2),
            "invested_capital": round(invested, 2),
            "positions": self.positions,
            "open_positions_count": len(self.positions),
            "total_trades": len(self.trade_history),
            "net_pnl": round((self.available_capital + invested) - self.starting_capital, 2)
        }

    def record_entry(self, symbol: str, qty: float, price: float, invested: float, model_version: str):
        self.positions[symbol] = {
            "symbol": symbol,
            "quantity": qty,
            "entry_price": price,
            "invested": invested,
            "entry_time": datetime.now().isoformat(),
            "model_version": model_version
        }
        self.available_capital -= invested

    def record_exit(self, symbol: str, exit_price: float, net_profit: float):
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            self.available_capital += (pos["invested"] + net_profit)
            self.total_capital = self.available_capital + sum(p["invested"] for p in self.positions.values())
            trade_rec = {
                "symbol": symbol,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "quantity": pos["quantity"],
                "invested": pos["invested"],
                "net_profit": net_profit,
                "model_version": pos.get("model_version", "unknown"),
                "closed_at": datetime.now().isoformat()
            }
            self.trade_history.append(trade_rec)
            return trade_rec
        return None
