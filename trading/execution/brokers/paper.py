"""
Paper Broker: High-fidelity simulation broker with real slippage, Zerodha fees, and 25% tax.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from ..broker_interface import BrokerInterface
from backend.risk.india_tax_engine import IndiaTaxEngine


class PaperBroker(BrokerInterface):
    def __init__(self, initial_capital: float = 100000.0, slippage_bps: float = 2.0):
        self.total_capital = initial_capital
        self.available_capital = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.slippage_bps = slippage_bps  # basis points (e.g. 2 bps = 0.02%)
        self.tax_engine = IndiaTaxEngine()

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        order_id = f"paper_{uuid.uuid4().hex[:10]}"
        slippage_mult = (1.0 + (self.slippage_bps / 10000.0)) if side == "BUY" else (1.0 - (self.slippage_bps / 10000.0))
        exec_price = round(price * slippage_mult, 2)
        turnover = exec_price * quantity

        order_record = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "requested_price": price,
            "executed_price": exec_price,
            "turnover": turnover,
            "status": "FILLED",
            "tag": tag,
            "timestamp": datetime.now().isoformat()
        }

        if side == "BUY":
            self.available_capital -= turnover
            self.positions[symbol] = {
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": exec_price,
                "invested": turnover,
                "entry_time": datetime.now().isoformat(),
                "tag": tag
            }
        elif side == "SELL":
            if symbol in self.positions:
                pos = self.positions.pop(symbol)
                gross_return = (exec_price - pos["entry_price"]) / (pos["entry_price"] + 1e-8)
                cost_bd = self.tax_engine.calculate_total_cost(pos["invested"], gross_return)
                net_profit = cost_bd["net_profit"]
                self.available_capital += (pos["invested"] + net_profit)
                self.total_capital += net_profit
                order_record["net_profit"] = net_profit
                order_record["costs"] = cost_bd

        self.orders[order_id] = order_record
        return order_record

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders and self.orders[order_id]["status"] == "PENDING":
            self.orders[order_id]["status"] = "CANCELLED"
            return True
        return False

    def get_positions(self) -> Dict[str, Any]:
        return self.positions

    def get_account_balance(self) -> Dict[str, float]:
        invested = sum(p["invested"] for p in self.positions.values())
        return {
            "total_capital": round(self.available_capital + invested, 2),
            "available_capital": round(self.available_capital, 2),
            "invested_capital": round(invested, 2)
        }
