"""
Order Manager: Coordinates order creation, life cycle tracking, and broker routing.
"""

from typing import Dict, Any, Optional
from .broker_interface import BrokerInterface
from .brokers.paper import PaperBroker


class OrderManager:
    def __init__(self, broker: Optional[BrokerInterface] = None):
        self.broker = broker or PaperBroker()
        self.order_history = []

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        model_version: str,
        order_type: str = "MARKET"
    ) -> Dict[str, Any]:
        order = self.broker.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            tag=model_version
        )
        order["model_version"] = model_version
        self.order_history.append(order)
        return order

    def get_order_history(self):
        return self.order_history
