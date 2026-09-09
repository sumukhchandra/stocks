"""
Broker Interface: Abstract base class decoupling execution from broker-specific APIs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BrokerInterface(ABC):
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit order to broker and return order details."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Any]:
        """Fetch active positions held at broker."""
        pass

    @abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """Fetch available margin/cash balance."""
        pass
