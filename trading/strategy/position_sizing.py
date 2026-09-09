"""
Position Sizing & Capital Allocation Rules.
"""

from typing import Dict, Any


class PositionSizer:
    @staticmethod
    def calculate_allocation(available_capital: float, confidence: float, min_confidence: float = 0.50) -> float:
        """
        Dynamic position sizing based on model probability/confidence.
        Tiered allocation:
        - Conf > 80%: 20% of available cash
        - Conf > 70%: 15% of available cash
        - Conf > 60%: 10% of available cash
        - Conf >= min_confidence: 8% of available cash
        - Otherwise: 0
        """
        if available_capital < 1000.0:
            return 0.0

        if confidence > 0.80:
            pct = 0.20
        elif confidence > 0.70:
            pct = 0.15
        elif confidence > 0.60:
            pct = 0.10
        elif confidence >= min_confidence:
            pct = 0.08
        else:
            pct = 0.0

        return round(available_capital * pct, 2)
