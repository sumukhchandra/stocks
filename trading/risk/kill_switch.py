"""
Kill Switch & Circuit Breakers: Halts trading upon catastrophic events or operator intervention.
"""

from typing import Dict, Any
from datetime import datetime


class KillSwitch:
    def __init__(self, max_daily_drawdown_pct: float = 3.0):
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.is_active = False
        self.triggered_at = None
        self.trigger_reason = None

    def trigger(self, reason: str):
        self.is_active = True
        self.triggered_at = datetime.now().isoformat()
        self.trigger_reason = reason

    def reset(self):
        self.is_active = False
        self.triggered_at = None
        self.trigger_reason = None

    def check_drawdown(self, starting_capital: float, current_capital: float) -> bool:
        if starting_capital <= 0:
            return False

        drawdown_pct = ((starting_capital - current_capital) / starting_capital) * 100.0
        if drawdown_pct >= self.max_daily_drawdown_pct:
            self.trigger(f"Max daily drawdown breached: {drawdown_pct:.2f}% >= {self.max_daily_drawdown_pct:.2f}%")
            return True
        return False
