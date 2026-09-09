import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class GlobalRiskController:
    """
    Enforces strict safety limits and handles emergency shutdowns.
    """
    def __init__(self, config=None):
        self.limits = config or {
            'max_daily_drawdown_pct': 0.02, # 2% daily limit
            'max_total_exposure_usd': 5000.0,
            'max_position_size_pct': 0.1,  # Max 10% per trade
            'volatility_cutoff': 0.05,     # Stop if 5m vol > 5%
            'crisis_mode_active': False
        }
        self.state_path = 'logs/live_metrics/risk_state.json'
        self.start_balance = 10000.0

    def check_safety(self, current_balance, open_positions, market_vol):
        """
        Returns (is_safe, reason)
        """
        # 1. Drawdown Check
        daily_pnl_pct = (current_balance - self.start_balance) / self.start_balance
        if daily_pnl_pct <= -self.limits['max_daily_drawdown_pct']:
            return False, "DAILY_DRAWDOWN_LIMIT_REACHED"

        # 2. Exposure Check
        total_exposure = sum([p['invested'] for p in open_positions.values()])
        if total_exposure > self.limits['max_total_exposure_usd']:
            return False, "MAX_EXPOSURE_EXCEEDED"

        # 3. Volatility Cutoff
        if market_vol > self.limits['volatility_cutoff']:
            return False, "EXTREME_VOLATILITY_SHUTDOWN"

        if self.limits['crisis_mode_active']:
            return False, "CRISIS_MODE_MANUAL_SHUTDOWN"

        return True, "SAFE"

    def update_limits(self, **kwargs):
        self.limits.update(kwargs)
        self.save_state()

    def save_state(self):
        with open(self.state_path, 'w') as f:
            json.dump(self.limits, f)

if __name__ == "__main__":
    controller = GlobalRiskController()
    is_safe, reason = controller.check_safety(9500, {}, 0.01)
    print(f"Safety Check: {is_safe}, Reason: {reason}")
