import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class INRAccountingEngine:
    """
    INR ledger for Indian stock-market simulations.
    Tracks gross/net PnL, reserved capital, and tax liability.
    """
    def __init__(self, initial_balance_inr=800000, ledger_path='logs/live_metrics/inr_ledger.json', reset=False):
        self.ledger_path = ledger_path
        self.usd_inr_rate = 83.5 # Example fixed rate, can be dynamic

        if os.path.exists(self.ledger_path) and not reset:
            self.load_ledger()
        else:
            self.state = {
                'initial_capital_inr': initial_balance_inr,
                'current_liquid_capital_inr': initial_balance_inr,
                'locked_tds_inr': 0.0,
                'tax_liability_inr': 0.0,
                'total_gross_pnl_inr': 0.0,
                'total_net_pnl_inr': 0.0,
                'trade_count': 0,
                'last_updated': datetime.now().isoformat()
            }
            self.save_ledger()


    def record_trade(self, profit_usd, volume_usd):
        """
        Updates the ledger based on a completed trade.
        """
        profit_inr = profit_usd * self.usd_inr_rate
        volume_inr = volume_usd * self.usd_inr_rate
        
        # 1. Optional capital reserve on exit value. Equity workflow defaults this to zero.
        tds_deduction = volume_inr * 0.0
        
        # 2. Example gain tax on profitable exits.
        tax_on_trade = max(0, profit_inr * 0.20)
        
        # Update State
        self.state['total_gross_pnl_inr'] += profit_inr
        self.state['locked_tds_inr'] += tds_deduction
        self.state['tax_liability_inr'] += tax_on_trade
        self.state['current_liquid_capital_inr'] += (profit_inr - tds_deduction)
        
        # Net PnL = Gross - estimated tax liability.
        self.state['total_net_pnl_inr'] = self.state['total_gross_pnl_inr'] - self.state['tax_liability_inr']
        
        self.state['trade_count'] += 1
        self.state['last_updated'] = datetime.now().isoformat()
        
        self.save_ledger()
        return self.get_summary()

    def get_summary(self):
        summary = self.state.copy()
        summary['roi_pct'] = (summary['total_net_pnl_inr'] / summary['initial_capital_inr']) * 100
        summary['capital_locked_pct'] = (summary['locked_tds_inr'] / summary['initial_capital_inr']) * 100
        return summary

    def save_ledger(self):
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        with open(self.ledger_path, 'w') as f:
            json.dump(self.state, f, indent=4)

    def load_ledger(self):
        with open(self.ledger_path, 'r') as f:
            self.state = json.load(f)

if __name__ == "__main__":
    engine = INRAccountingEngine(initial_balance_inr=100000)
    # Simulate a trade with $100 profit on $1000 volume
    summary = engine.record_trade(100, 1000)
    print(f"Post-Trade Summary: {json.dumps(summary, indent=2)}")
