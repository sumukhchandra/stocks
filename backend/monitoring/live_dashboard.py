import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime

class LiveDashboard:
    """
    Console-based monitoring dashboard for the trading system.
    """
    def __init__(self, metrics_dir='logs/live_metrics'):
        self.metrics_dir = metrics_dir
        self.metrics_path = os.path.join(self.metrics_dir, 'daily_metrics.csv')
        self.trades_path = os.path.join(self.metrics_dir, 'trades.csv')
        self.ledger_path = os.path.join(self.metrics_dir, 'inr_ledger.json')
        self.last_pnl = 0.0
        self.start_time = datetime.now()

    def get_live_state(self):
        """Reads logs and calculates real-time metrics."""
        state = {
            'pnl': 0.0,
            'net_pnl': 0.0,
            'regime': 'UNKNOWN',
            'exposure': 0.0,
            'sharpe': 0.0,
            'tax_efficiency': 0.0,
            'locked_tds': 0.0,
            'liquid_capital': 10000.0,
            'alerts': []
        }
        
        # 1. Load INR Ledger (Primary Source)
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, 'r') as f:
                    ledger = json.load(f)
                    state['pnl'] = ledger['total_gross_pnl_inr'] / 83.5
                    state['net_pnl'] = ledger['total_net_pnl_inr'] / 83.5
                    state['locked_tds'] = ledger['locked_tds_inr'] / 83.5
                    state['liquid_capital'] = ledger['current_liquid_capital_inr'] / 83.5
                    state['tax_efficiency'] = (ledger['total_net_pnl_inr'] / ledger['total_gross_pnl_inr']) if ledger['total_gross_pnl_inr'] > 0 else 0
            except: pass
        
        # 2. Load Trades for Sharpe
        if os.path.exists(self.trades_path):
            try:
                trades = pd.read_csv(self.trades_path)
                if not trades.empty and len(trades) > 5:
                    returns = trades['profit'] / 10000.0
                    state['sharpe'] = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
            except: pass
                
        if os.path.exists(self.metrics_path):
            try:
                metrics = pd.read_csv(self.metrics_path)
                if not metrics.empty:
                    latest = metrics.iloc[-1]
                    state['regime'] = latest.get('regime', 'UNKNOWN')
                    state['calibration'] = latest.get('confidence_score', 0.5)
            except: pass
                
        return state

    def update(self):
        """Updates the console with the latest live state."""
        state = self.get_live_state()
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*60)
        print(f" SYSTEM MONITORING DASHBOARD | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
        print(f" Strategy: Net-Edge Survival (Swing) | Uptime: {datetime.now() - self.start_time} ")
        print("="*60)
        
        # PnL Section
        net_pnl = state.get('net_pnl', 0.0)
        color = "\033[92m" if net_pnl >= 0 else "\033[91m"
        reset = "\033[0m"
        print(f" Net PnL (Post-Tax): {color}${net_pnl:,.2f}{reset}")
        print(f" Gross PnL: ${state.get('pnl', 0.0):,.2f}")
        print(f" Tax Efficiency: {state.get('tax_efficiency', 0.0)*100:.1f}%")
        
        # Capital Section
        print("-" * 60)
        print(f" Liquid Capital: ${state.get('liquid_capital', 0.0):,.2f}")
        print(f" Reserved Capital: \033[93m${state.get('locked_tds', 0.0):,.2f}\033[0m")
        
        # Risk & Regime
        print("-" * 60)
        print(f" Current Regime: {state.get('regime', 'UNKNOWN').upper()}")
        print(f" Risk Exposure: ${state.get('exposure', 0.0):,.2f}")
        
        # Performance Metrics
        print("-" * 60)
        print(f" Est. Sharpe Ratio: {state.get('sharpe', 0.0):.2f}")
        print(f" Calibration: {state.get('calibration', 0.0):.4f}")
        
        # Recent Alerts
        print("-" * 60)
        print(" RECENT ALERTS:")
        alerts = state.get('alerts', ["SYSTEM HEALTHY"])
        for alert in alerts[-3:]:
            print(f" [!] {alert}")
            
        print("="*60)

if __name__ == "__main__":
    dash = LiveDashboard()
    try:
        while True:
            dash.update()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
