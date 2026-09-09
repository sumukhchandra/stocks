"""
Simulation script to backtest the NSE Auto-Trader on the last full trading session.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta

# Fix Windows console unicode printing for Rupee symbol
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.nse_auto_trader import NSEAutoTrader
from data.fetch_stock_data import build_stock_dataset
from database.stocks_config import STOCK_SYMBOLS

# Custom state files for simulation
SIM_STATE_FILE = os.path.join(os.path.dirname(__file__), "logs", "sim_trader_state.json")
SIM_TRADE_LOG = os.path.join(os.path.dirname(__file__), "logs", "sim_trade_history.jsonl")

class SimulatorTrader(NSEAutoTrader):
    """Subclass of NSEAutoTrader to feed historical data sequentially."""
    
    def __init__(self, capital, full_data, current_timestamp):
        self.full_data = full_data
        self.current_timestamp = current_timestamp
        
        # Override file paths
        global STATE_FILE, TRADE_LOG_FILE
        import nse_auto_trader
        self.original_state_file = nse_auto_trader.STATE_FILE
        self.original_trade_log = nse_auto_trader.TRADE_LOG_FILE
        nse_auto_trader.STATE_FILE = SIM_STATE_FILE
        nse_auto_trader.TRADE_LOG_FILE = SIM_TRADE_LOG
        
        # Clean old sim state
        if os.path.exists(SIM_STATE_FILE):
            os.remove(SIM_STATE_FILE)
        if os.path.exists(SIM_TRADE_LOG):
            os.remove(SIM_TRADE_LOG)
            
        super().__init__(capital=capital)
        
    def _fetch_live_data(self, symbols=None):
        """Override to return data only up to current_timestamp."""
        mask = self.full_data['timestamp'] <= self.current_timestamp
        return self.full_data[mask].copy()
        
    def cleanup(self):
        """Restore original paths."""
        import nse_auto_trader
        nse_auto_trader.STATE_FILE = self.original_state_file
        nse_auto_trader.TRADE_LOG_FILE = self.original_trade_log

def run_simulation():
    print("Fetching historical data for simulation...")
    # Fetch last 5 days to ensure we have a full trading day and warmup data
    df = build_stock_dataset(period="5d", interval="5m")
    
    if df.empty:
        print("No data fetched.")
        return
        
    # Get unique dates
    df['date_only'] = df['timestamp'].dt.date
    dates = sorted(df['date_only'].unique())
    
    if len(dates) < 2:
        print("Not enough days of data to simulate.")
        return
        
    # Target the last full trading day
    target_date = dates[-2]  # Yesterday or last full session
    print(f"\nSimulating trading session for: {target_date}")
    
    # Filter timestamps for the target date
    day_data = df[df['date_only'] == target_date]
    timestamps = sorted(day_data['timestamp'].unique())
    
    print(f"Total 5m intervals in session: {len(timestamps)}")
    print("Starting simulation with Rs.100,000 capital...\n")
    print("-" * 60)
    
    trader = SimulatorTrader(capital=100000, full_data=df, current_timestamp=timestamps[0])
    
    for ts in timestamps:
        trader.current_timestamp = ts
        # Suppress prints for clean output, unless a trade happens
        import sys, io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        trader.run_single_cycle()
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        if "BUY" in output or "SELL" in output:
            print(f"[{ts.strftime('%H:%M')}] " + output.strip())
            
    # Force close any open positions at end of day
    print("\n" + "-" * 60)
    print("End of day reached. Closing open positions...")
    trader.current_timestamp = timestamps[-1]
    feature_df = trader._fetch_live_data()
    
    for symbol in list(trader.state["positions"].keys()):
        symbol_data = feature_df[feature_df["symbol"] == symbol].sort_values("timestamp")
        if not symbol_data.empty:
            current_price = float(symbol_data["close"].iloc[-1])
            pos = trader.state["positions"][symbol]
            gross_return = (current_price - pos["entry_price"]) / pos["entry_price"]
            trader._execute_sell(symbol, current_price, gross_return, "EOD_CLOSE")
            
    print("-" * 60)
    print("\n Simulation Results:")
    summary = trader.get_portfolio_summary()
    for k, v in summary.items():
        if k != "positions" and k != "recent_outcomes":
            print(f"  {k}: {v}")
            
    trader.cleanup()

if __name__ == "__main__":
    run_simulation()
