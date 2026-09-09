import os
import sys
import pandas as pd
from datetime import date

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.nse_auto_trader import NSEAutoTrader
from data.fetch_stock_data import build_stock_dataset

# Custom state files for simulation
SIM_STATE_FILE = os.path.join(os.path.dirname(__file__), "logs", "batch_sim_trader_state.json")
SIM_TRADE_LOG = os.path.join(os.path.dirname(__file__), "logs", "batch_sim_trade_history.jsonl")
os.makedirs(os.path.dirname(SIM_STATE_FILE), exist_ok=True)

class SimulatorTrader(NSEAutoTrader):
    def __init__(self, capital, full_data, current_timestamp):
        self.full_data = full_data
        self.current_timestamp = current_timestamp
        
        import backend.nse_auto_trader as nse_auto_trader
        self.original_state_file = nse_auto_trader.STATE_FILE
        self.original_trade_log = nse_auto_trader.TRADE_LOG_FILE
        nse_auto_trader.STATE_FILE = SIM_STATE_FILE
        nse_auto_trader.TRADE_LOG_FILE = SIM_TRADE_LOG
        
        if os.path.exists(SIM_STATE_FILE):
            os.remove(SIM_STATE_FILE)
        if os.path.exists(SIM_TRADE_LOG):
            os.remove(SIM_TRADE_LOG)
            
        super().__init__(capital=capital)
        self.model_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "ml_models", "models", "saved_models"))
        self._ensure_engine()
        
    def _fetch_live_data(self, symbols=None):
        mask = self.full_data['timestamp'] <= self.current_timestamp
        return self.full_data[mask].copy()
        
    def cleanup(self):
        import backend.nse_auto_trader as nse_auto_trader
        nse_auto_trader.STATE_FILE = self.original_state_file
        nse_auto_trader.TRADE_LOG_FILE = self.original_trade_log

def run_batch_simulation():
    target_dates = [
        date(2026, 8, 22), date(2026, 8, 23), date(2026, 8, 24), date(2026, 8, 25), 
        date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 29), 
        date(2026, 8, 30), date(2026, 8, 31), date(2026, 9, 1), date(2026, 9, 2), 
        date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 5), date(2026, 9, 6), date(2026, 9, 7)
    ]
    
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "master_labeled_dataset.parquet")
    if os.path.exists(data_path):
        print("Loading cached 60d master dataset for simulation...")
        df = pd.read_parquet(data_path)
    else:
        print("Fetching historical data for batch simulation (60d)...")
        df = build_stock_dataset(period="60d", interval="5m")
    
    if df.empty:
        print("No data fetched.")
        return
        
    df['date_only'] = df['timestamp'].dt.date
    available_dates = set(df['date_only'])
    
    cumulative_capital = 100000.0
    all_session_results = []
    
    print("\n" + "=" * 65)
    print("  NSE CHRONOLOGICAL BATCH SIMULATION ENGINE (AUG 22 - SEP 7)")
    print("=" * 65)
    
    for target_date in target_dates:
        if target_date not in available_dates:
            print(f"Skipping {target_date} - Weekend or market closed.")
            continue
            
        print(f"\n>>> Simulating Trading Session: {target_date}")
        print(f"    Opening Capital: Rs.{cumulative_capital:,.2f}")
        
        day_data = df[df['date_only'] == target_date]
        timestamps = sorted(day_data['timestamp'].unique())
        
        trader = SimulatorTrader(capital=cumulative_capital, full_data=df, current_timestamp=timestamps[0])
        
        day_trades = 0
        for ts in timestamps:
            trader.current_timestamp = ts
            cycle_result = trader.run_single_cycle(min_confidence=0.50)
            if cycle_result.get("actions"):
                for act in cycle_result["actions"]:
                    day_trades += 1
                    symbol = act.get("symbol")
                    action = act.get("action")
                    price = act.get("price", 0)
                    prob = act.get("probability", 0)
                    print(f"    [{ts.strftime('%H:%M')}] {action} {symbol} @ Rs.{price:.2f} (Conf: {prob*100:.1f}%)")
                
        # End of day square-off
        trader.current_timestamp = timestamps[-1]
        feature_df = trader._fetch_live_data()
        
        open_positions = list(trader.state["positions"].keys())
        if open_positions:
            print(f"    Closing {len(open_positions)} open position(s) at EOD square-off...")
            for symbol in open_positions:
                symbol_data = feature_df[feature_df["symbol"] == symbol].sort_values("timestamp")
                if not symbol_data.empty:
                    current_price = float(symbol_data["close"].iloc[-1])
                    pos = trader.state["positions"][symbol]
                    gross_return = (current_price - pos["entry_price"]) / pos["entry_price"]
                    trader._execute_sell(symbol, current_price, gross_return, "EOD_SQUAREOFF")
                
        summary = trader.get_portfolio_summary()
        closing_capital = summary["total_capital"]
        day_pnl = closing_capital - cumulative_capital
        cumulative_capital = closing_capital
        
        print(f"    Session Summary: Closing Capital: Rs.{closing_capital:,.2f} | Day Net P&L: Rs.{day_pnl:+,.2f} | Trades: {summary['total_trades']}")
        all_session_results.append({
            "date": str(target_date),
            "closing_capital": closing_capital,
            "day_pnl": day_pnl,
            "total_trades": summary["total_trades"],
            "win_rate": summary["win_rate"]
        })
        trader.cleanup()

    print("\n" + "=" * 65)
    print("  BATCH SIMULATION FINAL PERFORMANCE REPORT")
    print("=" * 65)
    total_pnl = cumulative_capital - 100000.0
    total_pct = (total_pnl / 100000.0) * 100
    total_sim_trades = sum(r["total_trades"] for r in all_session_results)
    print(f"  Starting Capital:   Rs.100,000.00")
    print(f"  Final Capital:      Rs.{cumulative_capital:,.2f}")
    print(f"  Total Net Return:   Rs.{total_pnl:+,.2f} ({total_pct:+.2f}%)")
    print(f"  Total Executions:   {total_sim_trades} trades across {len(all_session_results)} active sessions")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_batch_simulation()
