"""
Run Backtest CLI: Executes systematic backtest over historical candles and outputs metrics.
"""

import argparse
import sys
import os
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trading.data.storage.parquet_store import ParquetStore
from backtesting.engine.backtest_engine import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="Run Systematic Backtest")
    parser.add_argument("--capital", type=float, default=100000.0, help="Starting capital (INR)")
    parser.add_argument("--min-conf", type=float, default=0.60, help="Minimum confidence threshold")
    parser.add_argument("--dataset", default="master_labeled_dataset.parquet", help="Parquet dataset filename")
    args = parser.parse_args()

    store = ParquetStore()
    df = store.load_dataset(args.dataset)
    if df.empty:
        print(f"Error: Dataset {args.dataset} not found in storage.")
        return

    print(f"Running backtest on {len(df):,} candles with starting capital: ₹{args.capital:,.2f}...")
    engine = BacktestEngine(initial_capital=args.capital, min_confidence=args.min_conf)
    results = engine.run(df)

    print("\n" + "=" * 55)
    print("  SYSTEMATIC BACKTEST PERFORMANCE REPORT (INR / ₹)")
    print("=" * 55)
    print(f"  Starting Capital:    ₹{results.get('starting_capital', 0):,.2f}")
    print(f"  Final Capital:       ₹{results.get('final_capital', 0):,.2f}")
    print(f"  Total Net Return:    ₹{results.get('total_net_pnl', 0):+,.2f} ({results.get('net_profit_pct', 0):+.2f}%)")
    print(f"  Total Executions:    {results.get('total_trades', 0)} trades")
    print(f"  Win Rate:            {results.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor:       {results.get('profit_factor', 0):.2f}")
    print(f"  Sharpe Ratio:        {results.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown:        {results.get('max_drawdown_pct', 0):.2f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
