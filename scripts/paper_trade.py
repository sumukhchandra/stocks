"""
Live Paper Trade CLI: Runs continuous systematic trading loop with live market scanning.
"""

import argparse
import sys
import os
import time
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trading.engine import SystematicTradingEngine


def main():
    parser = argparse.ArgumentParser(description="Run Systematic 5-Minute Paper Trading")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial Capital (INR)")
    parser.add_argument("--min-conf", type=float, default=0.60, help="Minimum confidence threshold")
    parser.add_argument("--single", action="store_true", help="Execute single cycle only")
    parser.add_argument("--interval", type=int, default=180, help="Scan interval seconds")
    args = parser.parse_args()

    engine = SystematicTradingEngine(initial_capital=args.capital, min_confidence=args.min_conf)
    print(f"Initialized Systematic 5-Minute Trading Engine. Model: {engine.predictor.model_version}")

    if args.single:
        print("Running single cycle...")
        res = engine.execute_cycle()
        print(f"Cycle Status: {res.get('status')}")
        print(f"Predictions: {len(res.get('predictions', []))}")
        print(f"Orders: {len(res.get('executed_orders', []))}")
        print(f"Rejections: {len(res.get('rejected_signals', []))}")
        return

    print("Starting continuous trading daemon... (Press Ctrl+C to stop)")
    while True:
        try:
            now = datetime.now()
            print(f"\n[{now.strftime('%H:%M:%S')}] Executing 5-minute systematic cycle...")
            res = engine.execute_cycle()
            print(f"  Orders: {len(res.get('executed_orders', []))}, Rejections: {len(res.get('rejected_signals', []))}")
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nTrading loop stopped by operator.")
            break
        except Exception as e:
            print(f"  [ERROR] {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
