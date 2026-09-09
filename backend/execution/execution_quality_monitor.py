import pandas as pd
import numpy as np
import os
from datetime import datetime

class ExecutionQualityMonitor:
    """
    Tracks slippage, latency, and fill quality for live and paper trading.
    """
    def __init__(self, log_dir='logs/live_metrics'):
        self.log_path = os.path.join(log_dir, 'execution_quality.csv')

    def log_execution(self, symbol, side, requested_price, actual_price, latency_ms):
        """
        Records an execution event and calculates slippage.
        """
        slippage_pct = (actual_price - requested_price) / requested_price
        if side.lower() == 'sell':
            slippage_pct = -slippage_pct
            
        log_entry = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'requested_price': requested_price,
            'actual_price': actual_price,
            'slippage_pct': slippage_pct,
            'latency_ms': latency_ms,
            'quality_score': max(0, 1 - abs(slippage_pct) * 100) # Simple 0-1 score
        }
        
        df = pd.DataFrame([log_entry])
        header = not os.path.exists(self.log_path)
        df.to_csv(self.log_path, mode='a', header=header, index=False)

    def get_summary_metrics(self):
        if not os.path.exists(self.log_path):
            return {'avg_slippage': 0, 'avg_latency': 0}
            
        df = pd.read_csv(self.log_path)
        return {
            'avg_slippage_bps': df['slippage_pct'].mean() * 10000,
            'avg_latency_ms': df['latency_ms'].mean(),
            'max_slippage_pct': df['slippage_pct'].max(),
            'execution_count': len(df)
        }

if __name__ == "__main__":
    monitor = ExecutionQualityMonitor()
    monitor.log_execution('BTCUSDT', 'BUY', 60000, 60005, 45)
    print(monitor.get_summary_metrics())
