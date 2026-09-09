import pandas as pd
import numpy as np
import time

class LatencySimulator:
    """
    Simulates execution latency and slippage.
    """
    def __init__(self, base_latency_ms=50, jitter_ms=20):
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms

    def simulate_order_latency(self):
        """Returns total latency in seconds."""
        latency = self.base_latency_ms + np.random.randint(-self.jitter_ms, self.jitter_ms)
        return latency / 1000.0

    def calculate_slippage(self, price, volatility, side='buy'):
        """
        Estimates slippage based on volatility and latency.
        Higher volatility + higher latency = higher slippage.
        """
        latency = self.simulate_order_latency()
        # Simple slippage model: slippage % = latency * volatility * coefficient
        # coeff = 0.5 (aggressive)
        slippage_pct = latency * volatility * 0.5
        
        if side == 'buy':
            execution_price = price * (1 + slippage_pct)
        else:
            execution_price = price * (1 - slippage_pct)
            
        return execution_price, slippage_pct

class ExecutionFilter:
    """
    Applies filters before allowing a trade.
    """
    def __init__(self, min_prob=0.6, max_spread=0.0005, max_vpin=0.8):
        self.min_prob = min_prob
        self.max_spread = max_spread
        self.max_vpin = max_vpin

    def should_trade(self, prob, spread, vpin, regime):
        if prob < self.min_prob:
            return False, "Low probability"
        if spread > self.max_spread:
            return False, "High spread"
        if vpin > self.max_vpin:
            return False, "High toxic flow (VPIN)"
        if regime == 'high_volatility':
            # Maybe only trade breakouts in high vol
            return True, "High vol breakout"
        return True, "Condition met"

if __name__ == "__main__":
    sim = LatencySimulator()
    price = 50000.0
    vol = 0.02 # 2% per 5m (very high)
    exec_price, slip = sim.calculate_slippage(price, vol)
    print(f"Price: {price}, Exec Price: {exec_price:.2f}, Slippage: {slip*100:.4f}%")
