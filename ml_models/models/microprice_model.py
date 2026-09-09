import pandas as pd
import numpy as np

class MicropriceModel:
    """
    Estimates the 'True' fair price by accounting for orderbook imbalance and spread.
    Based on Stoikov (2018) 'The Micro-Price'.
    """
    def __init__(self):
        pass

    def calculate_microprice(self, bid_price, bid_qty, ask_price, ask_qty):
        """
        Standard Microprice: Weighted average of prices by opposite quantities.
        Higher bid qty pushes microprice toward the ask.
        """
        total_qty = bid_qty + ask_qty
        if total_qty == 0:
            return (bid_price + ask_price) / 2.0
            
        return (bid_price * ask_qty + ask_price * bid_qty) / total_qty

    def estimate_drift(self, df):
        """
        Estimates short-term directional pressure (drift) based on microprice 
        vs midprice divergence.
        """
        midprice = (df['best_bid'] + df['best_ask']) / 2.0
        microprice = self.calculate_microprice(
            df['best_bid'], df['best_bid_qty'], 
            df['best_ask'], df['best_ask_qty']
        )
        
        # Positive divergence means microprice is above midprice (upward pressure)
        return microprice - midprice

if __name__ == "__main__":
    # Test case
    model = MicropriceModel()
    bp, bq = 60000, 1.5
    ap, aq = 60001, 0.5
    
    mp = (bp + ap) / 2.0
    micp = model.calculate_microprice(bp, bq, ap, aq)
    
    print(f"Midprice: {mp}")
    print(f"Microprice: {micp}")
    print(f"Drift Pressure: {micp - mp}")
