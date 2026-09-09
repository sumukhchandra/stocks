import pandas as pd
import numpy as np
import os

class DynamicPositionSizer:
    """
    Calculates trade size based on model confidence, volatility, and account risk.
    """
    def __init__(self, account_balance=10000.0, max_risk_pct=0.01):
        self.account_balance = account_balance
        self.max_risk_pct = max_risk_pct # Risk 1% of account per trade

    def calculate_size(self, price, stop_loss_price, confidence_prob, volatility):
        """
        Uses fixed fractional sizing adjusted by confidence.
        size = (Account * Risk%) / (Entry - StopLoss) * ConfidenceFactor
        """
        risk_amount = self.account_balance * self.max_risk_pct
        price_risk = abs(price - stop_loss_price)
        
        if price_risk == 0:
            return 0
            
        base_size = risk_amount / price_risk
        
        # Adjust by confidence (Kelly Criterion inspired or simple linear scaling)
        # If prob = 0.5 (random), size = 0
        # If prob = 0.8, size = base_size * 1.0 (or more)
        confidence_factor = max(0, (confidence_prob - 0.5) * 2)
        
        # Volatility scaling: reduce size if volatility is extremely high
        vol_scaler = 1.0
        if volatility > 0.05: # Threshold example
            vol_scaler = 0.5
            
        final_size = base_size * confidence_factor * vol_scaler
        
        # Cap at max leverage (e.g. 5x)
        max_leverage_size = (self.account_balance * 5) / price
        final_size = min(final_size, max_leverage_size)
        
        return final_size

if __name__ == "__main__":
    sizer = DynamicPositionSizer()
    entry = 50000.0
    sl = 49500.0
    prob = 0.75
    vol = 0.01
    
    size = sizer.calculate_size(entry, sl, prob, vol)
    print(f"Confidence: {prob}, Position Size: {size:.4f} units (${size*entry:.2f})")
