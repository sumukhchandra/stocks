import pandas as pd
import numpy as np

class MarketImpactModel:
    """
    Simulates realistic price slippage based on liquidity and volatility.
    Replaces static bps penalties with depth-aware impact.
    """
    def __init__(self, base_slippage_bps=2.5):
        self.base_slippage_bps = base_slippage_bps

    def calculate_slippage(self, order_size_usd, spread_bps, volatility_5m, side='buy'):
        """
        Estimates slippage in bps.
        f(size, spread, volatility)
        """
        # 1. Base Spread Impact
        # Half the spread is the minimum cost to cross.
        spread_impact = spread_bps / 2.0
        
        # 2. Size Impact (Square root model common in HFT)
        # Assuming ADV (Average Daily Volume) proxy. 
        # For simplicity, we use a constant liquidity factor.
        liquidity_factor = 0.0001 # Higher = more impact
        size_impact = np.sqrt(order_size_usd) * liquidity_factor * 10000 # in bps
        
        # 3. Volatility Impact
        # High volatility increases the uncertainty of fill.
        vol_multiplier = 1.0 + (volatility_5m * 100)
        
        total_slippage_bps = (spread_impact + size_impact + self.base_slippage_bps) * vol_multiplier
        
        # Upper bound to prevent extreme outlier simulation
        total_slippage_bps = min(total_slippage_bps, 150) # Max 1.5% slippage
        
        return total_slippage_bps

if __name__ == "__main__":
    model = MarketImpactModel()
    # Large order in volatile market
    slippage = model.calculate_slippage(50000, 1.0, 0.02)
    print(f"Slippage for $50k order (spread 1bps, vol 2%): {slippage:.2f} bps")
    
    # Small order in calm market
    slippage = model.calculate_slippage(1000, 0.5, 0.002)
    print(f"Slippage for $1k order (spread 0.5bps, vol 0.2%): {slippage:.2f} bps")
