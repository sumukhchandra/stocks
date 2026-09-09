import numpy as np

class PositionSizer:
    """
    Institutional-grade position sizing engine.
    Combines Kelly Criterion with Volatility Scaling (ATR).
    """
    
    def __init__(self, Kelly_fraction: float = 0.5, max_risk_per_trade: float = 0.02):
        self.Kelly_fraction = Kelly_fraction # Fractional Kelly to be conservative
        self.max_risk_per_trade = max_risk_per_trade # Cap risk at 2% of equity

    def calculate_kelly_size(self, prob: float, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculates the Kelly optimal position size.
        f* = (p * b - (1 - p)) / b
        where b = avg_win / avg_loss
        """
        if avg_loss == 0:
            return 0.0
            
        b = avg_win / avg_loss
        if b == 0:
            return 0.0
            
        # Standard Kelly Formula
        f_star = (prob * b - (1 - prob)) / b
        
        # Apply fractional Kelly for safety
        size = f_star * self.Kelly_fraction
        return max(0.0, size)

    def calculate_volatility_scaled_size(self, base_size: float, current_vol: float, target_vol: float) -> float:
        """
        Scales the position size based on current vs target volatility.
        size = base_size * (target_vol / current_vol)
        """
        if current_vol == 0:
            return base_size
            
        scaling_factor = target_vol / current_vol
        # Limit scaling factor to prevent extreme sizes
        scaling_factor = min(2.0, max(0.2, scaling_factor))
        
        return base_size * scaling_factor

    def get_final_size(self, prob: float, current_vol: float, 
                       avg_win: float, avg_loss: float, 
                       equity: float, target_vol: float = 0.01) -> float:
        """
        Returns the final recommended position size in equity units.
        """
        # 1. Start with Kelly-inspired size based on model confidence
        # We use 'prob' as our 'p' for this specific trade
        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        # If prob is below the "no edge" threshold (1/(b+1)), size is 0
        if prob <= 1 / (b + 1):
            return 0.0
            
        kelly_size = self.calculate_kelly_size(prob, prob, avg_win, avg_loss)
        
        # 2. Scale by volatility
        final_size_pct = self.calculate_volatility_scaled_size(kelly_size, current_vol, target_vol)
        
        # 3. Apply hard risk cap
        final_size_pct = min(final_size_pct, self.max_risk_per_trade)
        
        return final_size_pct * equity

if __name__ == "__main__":
    # Example Usage
    sizer = PositionSizer()
    # Assume 55% prob, 2:1 reward/risk, 100k equity, current vol 2%, target vol 1%
    size = sizer.get_final_size(prob=0.55, current_vol=0.02, avg_win=0.01, avg_loss=0.005, equity=100000)
    print(f"Recommended Position Size: ${size:,.2f}")
