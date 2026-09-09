import pandas as pd
import numpy as np

class VolatilityFilter:
    def __init__(self, atr_multiplier=3.0, volume_multiplier=5.0):
        self.atr_multiplier = atr_multiplier
        self.volume_multiplier = volume_multiplier
        
    def is_safe_to_trade(self, current_candle, recent_candles):
        """
        Determines if current market conditions are safe to trade.
        Blocks trades during abnormal volatility or massive volume spikes.
        
        Args:
            current_candle (dict or pd.Series): Current 5m candle features.
            recent_candles (pd.DataFrame): Recent historical candles (e.g., last 20 periods).
            
        Returns:
            bool: True if safe to trade, False otherwise.
        """
        if recent_candles is None or len(recent_candles) < 14:
            # Not enough data to determine safety
            return False
            
        # Calculate recent average ATR
        avg_atr = recent_candles['atr_14'].mean()
        
        # Calculate recent average volume
        avg_volume = recent_candles['volume'].mean()
        
        # Check for abnormal price volatility (e.g. current range > 3x average ATR)
        # Assuming current_candle has 'high' and 'low'
        current_range = current_candle.get('high', 0) - current_candle.get('low', 0)
        if avg_atr > 0 and current_range > (self.atr_multiplier * avg_atr):
            print("Volatility filter triggered: Abnormal price range.")
            return False # Volatility spike
            
        # Check for abnormal volume (liquidation cascade or news spike)
        if avg_volume > 0 and current_candle.get('volume', 0) > (self.volume_multiplier * avg_volume):
            print("Volatility filter triggered: Abnormal volume spike.")
            return False # Volume spike
            
        return True

if __name__ == "__main__":
    pass
