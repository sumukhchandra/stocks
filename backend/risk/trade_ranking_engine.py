import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from risk.india_tax_engine import IndiaTaxEngine

class TradeRankingEngine:
    """
    Synthesizes multiple signals into a single Trade Quality Score.
    Ranks opportunities by Net Post-Tax Expectancy and penalizes overtrading.
    """
    def __init__(self, min_gross_move=None):
        self.tax_engine = IndiaTaxEngine()
        self.min_gross_move = min_gross_move or self.tax_engine.get_required_gross_for_net(0.01)

    def calculate_score(self, prob, expected_return, confidence, volatility, regime, daily_trade_count=0):
        """
        Combines directional probability, magnitude, and risk into a score.
        """
        # 1. Hard Filter: Dynamic Profit Threshold
        # If expected return is too small, reject immediately to avoid cost drag.
        abs_expected_ret = abs(expected_return)
        if abs_expected_ret < self.min_gross_move:
            return {'quality_score': 0.0, 'is_high_quality': False, 'reason': 'MOVE_TOO_SMALL'}

        # 2. Base Expected Gross Return (Direction-adjusted)
        direction_mult = (prob - 0.5) * 2.0
        adjusted_gross = expected_return * direction_mult
        
        # 3. Net Post-Tax Return
        slippage_bps = 2.5 + (volatility * 100)
        net_return = self.tax_engine.calculate_net_return(adjusted_gross, slippage_bps=slippage_bps)
        
        # 4. Confidence and Regime Weighting
        regime_penalty = 1.0
        if regime in ['high_volatility', 'flash_crash']:
            regime_penalty = 0.5
            
        # 5. Trade Frequency Penalty (Exponential)
        # Each trade in last 24h makes the next one harder to justify.
        frequency_penalty = 1.0 / (1.0 + (daily_trade_count ** 1.5) * 0.05)
            
        # Final Score
        quality_score = net_return * confidence * regime_penalty * frequency_penalty
        
        return {
            'quality_score': quality_score,
            'net_expectancy_bps': net_return * 10000,
            'adjusted_gross_bps': adjusted_gross * 10000,
            'frequency_penalty': frequency_penalty,
            'is_high_quality': quality_score > 0.0005 # Stricter threshold for swing
        }

if __name__ == "__main__":
    ranker = TradeRankingEngine()
    # High confidence, high return trade
    res = ranker.calculate_score(0.7, 0.02, 0.8, 0.01, 'sideways')
    print(f"High Quality Trade Result: {res}")
    
    # Low confidence trade
    res = ranker.calculate_score(0.55, 0.005, 0.4, 0.02, 'high_volatility')
    print(f"Low Quality Trade Result: {res}")
