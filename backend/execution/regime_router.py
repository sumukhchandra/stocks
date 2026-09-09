import pandas as pd
import numpy as np
import joblib
import os

class RegimeRouter:
    """
    Decides which model to use and adjusts risk based on the current regime.
    """
    def __init__(self, model_map, default_model='ensemble'):
        self.model_map = model_map # {'sideways': 'mean_rev', 'trending_bull': 'momentum', ...}
        self.default_model = default_model

    def route(self, regime):
        model_name = self.model_map.get(regime, self.default_model)
        return model_name

    def adjust_risk_multiplier(self, regime, last_macro_severity, macro_risk_factor):
        """
        Reduces risk in high-risk macro regimes or crisis events.
        """
        multiplier = 1.0
        
        if regime == 'high_volatility':
            multiplier *= 0.5
        elif regime == 'trending_bear':
            multiplier *= 0.8 # More cautious in bear markets
            
        # Macro risk adjustment
        if last_macro_severity > 0.8:
            multiplier *= 0.3 # Sharp reduction during systemic crashes
            
        if macro_risk_factor > 0.5:
            multiplier *= 0.7 # Reduction when close to macro events
            
        return max(0.1, multiplier)

if __name__ == "__main__":
    router = RegimeRouter({
        'sideways': 'mean_reversion_model',
        'trending_bull': 'momentum_model',
        'high_volatility': 'breakout_model'
    })
    
    print(f"Route for sideways: {router.route('sideways')}")
    print(f"Risk multiplier for high_volatility: {router.adjust_risk_multiplier('high_volatility', 0, 0)}")
    print(f"Risk multiplier during FTX collapse (sev 1.0): {router.adjust_risk_multiplier('sideways', 1.0, 1.0)}")
