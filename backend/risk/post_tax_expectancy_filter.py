import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from risk.india_tax_engine import IndiaTaxEngine

class PostTaxExpectancyFilter:
    """
    Rejects trades that do not meet post-tax profitability thresholds.
    """
    def __init__(self, min_net_bps=15):
        self.tax_engine = IndiaTaxEngine()
        self.min_net_bps = min_net_bps

    def should_execute(self, predicted_return, volatility):
        """
        Decides if a trade is viable after considering taxes and risk-adjusted costs.
        """
        # Adjust slippage expectation based on volatility
        slippage_bps = 2.5 + (volatility * 100) # Simple linear scaling
        
        net_return = self.tax_engine.calculate_net_return(predicted_return, slippage_bps=slippage_bps)
        
        is_viable = net_return > (self.min_net_bps / 10000)
        
        return is_viable, {
            'predicted_gross_bps': predicted_return * 10000,
            'expected_net_bps': net_return * 10000,
            'estimated_slippage_bps': slippage_bps,
            'tax_drag_bps': (predicted_return - net_return) * 10000
        }

if __name__ == "__main__":
    filter = PostTaxExpectancyFilter()
    viable, metrics = filter.should_execute(0.02, 0.01)
    print(f"Is trade viable? {viable}")
    print(f"Metrics: {metrics}")
