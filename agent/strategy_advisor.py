"""
agent.strategy_advisor: Quantitative Strategy and Threshold Advisor.
Analyzes historical performance, evaluates risk-adjusted metrics, and advises
on optimal entry confidence thresholds based on recent market conditions.
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.db_manager import DatabaseManager

class StrategyAdvisor:
    """Evaluates trading performance and optimizes decision thresholds."""

    def __init__(self):
        self.db = DatabaseManager()

    def evaluate_performance(self):
        """Compute Sharpe ratio, Win Rate, and Profit Factor from trade history."""
        trades = self.db.get_recent_trades(limit=200)
        if not trades:
            return {
                "status": "NO_TRADES",
                "message": "No executed trades found in database.",
                "recommended_threshold": 0.55
            }

        df = pd.DataFrame(trades)
        net_pnls = df["net_profit"].astype(float).values
        returns = df["net_return_pct"].astype(float).values / 100.0

        total_trades = len(net_pnls)
        wins = np.sum(net_pnls > 0)
        losses = np.sum(net_pnls <= 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        gross_gains = np.sum(net_pnls[net_pnls > 0]) if wins > 0 else 0.0
        gross_losses = np.abs(np.sum(net_pnls[net_pnls < 0])) if losses > 0 else 0.0
        profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (gross_gains if gross_gains > 0 else 1.0)

        mean_ret = np.mean(returns) if len(returns) > 0 else 0.0
        std_ret = np.std(returns) if len(returns) > 0 else 1e-6
        sharpe = (mean_ret / std_ret * np.sqrt(252 * 75)) if std_ret > 0 else 0.0

        # Dynamic threshold recommendation
        if win_rate < 50.0 and total_trades >= 10:
            recommended_threshold = 0.65  # Tighten up to filter false positives
            action_advice = "TIGHTEN: Model win rate is sub-50%. Raise minimum confidence to 65% to demand higher certainty."
        elif win_rate > 70.0 and total_trades >= 10:
            recommended_threshold = 0.52  # Relax slightly to capture more volume
            action_advice = "EXPAND: Win rate is strong (>70%). Confidence threshold can be safely relaxed to 52% to capture more opportunities."
        else:
            recommended_threshold = 0.55
            action_advice = "MAINTAIN: System operating within normal equilibrium parameters. Maintain default 55% threshold."

        return {
            "status": "ANALYZED",
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_net_pnl": round(float(np.sum(net_pnls)), 2),
            "recommended_threshold": recommended_threshold,
            "advice": action_advice
        }
