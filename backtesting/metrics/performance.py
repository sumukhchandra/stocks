"""
Performance Metrics: Sharpe ratio, Max Drawdown, Win Rate, Profit Factor, and Expectancy.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


class PerformanceMetrics:
    @staticmethod
    def calculate(trade_returns: List[float], risk_free_rate: float = 0.06) -> Dict[str, Any]:
        if not trade_returns:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "net_profit_pct": 0.0
            }

        returns = np.array(trade_returns)
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = (len(wins) / len(returns)) * 100.0
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.0

        # Annualized Sharpe (assuming 252 days * 75 five-minute bars = 18,900 bars/year)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = ((mean_ret - (risk_free_rate / 18900.0)) / (std_ret + 1e-8)) * np.sqrt(18900) if std_ret > 0 else 0.0

        # Max Drawdown
        cum_ret = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cum_ret)
        drawdowns = (cum_ret - running_max) / running_max
        max_dd = abs(float(np.min(drawdowns))) * 100.0 if len(drawdowns) > 0 else 0.0

        return {
            "total_trades": len(returns),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown_pct": round(max_dd, 2),
            "net_profit_pct": round(float(np.sum(returns)) * 100.0, 2)
        }
