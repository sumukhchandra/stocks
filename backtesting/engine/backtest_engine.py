"""
Event-Driven Backtest Engine: Simulates historical execution with compounding and exact friction costs.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from ..metrics.performance import PerformanceMetrics
from backend.risk.india_tax_engine import IndiaTaxEngine


class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, min_confidence: float = 0.60):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.min_confidence = min_confidence
        self.tax_engine = IndiaTaxEngine()
        self.trades: List[Dict[str, Any]] = []

    def run(self, historical_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest across historical 5-minute dataset.
        """
        if historical_df.empty:
            return {"error": "Empty dataset"}

        df = historical_df.sort_values("timestamp").reset_index(drop=True)
        trade_returns = []

        # Group by timestamp and simulate walk-forward execution
        open_positions = {}
        for ts, bar in df.groupby("timestamp"):
            price_map = bar.set_index("symbol")["close"].to_dict()

            # 1. Evaluate Exits
            for sym in list(open_positions.keys()):
                if sym not in price_map:
                    continue
                curr_price = float(price_map[sym])
                pos = open_positions[sym]
                gross_ret = (curr_price - pos["entry_price"]) / pos["entry_price"]

                # Stop loss (0.8%) or take profit (0.5% net)
                if gross_ret <= -0.008 or gross_ret >= 0.005:
                    bd = self.tax_engine.calculate_total_cost(pos["invested"], gross_ret)
                    net_profit = bd["net_profit"]
                    self.capital += (pos["invested"] + net_profit)
                    trade_returns.append(bd["net_return_pct"] / 100.0)
                    self.trades.append({
                        "symbol": sym,
                        "entry_price": pos["entry_price"],
                        "exit_price": curr_price,
                        "net_profit": net_profit,
                        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    })
                    del open_positions[sym]

            # 2. Evaluate Entries
            if len(open_positions) < 5 and self.capital > 1000:
                for _, row in bar.iterrows():
                    sym = row["symbol"]
                    if sym in open_positions:
                        continue
                    prob = row.get("probability", 0.65)
                    if prob >= self.min_confidence:
                        alloc = self.capital * 0.15
                        self.capital -= alloc
                        open_positions[sym] = {
                            "symbol": sym,
                            "entry_price": float(row["close"]),
                            "invested": alloc
                        }
                        if len(open_positions) >= 5 or self.capital < 1000:
                            break

        metrics = PerformanceMetrics.calculate(trade_returns)
        metrics["starting_capital"] = self.initial_capital
        metrics["final_capital"] = round(self.capital + sum(p["invested"] for p in open_positions.values()), 2)
        metrics["total_net_pnl"] = round(metrics["final_capital"] - self.initial_capital, 2)
        return metrics
