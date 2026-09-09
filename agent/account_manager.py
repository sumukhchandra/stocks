"""
agent.account_manager: Account and Risk Watchdog.
Monitors portfolio balances, sector concentrations, drawdown, and tax obligations.
"""
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.relations import get_stock_sector
from database.db_manager import DatabaseManager

class AccountManager:
    """Monitors and manages capital, positions, and risk exposures."""

    def __init__(self, state_file=None):
        self.state_file = state_file or os.path.join(PROJECT_ROOT, "backend", "logs", "trader_state.json")
        self.db = DatabaseManager()

    def get_account_report(self):
        """Generate comprehensive account financial report."""
        state = self._read_state()
        total_cap = float(state.get("total_capital", 100000.0))
        avail_cap = float(state.get("available_capital", 100000.0))
        start_cap = float(state.get("starting_capital", 100000.0))
        net_pnl = float(state.get("total_net_pnl", 0.0))
        positions = state.get("positions", {})

        invested = sum(p.get("invested", 0.0) for p in positions.values())
        invested_pct = (invested / total_cap * 100) if total_cap > 0 else 0.0
        pnl_pct = (net_pnl / start_cap * 100) if start_cap > 0 else 0.0

        # Sector breakdown
        sector_allocation = {}
        for symbol, pos in positions.items():
            sec = get_stock_sector(symbol)
            sector_allocation[sec] = sector_allocation.get(sec, 0.0) + pos.get("invested", 0.0)

        sector_pcts = {
            s: (amt / total_cap * 100) for s, amt in sector_allocation.items()
        } if total_cap > 0 else {}

        # Risk warnings
        risk_alerts = []
        for sec, pct in sector_pcts.items():
            if pct > 40.0:
                risk_alerts.append(f"Sector Concentration Warning: {sec} is {pct:.1f}% of capital (limit 40%)")

        if invested_pct > 80.0:
            risk_alerts.append(f"High Capital Utilization Warning: {invested_pct:.1f}% currently allocated")

        return {
            "total_capital": total_cap,
            "available_capital": avail_cap,
            "invested_capital": invested,
            "invested_pct": round(invested_pct, 2),
            "net_pnl": round(net_pnl, 2),
            "net_pnl_pct": round(pnl_pct, 3),
            "open_positions_count": len(positions),
            "positions": positions,
            "sector_allocation_pct": sector_pcts,
            "risk_alerts": risk_alerts,
            "health": "STABLE" if not risk_alerts else "ATTENTION_REQUIRED"
        }

    def _read_state(self):
        """Read trader state from JSON or database."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        
        acc = self.db.get_account("main")
        if acc:
            return acc
        return {"total_capital": 100000.0, "available_capital": 100000.0, "positions": {}}
