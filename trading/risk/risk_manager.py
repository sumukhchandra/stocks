"""
Sovereign Risk Manager: Evaluates signals and portfolio state to approve or veto orders.
"""

from typing import Dict, Any, Tuple, Optional
from datetime import datetime, time
from .kill_switch import KillSwitch
from ..data.processors.validator import DataValidator
from backend.risk.india_tax_engine import IndiaTaxEngine
from database.relations import STOCK_SECTOR_MAP


class RiskManager:
    def __init__(
        self,
        max_positions: int = 5,
        max_sector_pct: float = 40.0,
        max_daily_drawdown_pct: float = 3.0,
        min_net_profit_pct: float = 0.005,
        max_gross_loss_pct: float = 0.008,
    ):
        self.max_positions = max_positions
        self.max_sector_pct = max_sector_pct
        self.min_net_profit_pct = min_net_profit_pct
        self.max_gross_loss_pct = max_gross_loss_pct
        self.kill_switch = KillSwitch(max_daily_drawdown_pct=max_daily_drawdown_pct)
        self.tax_engine = IndiaTaxEngine()

    def audit_signal(
        self,
        signal: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        latest_timestamp: Optional[datetime] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Sovereign Risk Audit.
        Returns (is_approved, rejection_reason, risk_metadata).
        """
        # 1. Kill switch check
        if self.kill_switch.is_active:
            return False, f"Kill Switch ACTIVE: {self.kill_switch.trigger_reason}", {}

        # 2. Drawdown check
        start_cap = portfolio_state.get("starting_capital", 100000.0)
        total_cap = portfolio_state.get("total_capital", 100000.0)
        if self.kill_switch.check_drawdown(start_cap, total_cap):
            return False, f"Circuit Breaker Tripped: {self.kill_switch.trigger_reason}", {}

        # 3. Market hours check
        now = datetime.now()
        if not DataValidator.is_market_hours(now):
            return False, "Market is closed (Allowed: 9:15 AM - 3:30 PM IST Mon-Fri)", {}

        # 4. EOD square-off cutoff (3:15 PM)
        if now.time() >= time(15, 15):
            return False, "EOD Cutoff passed (No new entries after 15:15 IST)", {}

        # 5. Stale data check
        if latest_timestamp:
            stale, reason = DataValidator.is_stale(latest_timestamp)
            if stale:
                return False, f"Stale Data Reject: {reason}", {}

        action = signal.get("action")
        if action != "BUY":
            return False, f"Non-actionable signal: {action}", {}

        symbol = signal.get("symbol")
        positions = portfolio_state.get("positions", {})

        # 6. Already holding symbol
        if symbol in positions:
            return False, f"Symbol {symbol} already held in active portfolio", {}

        # 7. Max concurrent positions
        if len(positions) >= self.max_positions:
            return False, f"Max concurrent positions reached ({len(positions)}/{self.max_positions})", {}

        # 8. Available liquidity check
        available_cash = portfolio_state.get("available_capital", 0.0)
        if available_cash < 1000.0:
            return False, f"Insufficient liquidity: ₹{available_cash:,.2f}", {}

        # 9. Sector concentration check
        sector = STOCK_SECTOR_MAP.get(symbol, "Unknown")
        sector_invested = sum(
            pos.get("invested", 0.0) for sym, pos in positions.items()
            if STOCK_SECTOR_MAP.get(sym) == sector
        )
        if total_cap > 0 and (sector_invested / total_cap) * 100.0 >= self.max_sector_pct:
            return False, f"Sector limit reached for {sector} (>= {self.max_sector_pct}%)", {}

        # 10. Cost & Viability Gating (Zerodha fees + 25% STCG tax)
        exp_return = signal.get("expected_return", 0.0)
        viable, breakdown = self.tax_engine.is_trade_viable(
            capital=available_cash * 0.15,
            expected_return=exp_return,
            min_net_pct=self.min_net_profit_pct
        )

        if not viable:
            return False, (
                f"Trade not viable after Indian taxes/fees "
                f"(Expected Net: {breakdown.get('net_return_pct', 0):.3f}%, Min Required: {self.min_net_profit_pct*100:.2f}%)"
            ), breakdown

        return True, "Approved by Sovereign Risk Engine", breakdown
