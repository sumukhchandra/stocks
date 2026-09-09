"""
database.rules: Centralized business, risk, timing, and execution rules.
"""
from datetime import datetime, time
import pytz

class TradingRulesEngine:
    """Evaluates and enforces core business and risk rules."""

    def __init__(self, tz_str="Asia/Kolkata"):
        self.tz = pytz.timezone(tz_str)
        # Market timing rules (NSE: 9:15 AM to 3:30 PM IST, Monday-Friday)
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
        self.eod_square_off = time(15, 15) # Force exit 15 min before close
        
        # Risk & capital rules
        self.min_confidence = 0.50
        self.min_net_profit_pct = 0.0005 # 0.05% net after all costs
        self.max_gross_loss_pct = 0.003  # 0.3% stop-loss
        self.max_concurrent_positions = 5
        self.max_sector_allocation_pct = 0.40 # Max 40% in single sector
        self.max_portfolio_drawdown_pct = 0.03 # 3% max daily drawdown circuit breaker

    def check_market_hours(self, dt=None):
        """Returns True if within active NSE trading hours."""
        if dt is None:
            now = datetime.now(self.tz)
        elif dt.tzinfo is None:
            now = self.tz.localize(dt)
        else:
            now = dt.astimezone(self.tz)

        # Check weekday (0=Mon, 4=Fri)
        if now.weekday() > 4:
            return False, "Market closed (Weekend)"

        curr_time = now.time()
        if curr_time < self.market_open:
            return False, f"Market pre-open (Opens at {self.market_open.strftime('%H:%M')})"
        if curr_time >= self.market_close:
            return False, f"Market closed (Closed at {self.market_close.strftime('%H:%M')})"

        return True, "Market open"

    def is_square_off_time(self, dt=None):
        """Returns True if at or past intraday square-off time (3:15 PM)."""
        if dt is None:
            now = datetime.now(self.tz)
        elif dt.tzinfo is None:
            now = self.tz.localize(dt)
        else:
            now = dt.astimezone(self.tz)

        return now.time() >= self.eod_square_off

    def check_position_limits(self, current_positions, new_symbol):
        """Ensures max concurrent positions and duplicate prevention."""
        if new_symbol in current_positions:
            return False, f"Already holding position in {new_symbol}"
        if len(current_positions) >= self.max_concurrent_positions:
            return False, f"Max concurrent positions reached ({self.max_concurrent_positions})"
        return True, "Position limit check passed"

    def check_capital_adequacy(self, available_capital, min_capital=1000.0):
        """Ensures sufficient capital exists to execute trade."""
        if available_capital < min_capital:
            return False, f"Insufficient capital (Available: Rs.{available_capital:.2f}, Required: Rs.{min_capital:.2f})"
        return True, "Capital adequacy check passed"

    def check_circuit_breaker(self, peak_capital, current_capital):
        """Checks if current drawdown triggers circuit breaker."""
        if peak_capital <= 0:
            return True, "Peak capital invalid or zero"
        drawdown = (peak_capital - current_capital) / peak_capital
        if drawdown >= self.max_portfolio_drawdown_pct:
            return False, f"Circuit breaker tripped: Drawdown is {drawdown*100:.2f}% (Limit: {self.max_portfolio_drawdown_pct*100:.2f}%)"
        return True, "Circuit breaker check passed"

    def check_trade_viability(self, confidence, predicted_net_return_pct):
        """Ensures confidence score and predicted net return pass trading gates."""
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence*100:.1f}% below minimum {self.min_confidence*100:.1f}%"
        if predicted_net_return_pct < (self.min_net_profit_pct * 100):
            return False, f"Predicted net return {predicted_net_return_pct:.4f}% below threshold {self.min_net_profit_pct*100:.2f}%"
        return True, "Viability check passed"
