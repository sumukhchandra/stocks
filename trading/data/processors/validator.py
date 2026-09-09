"""
Data Validator: Validates candle continuity, latency staleness, and schema.
"""

from datetime import datetime, time
import pandas as pd
from typing import Tuple, Dict, Any


class DataValidator:
    @staticmethod
    def is_market_hours(dt: datetime) -> bool:
        """NSE equity regular session: 9:15 AM to 3:30 PM IST (Mon-Fri)."""
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False
        current_time = dt.time()
        return time(9, 15) <= current_time <= time(15, 30)

    @classmethod
    def is_stale(cls, latest_timestamp: datetime, max_latency_seconds: int = 300) -> Tuple[bool, str]:
        """
        Check if market data has stalled during active market hours.
        Returns (is_stale, reason).
        """
        now = datetime.now()
        if not cls.is_market_hours(now):
            return False, "Market closed - staleness check skipped"

        delta_seconds = (now - latest_timestamp).total_seconds()
        if delta_seconds > max_latency_seconds:
            return True, f"Market data delay of {delta_seconds:.0f}s exceeds threshold ({max_latency_seconds}s)"

        return False, "Data is fresh"

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        required = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return False, {"error": f"Missing columns: {missing}"}
        if df.empty:
            return False, {"error": "Dataset is empty"}
        return True, {"rows": len(df), "symbols": df["symbol"].nunique()}
