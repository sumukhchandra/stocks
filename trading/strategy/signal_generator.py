"""
Signal Generator: Evaluates 5-minute predictions and market conditions to produce trading signals.
"""

from typing import Dict, Any, Optional
from datetime import datetime


class SignalGenerator:
    def __init__(self, min_confidence: float = 0.60, min_expected_return: float = 0.001):
        self.min_confidence = min_confidence
        self.min_expected_return = min_expected_return

    def generate_signal(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate prediction and emit BUY, SELL, or HOLD.
        """
        symbol = prediction["symbol"]
        prob = prediction.get("probability", 0.5)
        exp_ret = prediction.get("expected_return", 0.0)
        current_price = prediction.get("current_price", 0.0)
        model_version = prediction.get("model_version", "unknown")

        if prob >= self.min_confidence and exp_ret >= self.min_expected_return:
            action = "BUY"
            reason = f"High Confidence ({prob:.1%}) & Viable Return ({exp_ret*100:+.3f}%)"
        elif prob < 0.40 and exp_ret < -self.min_expected_return:
            action = "SELL"
            reason = f"Bearish Drift (Prob: {prob:.1%}, Exp Ret: {exp_ret*100:+.3f}%)"
        else:
            action = "HOLD"
            reason = f"Within Noise Band (Prob: {prob:.1%}, Exp Ret: {exp_ret*100:+.3f}%)"

        return {
            "symbol": symbol,
            "action": action,
            "confidence": prob,
            "expected_return": exp_ret,
            "current_price": current_price,
            "model_version": model_version,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
