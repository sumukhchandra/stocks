"""
Unit Tests for Risk Management Engine and Kill Switch.
"""

import pytest
from datetime import datetime, time
from trading.risk.risk_manager import RiskManager
from trading.risk.kill_switch import KillSwitch


def test_kill_switch_trigger_and_reset():
    ks = KillSwitch(max_daily_drawdown_pct=3.0)
    assert not ks.is_active

    # Trigger
    ks.trigger("Manual test trigger")
    assert ks.is_active
    assert ks.trigger_reason == "Manual test trigger"

    # Reset
    ks.reset()
    assert not ks.is_active
    assert ks.trigger_reason is None


def test_drawdown_circuit_breaker():
    ks = KillSwitch(max_daily_drawdown_pct=3.0)
    # 2% drawdown -> Should not trigger
    triggered = ks.check_drawdown(starting_capital=100000.0, current_capital=98000.0)
    assert not triggered
    assert not ks.is_active

    # 3.5% drawdown -> Should trigger
    triggered = ks.check_drawdown(starting_capital=100000.0, current_capital=96500.0)
    assert triggered
    assert ks.is_active


def test_risk_manager_rejections():
    rm = RiskManager(max_positions=2)
    # Reject when kill switch is active
    rm.kill_switch.trigger("Emergency halt")

    signal = {"symbol": "RELIANCE.NS", "action": "BUY", "expected_return": 0.01}
    portfolio = {"starting_capital": 100000.0, "total_capital": 100000.0, "available_capital": 100000.0, "positions": {}}

    approved, reason, _ = rm.audit_signal(signal, portfolio)
    assert not approved
    assert "Kill Switch ACTIVE" in reason
