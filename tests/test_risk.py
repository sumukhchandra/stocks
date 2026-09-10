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


def test_dynamic_trailing_stop_ratchet():
    from database.stocks_config import (
        TRAILING_STOP_ACTIVATION_PCT,
        TRAILING_STOP_DISTANCE_PCT,
        MAX_GROSS_LOSS_PCT,
        PROFIT_TARGET_PCT,
    )

    entry_price = 1000.0
    initial_sl = entry_price * (1 - MAX_GROSS_LOSS_PCT)
    initial_tp = entry_price * (1 + PROFIT_TARGET_PCT)

    assert initial_sl == 992.0
    assert initial_tp == 1020.0

    # Before activation: highest reaches 1005 (+0.50% gain)
    highest_price = 1005.0
    gain_from_entry = (highest_price - entry_price) / entry_price
    assert gain_from_entry < TRAILING_STOP_ACTIVATION_PCT

    # After activation: highest reaches 1010 (+1.0% gain)
    highest_price = 1010.0
    gain_from_entry = (highest_price - entry_price) / entry_price
    assert gain_from_entry >= TRAILING_STOP_ACTIVATION_PCT

    breakeven_price = entry_price * 1.0025  # 1002.5
    trail_price = highest_price * (1 - TRAILING_STOP_DISTANCE_PCT)  # 1010 * 0.995 = 1004.95
    new_sl = max(initial_sl, trail_price, breakeven_price)

    assert new_sl > entry_price  # Locked-in profit above entry!
    assert new_sl == pytest.approx(1004.95, rel=1e-3)

