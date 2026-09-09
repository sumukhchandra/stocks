"""
backend package: Core trading execution, risk management, and market monitoring.
Uses lazy attribute loading to eliminate circular dependencies across data and models.
"""
import sys

# Ensure backend submodules are importable both directly and via backend namespace
import backend.risk as _risk
import backend.execution as _execution
import backend.monitoring as _monitoring

sys.modules.setdefault('risk', _risk)
sys.modules.setdefault('execution', _execution)
sys.modules.setdefault('monitoring', _monitoring)

def __getattr__(name):
    if name == 'NSEAutoTrader':
        from backend.nse_auto_trader import NSEAutoTrader
        return NSEAutoTrader
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['NSEAutoTrader']
