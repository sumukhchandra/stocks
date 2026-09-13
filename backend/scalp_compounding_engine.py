"""
Scalp Compounding Engine — Aggressive 5-Minute Intraday Scalping with Compounding.

Core workflow per session (9:20 AM → 3:20 PM IST):
1. Scan 30 stocks every 30 seconds for 5-min return > 1.0% with high ML probability.
2. Pick the #1 ranked candidate and invest 100% of the compounding pool.
3. Set TP +1.2%, SL -0.4%, max hold 15 minutes (3 bars).
4. On profitable exit: roll capital + net_profit into next trade.
5. On SL hit: pause 10 minutes, then resume chain.
6. Stop when: daily net >= 0.5%, 15 trades done, or 2% drawdown.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database.stocks_config import (
    STOCK_SYMBOLS,
    STOCK_UNIVERSE,
    SCALP_MIN_RETURN_PCT,
    SCALP_NET_PROFIT_TARGET,
    SCALP_MIN_PROBABILITY,
    SCALP_MIN_TRADES_PER_SESSION,
    SCALP_MAX_TRADES_PER_SESSION,
    SCALP_MAX_HOLD_BARS,
    SCALP_TP_PCT,
    SCALP_SL_PCT,
    SCALP_CHAIN_STOP_ON_LOSS,
    SCALP_CHAIN_PAUSE_MINUTES,
    SCALP_DRAWDOWN_HALT_PCT,
    COMPOUND_REINVEST_PCT,
    SCALP_SCAN_INTERVAL_SECONDS,
    AUTO_EOD_SQUAREOFF_HOUR,
    AUTO_EOD_SQUAREOFF_MINUTE,
    DEFAULT_CAPITAL,
)
from backend.risk.india_tax_engine import IndiaTaxEngine
from backend.market_data_feed import market_feed

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
SCALP_STATE_FILE = os.path.join(LOGS_DIR, "scalp_session.json")
SCALP_HISTORY_FILE = os.path.join(LOGS_DIR, "scalp_history.jsonl")
os.makedirs(LOGS_DIR, exist_ok=True)


class ScalpCompoundingEngine:
    """
    Manages a single intraday scalp compounding session.
    
    Lifecycle:
        start_session(capital) → scan_and_execute() loop → end_session()
    """

    def __init__(self, capital=None):
        self.tax_engine = IndiaTaxEngine()
        self.model_dir = os.path.abspath(
            os.path.join(SCRIPT_DIR, "..", "ml_models", "models", "saved_models")
        )
        self.engine = None  # Lazy-loaded ML ensemble
        self._load_session(capital)

    # ─── ML Engine ─────────────────────────────────────────────────────

    def _ensure_engine(self):
        """Lazy-load the ML ensemble engine."""
        if self.engine is None:
            try:
                from ml_models.models.final_ensemble_engine import FinalEnsembleEngine
                self.engine = FinalEnsembleEngine(model_dir=self.model_dir)
            except Exception as e:
                print(f"[SCALP ERROR] Could not load models: {e}")
                return False
        return True

    # ─── Session State ─────────────────────────────────────────────────

    def _load_session(self, override_capital=None):
        """Load or create scalp session state."""
        if os.path.exists(SCALP_STATE_FILE):
            try:
                with open(SCALP_STATE_FILE, "r") as f:
                    self.session = json.load(f)
                return
            except Exception:
                pass

        cap = override_capital or DEFAULT_CAPITAL
        self.session = self._create_fresh_session(cap)
        self._save_session()

    def _create_fresh_session(self, capital):
        """Create a new clean session state."""
        return {
            "status": "idle",           # idle | active | paused | completed | halted
            "starting_capital": capital,
            "current_pool": capital,
            "peak_pool": capital,
            "session_start": None,
            "session_end": None,
            "active_position": None,    # Current open trade
            "chain": [],                # Completed trades in this chain
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_net_pnl": 0.0,
            "overall_net_return_pct": 0.0,
            "target_met": False,
            "halt_reason": None,
            "paused_until": None,       # ISO timestamp when chain resumes after loss
            "last_scan": None,
            "last_scan_results": [],    # Top candidates from last scan
        }

    def _save_session(self):
        """Persist session state to disk."""
        with open(SCALP_STATE_FILE, "w") as f:
            json.dump(self.session, f, indent=2, default=str)

    def _reload_session(self):
        """Reload session from disk (for dashboard polling)."""
        if os.path.exists(SCALP_STATE_FILE):
            try:
                with open(SCALP_STATE_FILE, "r") as f:
                    self.session = json.load(f)
            except Exception:
                pass

    def _log_session_history(self):
        """Append completed session summary to history file."""
        with open(SCALP_HISTORY_FILE, "a") as f:
            f.write(json.dumps(self.session, default=str) + "\n")

    # ─── Session Lifecycle ─────────────────────────────────────────────

    def start_session(self, capital=None):
        """
        Initialize a new scalp compounding session.
        Called at ~9:20 AM when market stabilizes after open.
        """
        cap = capital or self.session.get("starting_capital", DEFAULT_CAPITAL)
        self.session = self._create_fresh_session(cap)
        self.session["status"] = "active"
        self.session["session_start"] = datetime.now().isoformat()
        self._save_session()

        print(f"\n{'='*60}")
        print(f"  SCALP COMPOUNDING SESSION STARTED")
        print(f"  Capital: ₹{cap:,.2f}")
        print(f"  Target: {SCALP_NET_PROFIT_TARGET*100:.1f}% net ({SCALP_MIN_TRADES_PER_SESSION}+ trades)")
        print(f"  TP: +{SCALP_TP_PCT*100:.1f}% | SL: -{SCALP_SL_PCT*100:.1f}% | Max Hold: {SCALP_MAX_HOLD_BARS*5}min")
        print(f"{'='*60}\n")

        return {"status": "started", "capital": cap}

    def end_session(self, reason="manual"):
        """End the current session and archive it."""
        self.session["status"] = "completed"
        self.session["session_end"] = datetime.now().isoformat()
        self.session["halt_reason"] = reason
        self._recalculate_session_stats()
        self._save_session()
        self._log_session_history()

        print(f"\n{'='*60}")
        print(f"  SESSION ENDED: {reason}")
        print(f"  Trades: {self.session['total_trades']} | "
              f"W: {self.session['winning_trades']} L: {self.session['losing_trades']}")
        print(f"  Net P&L: ₹{self.session['total_net_pnl']:+,.2f} "
              f"({self.session['overall_net_return_pct']:+.4f}%)")
        print(f"  Final Pool: ₹{self.session['current_pool']:,.2f}")
        print(f"{'='*60}\n")

        return self.get_session_status()

    # ─── Core Scan & Execute Loop ──────────────────────────────────────

    def run_single_scan(self):
        """
        Execute one scan-and-trade cycle:
        1. If holding a position → check exit conditions
        2. If no position → scan all stocks, rank, and enter best candidate
        """
        if not self._ensure_engine():
            return {"status": "error", "message": "Models not loaded"}

        self._reload_session()

        # Gate: session must be active
        if self.session["status"] not in ("active", "paused"):
            return {"status": "inactive", "message": f"Session is {self.session['status']}"}

        # Gate: check if paused (cooldown after loss)
        if self.session["status"] == "paused":
            paused_until = self.session.get("paused_until")
            if paused_until:
                pause_dt = datetime.fromisoformat(paused_until)
                if datetime.now() < pause_dt:
                    remaining = (pause_dt - datetime.now()).seconds
                    return {"status": "paused", "message": f"Cooldown: {remaining}s remaining"}
            # Resume
            self.session["status"] = "active"
            self.session["paused_until"] = None
            print(f"  [SCALP] Cooldown ended. Resuming chain.")

        # Gate: max trades reached
        if self.session["total_trades"] >= SCALP_MAX_TRADES_PER_SESSION:
            self.end_session(reason="max_trades_reached")
            return {"status": "completed", "message": "Max trades per session reached"}

        # Gate: daily target met
        if self.session.get("target_met", False):
            self.end_session(reason="target_achieved")
            return {"status": "completed", "message": "Daily profit target achieved! 🎯"}

        # Gate: drawdown halt
        starting = self.session["starting_capital"]
        current = self.session["current_pool"]
        if starting > 0 and (starting - current) / starting >= SCALP_DRAWDOWN_HALT_PCT:
            self.session["status"] = "halted"
            self.end_session(reason="drawdown_halt")
            return {"status": "halted", "message": f"Pool dropped {SCALP_DRAWDOWN_HALT_PCT*100:.1f}% — session halted"}

        # Gate: EOD auto-exit
        now = datetime.now()
        is_eod = (now.hour == AUTO_EOD_SQUAREOFF_HOUR and now.minute >= AUTO_EOD_SQUAREOFF_MINUTE) or \
                 now.hour > AUTO_EOD_SQUAREOFF_HOUR
        if is_eod:
            # Force exit any open position first
            if self.session.get("active_position"):
                self._force_exit_position("EOD_SQUAREOFF")
            self.end_session(reason="eod_squareoff")
            return {"status": "completed", "message": "End-of-day auto square-off"}

        result = {"status": "ok", "action": None, "candidates": []}

        # STEP 1: Check active position for exit
        if self.session.get("active_position"):
            exit_action = self._check_scalp_exit()
            if exit_action:
                result["action"] = exit_action
                # After exit, immediately scan for next if chain continues
                if self.session["status"] == "active":
                    return result  # Will scan next cycle in 30s

            return result  # Still holding

        # STEP 2: No active position — scan and enter
        candidates = self._scan_scalp_candidates()
        result["candidates"] = candidates[:5]  # Top 5 for display
        self.session["last_scan"] = datetime.now().isoformat()
        self.session["last_scan_results"] = candidates[:10]

        if candidates:
            best = candidates[0]
            entry_action = self._execute_scalp_entry(best)
            if entry_action:
                result["action"] = entry_action

        self._save_session()
        return result

    def _scan_scalp_candidates(self):
        """
        Scan all 30 stocks, run ML batch inference, and return ranked list
        of stocks with predicted 5-min return > SCALP_MIN_RETURN_PCT.
        """
        feature_df = market_feed.get_live_feature_dataset(force_refresh=True)
        if feature_df.empty:
            return []

        # Build stock windows for batch evaluation
        stock_windows = {}
        for symbol in STOCK_SYMBOLS:
            sub = feature_df[feature_df["symbol"] == symbol].sort_values("timestamp")
            if len(sub) >= 20:
                stock_windows[symbol] = sub.tail(50)

        if not stock_windows:
            return []

        # Fast batch ML evaluation
        try:
            batch_decisions = self.engine.evaluate_batch(stock_windows)
        except Exception as e:
            print(f"  [SCALP] Batch eval error: {e}")
            return []

        # Filter and rank
        candidates = []
        for symbol, decision in batch_decisions.items():
            prob = float(decision.get("final_probability", 0))
            exp_ret = float(decision.get("expected_return", 0))
            price = float(decision.get("price", 0))
            confidence = float(decision.get("confidence_score", prob))

            # Gate: minimum probability and return
            if prob < SCALP_MIN_PROBABILITY:
                continue
            if exp_ret < SCALP_MIN_RETURN_PCT:
                continue

            # Check tax viability at current pool capital
            viable, cost_bd = self.tax_engine.is_trade_viable(
                self.session["current_pool"],
                exp_ret,
                min_net_pct=0.0  # We'll check net ourselves
            )

            # Calculate expected net return
            net_ret_pct = cost_bd.get("net_return_pct", 0) / 100.0

            # Score: expected_return × probability × confidence
            score = exp_ret * prob * max(0.5, confidence)

            candidates.append({
                "symbol": symbol,
                "company": STOCK_UNIVERSE.get(symbol, symbol),
                "price": price,
                "probability": prob,
                "expected_return": exp_ret,
                "expected_net_return": net_ret_pct,
                "confidence": confidence,
                "score": score,
                "regime": decision.get("regime", "unknown"),
                "breakdown": decision.get("signals", {}).get("breakdown", {}),
                "cost_breakdown": {
                    "total_cost": cost_bd.get("total_cost", 0),
                    "tax": cost_bd.get("tax", 0),
                    "net_profit": cost_bd.get("net_profit", 0),
                },
            })

        # Sort by composite score (highest first)
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def _execute_scalp_entry(self, candidate):
        """Execute a scalp buy order."""
        symbol = candidate["symbol"]
        price = candidate["price"]
        pool = self.session["current_pool"]

        if pool < 500:
            return None

        # Full pool allocation (scalp mode = 100% into one trade)
        allocation = pool * COMPOUND_REINVEST_PCT
        qty = allocation / price if price > 0 else 0

        tp_price = price * (1 + SCALP_TP_PCT)
        sl_price = price * (1 - SCALP_SL_PCT)

        self.session["active_position"] = {
            "symbol": symbol,
            "company": candidate["company"],
            "entry_price": price,
            "highest_price": price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "qty": qty,
            "invested": allocation,
            "entry_time": datetime.now().isoformat(),
            "entry_bar": 0,  # Track bars held
            "probability": candidate["probability"],
            "expected_return": candidate["expected_return"],
            "confidence": candidate["confidence"],
        }

        # Deduct from pool
        self.session["current_pool"] = 0.0
        self._save_session()

        trade_num = self.session["total_trades"] + 1
        print(
            f"  [SCALP BUY #{trade_num}] {symbol} @ ₹{price:,.2f} | "
            f"Qty: {qty:.2f} | ₹{allocation:,.2f} | "
            f"TP: ₹{tp_price:,.2f} (+{SCALP_TP_PCT*100:.1f}%) | "
            f"SL: ₹{sl_price:,.2f} (-{SCALP_SL_PCT*100:.1f}%) | "
            f"AI: {candidate['probability']:.1%} / Exp: {candidate['expected_return']*100:+.2f}%"
        )

        return {
            "action": "BUY",
            "symbol": symbol,
            "price": price,
            "qty": qty,
            "invested": allocation,
            "tp": tp_price,
            "sl": sl_price,
            "trade_num": trade_num,
        }

    def _check_scalp_exit(self):
        """Check if the active scalp position should be exited."""
        pos = self.session.get("active_position")
        if not pos:
            return None

        symbol = pos["symbol"]
        entry_price = float(pos["entry_price"])
        tp_price = float(pos["tp_price"])
        sl_price = float(pos["sl_price"])
        invested = float(pos["invested"])

        # Get current price
        quotes_df = market_feed.get_realtime_quotes(symbols=[symbol], force_refresh=True)
        if quotes_df.empty:
            return None

        current_price = float(quotes_df.iloc[0]["price"])
        if current_price <= 0:
            return None

        high_price = float(quotes_df.iloc[0].get("high", current_price))
        low_price = float(quotes_df.iloc[0].get("low", current_price))

        # Update watermark
        highest_price = max(float(pos.get("highest_price", entry_price)), current_price, high_price)
        pos["highest_price"] = highest_price

        # Increment bar counter
        pos["entry_bar"] = pos.get("entry_bar", 0) + 1

        gross_return = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

        # EXIT 1: Take Profit
        if current_price >= tp_price or high_price >= tp_price:
            exec_price = max(current_price, tp_price)
            ret = (exec_price - entry_price) / entry_price
            return self._execute_scalp_exit(exec_price, ret, "TAKE_PROFIT")

        # EXIT 2: Stop Loss
        if current_price <= sl_price or low_price <= sl_price:
            exec_price = min(current_price, sl_price)
            ret = (exec_price - entry_price) / entry_price
            return self._execute_scalp_exit(exec_price, ret, "STOP_LOSS")

        # EXIT 3: Max hold time (3 bars = 15 min)
        if pos.get("entry_bar", 0) >= SCALP_MAX_HOLD_BARS:
            return self._execute_scalp_exit(current_price, gross_return, "MAX_HOLD_TIME")

        # Still holding — update live P&L in session
        bd = self.tax_engine.calculate_total_cost(invested, gross_return)
        pos["current_price"] = current_price
        pos["unrealized_gross_pct"] = round(gross_return * 100, 4)
        pos["unrealized_net_pnl"] = bd["net_profit"]
        self._save_session()
        return None

    def _execute_scalp_exit(self, exit_price, gross_return, reason):
        """Execute a scalp sell and compound the result."""
        pos = self.session["active_position"]
        symbol = pos["symbol"]
        invested = float(pos["invested"])

        # Calculate full cost breakdown
        bd = self.tax_engine.calculate_total_cost(invested, gross_return)
        net_profit = bd["net_profit"]
        is_win = net_profit > 0

        # Compound: add capital + net_profit back to pool
        returned = invested + net_profit
        self.session["current_pool"] = returned
        self.session["peak_pool"] = max(self.session["peak_pool"], returned)

        # Track stats
        self.session["total_trades"] += 1
        if is_win:
            self.session["winning_trades"] += 1
        else:
            self.session["losing_trades"] += 1
        self.session["total_net_pnl"] += net_profit

        # Hold duration
        try:
            entry_dt = datetime.fromisoformat(pos["entry_time"])
            hold_secs = int((datetime.now() - entry_dt).total_seconds())
            hold_str = f"{hold_secs // 60}m {hold_secs % 60}s"
        except Exception:
            hold_str = f"{pos.get('entry_bar', 0) * 5}m"

        # Record trade in chain
        trade_record = {
            "trade_num": self.session["total_trades"],
            "symbol": symbol,
            "company": pos.get("company", STOCK_UNIVERSE.get(symbol, symbol)),
            "action": "SELL",
            "reason": reason,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "qty": pos["qty"],
            "invested": invested,
            "gross_return_pct": round(gross_return * 100, 4),
            "gross_profit": bd["gross_profit"],
            "total_cost": bd["total_cost"],
            "tax": bd["tax"],
            "net_profit": round(net_profit, 2),
            "net_return_pct": bd["net_return_pct"],
            "hold_duration": hold_str,
            "pool_after": round(self.session["current_pool"], 2),
            "timestamp": datetime.now().isoformat(),
        }
        self.session["chain"].append(trade_record)

        # Clear active position
        self.session["active_position"] = None

        # Recalculate overall stats
        self._recalculate_session_stats()

        # Check if target met
        if self.session["overall_net_return_pct"] >= SCALP_NET_PROFIT_TARGET * 100:
            self.session["target_met"] = True

        # Handle loss: pause chain if configured
        if not is_win and SCALP_CHAIN_STOP_ON_LOSS:
            pause_until = datetime.now() + timedelta(minutes=SCALP_CHAIN_PAUSE_MINUTES)
            self.session["status"] = "paused"
            self.session["paused_until"] = pause_until.isoformat()
            print(f"  [SCALP] Chain paused for {SCALP_CHAIN_PAUSE_MINUTES}min after stop-loss")

        self._save_session()

        tag = "🟢 WIN" if is_win else "🔴 LOSS"
        print(
            f"  [{tag}] SELL #{self.session['total_trades']} {symbol} @ ₹{exit_price:,.2f} | "
            f"Gross: {gross_return*100:+.2f}% | Net: ₹{net_profit:+,.2f} ({bd['net_return_pct']:+.3f}%) | "
            f"Hold: {hold_str} | Reason: {reason} | "
            f"Pool: ₹{self.session['current_pool']:,.2f}"
        )

        return trade_record

    def _force_exit_position(self, reason="FORCED"):
        """Force exit the active position at current market price."""
        pos = self.session.get("active_position")
        if not pos:
            return None

        symbol = pos["symbol"]
        quotes_df = market_feed.get_realtime_quotes(symbols=[symbol], force_refresh=True)
        if quotes_df.empty:
            price = float(pos["entry_price"])
        else:
            price = float(quotes_df.iloc[0]["price"])

        entry_price = float(pos["entry_price"])
        gross_return = (price - entry_price) / entry_price if entry_price > 0 else 0.0
        return self._execute_scalp_exit(price, gross_return, reason)

    def _recalculate_session_stats(self):
        """Recalculate cumulative session statistics."""
        starting = self.session["starting_capital"]
        current = self.session["current_pool"]
        if self.session.get("active_position"):
            current += float(self.session["active_position"].get("invested", 0))
        net_pnl = current - starting
        net_return = (net_pnl / starting * 100) if starting > 0 else 0.0
        self.session["total_net_pnl"] = round(net_pnl, 2)
        self.session["overall_net_return_pct"] = round(net_return, 4)

    # ─── Dashboard / API Accessors ─────────────────────────────────────

    def get_session_status(self):
        """Get the full session state for dashboard display."""
        self._reload_session()
        s = self.session

        # Calculate win rate
        total = s["total_trades"]
        win_rate = (s["winning_trades"] / total * 100) if total > 0 else 0.0

        # Progress toward target
        target_progress = min(100, (s["overall_net_return_pct"] / (SCALP_NET_PROFIT_TARGET * 100)) * 100) \
            if SCALP_NET_PROFIT_TARGET > 0 else 0.0

        # Active position live P&L
        active_pos = None
        if s.get("active_position"):
            ap = s["active_position"]
            active_pos = {
                "symbol": ap["symbol"],
                "company": ap.get("company", ""),
                "entry_price": ap["entry_price"],
                "current_price": ap.get("current_price", ap["entry_price"]),
                "tp_price": ap["tp_price"],
                "sl_price": ap["sl_price"],
                "unrealized_pct": ap.get("unrealized_gross_pct", 0),
                "unrealized_net_pnl": ap.get("unrealized_net_pnl", 0),
                "bars_held": ap.get("entry_bar", 0),
                "max_bars": SCALP_MAX_HOLD_BARS,
                "probability": ap.get("probability", 0),
                "expected_return": ap.get("expected_return", 0),
            }

        return {
            "status": s["status"],
            "starting_capital": s["starting_capital"],
            "current_pool": s["current_pool"],
            "total_net_pnl": s["total_net_pnl"],
            "overall_net_return_pct": s["overall_net_return_pct"],
            "total_trades": total,
            "winning_trades": s["winning_trades"],
            "losing_trades": s["losing_trades"],
            "win_rate": round(win_rate, 1),
            "target_met": s.get("target_met", False),
            "target_progress": round(target_progress, 1),
            "session_start": s.get("session_start"),
            "halt_reason": s.get("halt_reason"),
            "paused_until": s.get("paused_until"),
            "active_position": active_pos,
            "chain": s.get("chain", [])[-15:],  # Last 15 trades
            "last_scan_results": s.get("last_scan_results", [])[:5],
            "config": {
                "min_return": SCALP_MIN_RETURN_PCT * 100,
                "min_probability": SCALP_MIN_PROBABILITY * 100,
                "tp_pct": SCALP_TP_PCT * 100,
                "sl_pct": SCALP_SL_PCT * 100,
                "max_hold_mins": SCALP_MAX_HOLD_BARS * 5,
                "max_trades": SCALP_MAX_TRADES_PER_SESSION,
                "net_target": SCALP_NET_PROFIT_TARGET * 100,
            },
        }

    def get_session_history(self):
        """Read all completed scalp session summaries."""
        sessions = []
        if os.path.exists(SCALP_HISTORY_FILE):
            with open(SCALP_HISTORY_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            sessions.append(json.loads(line))
                        except Exception:
                            pass
        return sessions

    # ─── Continuous Loop ───────────────────────────────────────────────

    def run_session_loop(self, capital=None):
        """
        Run the full scalp compounding session in a loop.
        Call this from the trading dashboard or API.
        """
        self.start_session(capital)

        while self.session.get("status") in ("active", "paused"):
            try:
                now = datetime.now()

                # Only trade during market hours
                if now.hour < 9 or (now.hour == 9 and now.minute < 20):
                    print(f"  [SCALP] Waiting for market to stabilize (9:20 AM)...")
                    time.sleep(30)
                    continue

                if now.hour > 15 or (now.hour == 15 and now.minute > 20):
                    self._force_exit_position("EOD_SQUAREOFF")
                    self.end_session("eod_squareoff")
                    break

                result = self.run_single_scan()

                if result.get("status") in ("completed", "halted"):
                    break

                # Reload to check if user stopped session
                self._reload_session()
                if self.session.get("status") not in ("active", "paused"):
                    break

                time.sleep(SCALP_SCAN_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                self._force_exit_position("KEYBOARD_INTERRUPT")
                self.end_session("keyboard_interrupt")
                break
            except Exception as e:
                print(f"  [SCALP ERROR] {e}")
                time.sleep(30)

        return self.get_session_status()


# Singleton instance
scalp_engine = ScalpCompoundingEngine()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NSE Scalp Compounding Engine")
    parser.add_argument("--capital", type=float, default=None, help="Starting capital (INR)")
    parser.add_argument("--single", action="store_true", help="Run single scan only")
    args = parser.parse_args()

    engine = ScalpCompoundingEngine(capital=args.capital)
    if args.single:
        engine.start_session(args.capital)
        result = engine.run_single_scan()
        print(json.dumps(result, indent=2, default=str))
    else:
        engine.run_session_loop(args.capital)
