"""
NSE Automated Stock Trader -- Paper Trading Engine with Compounding.

Core behaviours:
1. Fetches live 5m data for all configured NSE stocks via yfinance.
2. Runs the ensemble engine to get predicted probability + expected return.
3. Only buys when predicted net return > 0.5% after ALL costs (Zerodha rates).
4. Sells when realised net profit > 0.5% OR gross loss > 0.3% (stop-loss).
5. Compounding: after every trade closes, net profit/loss is added to capital.
6. Multiple simultaneous positions allowed with dynamic confidence-based sizing.
7. All state and trades persisted to JSON files.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database.stocks_config import (
    STOCK_SYMBOLS,
    STOCK_UNIVERSE,
    MIN_NET_PROFIT_PCT,
    MAX_GROSS_LOSS_PCT,
    DEFAULT_CAPITAL,
    POSITION_SIZING,
    SCAN_INTERVAL_SECONDS,
    ACCURACY_RETRAIN_THRESHOLD,
    ROLLING_ACCURACY_WINDOW,
)
import ml_models.validation
import ml_models.validation.purged_cv
from backend.risk.india_tax_engine import IndiaTaxEngine
from data.fetch_stock_data import add_stock_features, _flatten_columns, fetch_nifty_index_features
from database.db_manager import DatabaseManager
from database.rules import TradingRulesEngine


# -- File paths ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
STATE_FILE = os.path.join(LOGS_DIR, "trader_state.json")
TRADE_LOG_FILE = os.path.join(LOGS_DIR, "trade_history.jsonl")
RETRAIN_LOG_FILE = os.path.join(LOGS_DIR, "retrain_log.jsonl")

os.makedirs(LOGS_DIR, exist_ok=True)


class NSEAutoTrader:
    """Paper trading engine for NSE stocks with compounding."""

    def __init__(self, capital=None):
        self.tax_engine = IndiaTaxEngine()
        self.rules_engine = TradingRulesEngine()
        self.db = DatabaseManager()
        self.model_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ml_models", "models", "saved_models"))
        self.engine = None  # Lazy-loaded
        self._load_state(capital)

    # -- State Management -----------------------------------------------------

    def _reload_state(self):
        """Reload latest persisted state from disk."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _load_state(self, override_capital=None):
        """Load persisted state or initialise fresh."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}

            # Allow capital override on fresh start
            if override_capital is not None and not self.state.get("positions"):
                self.state["total_capital"] = override_capital
                self.state["available_capital"] = override_capital
                self._save_state()
        else:
            cap = override_capital or DEFAULT_CAPITAL
            self.state = {
                "total_capital": cap,
                "available_capital": cap,
                "starting_capital": cap,
                "positions": {},
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "is_trading": False,
                "last_scan": None,
                "last_retrain": None,
                "recent_outcomes": [],  # List of True/False for rolling accuracy
            }
            self._save_state()

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        try:
            if hasattr(self, 'db') and self.db:
                self.db.sync_account(
                    total_capital=float(self.state.get("total_capital", 100000)),
                    available_capital=float(self.state.get("available_capital", 100000)),
                    starting_capital=float(self.state.get("starting_capital", 100000)),
                    total_net_pnl=float(self.state.get("total_net_pnl", 0)),
                    total_trades=int(self.state.get("total_trades", 0)),
                    winning_trades=int(self.state.get("winning_trades", 0)),
                    losing_trades=int(self.state.get("losing_trades", 0)),
                )
        except Exception:
            pass

    def _log_trade(self, trade_record):
        """Append a single trade to the JSONL trade log and database."""
        with open(TRADE_LOG_FILE, "a") as f:
            f.write(json.dumps(trade_record, default=str) + "\n")
        try:
            if hasattr(self, 'db') and self.db:
                self.db.record_trade(trade_record)
        except Exception:
            pass

    def get_trade_history(self):
        """Read all trade records from the log file."""
        trades = []
        if os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trades.append(json.loads(line))
        return trades

    # -- Model Loading --------------------------------------------------------

    def _ensure_engine(self):
        """Lazy-load or reload the model engine."""
        if self.engine is None:
            try:
                from ml_models.models.final_ensemble_engine import FinalEnsembleEngine
                self.engine = FinalEnsembleEngine(model_dir=self.model_dir)
            except Exception as e:
                print(f"[ERROR] Could not load models: {e}")
                return False
        return True

    def reload_models(self):
        """Force reload models after retrain."""
        self.engine = None
        return self._ensure_engine()

    # -- Data Fetching --------------------------------------------------------

    def _fetch_live_data(self, symbols=None):
        """Fetch recent 5m candle data for all stocks with feature engineering."""
        symbols = symbols or STOCK_SYMBOLS
        raw_frames = []
        for symbol in symbols:
            try:
                df = yf.download(
                    symbol,
                    period="5d",
                    interval="5m",
                    auto_adjust=False,
                    progress=False,
                    group_by="column",
                    threads=False,
                )
                if df.empty:
                    continue
                df = _flatten_columns(df.reset_index())
                time_col = (
                    "datetime" if "datetime" in df.columns
                    else "date" if "date" in df.columns
                    else "index"
                )
                df = df.rename(columns={time_col: "timestamp", "adj close": "adj_close"})
                required = ["timestamp", "open", "high", "low", "close", "volume"]
                if any(c not in df.columns for c in required):
                    continue
                df = df[required].copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                df["symbol"] = symbol
                df = df.dropna(subset=["open", "high", "low", "close"])
                raw_frames.append(df)
            except Exception:
                continue

        if not raw_frames:
            return pd.DataFrame()

        combined = pd.concat(raw_frames, ignore_index=True)
        nifty_df = fetch_nifty_index_features(period="5d", interval="5m")
        return add_stock_features(combined, nifty_df=nifty_df)

    # -- Position Sizing ------------------------------------------------------

    def _get_allocation(self, confidence, min_confidence=0.50):
        """Dynamic allocation based on model confidence."""
        available = self.state["available_capital"]
        if confidence > 0.8:
            pct = POSITION_SIZING["high"]
        elif confidence > 0.7:
            pct = POSITION_SIZING["medium"]
        elif confidence > 0.6:
            pct = POSITION_SIZING["low"]
        elif confidence >= min_confidence:
            pct = 0.10  # 10% allocation for moderate confidence
        else:
            return 0  # Below minimum confidence
        return available * pct

    # -- Core Trading Logic ---------------------------------------------------

    def run_single_cycle(self, min_confidence=0.50):
        """Execute one full scan-and-trade cycle."""
        if not self._ensure_engine():
            return {"status": "error", "message": "Models not loaded"}

        self.state["last_scan"] = datetime.now().isoformat()
        feature_df = self._fetch_live_data()
        if feature_df.empty:
            self._save_state()
            return {"status": "no_data", "message": "No market data available"}

        results = {"status": "ok", "signals": [], "actions": []}

        # 1. CHECK EXISTING POSITIONS -- sell if profitable or stop-loss
        for symbol in list(self.state["positions"].keys()):
            action = self._check_exit(symbol, feature_df)
            if action:
                results["actions"].append(action)

        # 2. SCAN FOR NEW ENTRIES
        for symbol in STOCK_SYMBOLS:
            if symbol in self.state["positions"]:
                continue  # Already holding
            if self.state["available_capital"] < 1000:
                break  # Not enough capital

            signal = self._evaluate_entry(symbol, feature_df, min_confidence=min_confidence)
            results["signals"].append(signal)
            if signal.get("action") == "BUY":
                results["actions"].append(signal)

        self.state["last_signals"] = results["signals"]
        self._save_state()
        return results

    def simulate_intraday_session(self, n_days=3, min_confidence=0.50):
        """Simulate recent trading sessions candle-by-candle with compounding and real tax engine."""
        if not self._ensure_engine():
            return {"status": "error", "message": "Models not loaded"}
            
        data_path = os.path.join(ROOT_DIR, "data", "processed", "master_labeled_dataset.parquet")
        if os.path.exists(data_path):
            feature_df = pd.read_parquet(data_path)
        else:
            feature_df = self._fetch_live_data()
            
        if feature_df.empty:
            return {"status": "no_data", "message": "No dataset found"}
                
        # Filter to recent n_days
        timestamps = sorted(feature_df["timestamp"].unique())
        if len(timestamps) > 75 * n_days:
            cutoff = timestamps[-75 * n_days]
            df_subset = feature_df[feature_df["timestamp"] >= cutoff].copy()
        else:
            df_subset = feature_df.copy()
            
        # Fast Vectorized Pre-computations across all models
        X_vec = df_subset.reindex(columns=self.engine.primary_features, fill_value=0)
        df_subset["pred_prob"] = self.engine.predict_universal(X_vec)
        df_subset["pred_ret"] = self.engine.predict_return_ensemble(X_vec)
        
        executed_trades = []
        # Group by timestamp chronologically
        for ts, group in df_subset.groupby("timestamp", sort=True):
            price_map = group.set_index("symbol")["close"].to_dict()
            
            # 1. Check exits
            for symbol in list(self.state["positions"].keys()):
                if symbol not in price_map:
                    continue
                curr_price = float(price_map[symbol])
                pos = self.state["positions"][symbol]
                gross_return = (curr_price - pos["entry_price"]) / pos["entry_price"]
                
                # Stop loss
                if gross_return < -MAX_GROSS_LOSS_PCT:
                    action = self._execute_sell(symbol, curr_price, gross_return, "STOP_LOSS", exit_time=ts)
                    executed_trades.append(action)
                # Take profit
                elif gross_return > 0:
                    is_profitable, bd = self.tax_engine.is_exit_profitable(
                        pos["invested"], gross_return, min_net_pct=MIN_NET_PROFIT_PCT
                    )
                    if is_profitable:
                        action = self._execute_sell(symbol, curr_price, gross_return, "TAKE_PROFIT", bd, exit_time=ts)
                        executed_trades.append(action)
                        
            # 2. Check entries
            for _, row in group.iterrows():
                symbol = row["symbol"]
                if symbol in self.state["positions"]:
                    continue
                if self.state["available_capital"] < 1000:
                    break
                    
                prob = float(row["pred_prob"])
                exp_ret = float(row["pred_ret"])
                curr_price = float(row["close"])
                
                if prob >= min_confidence and exp_ret > 0.0005:
                    alloc = self._get_allocation(prob, min_confidence=min_confidence)
                    if alloc > 0:
                        qty = alloc / curr_price
                        self.state["positions"][symbol] = {
                            "entry_price": curr_price,
                            "qty": qty,
                            "invested": alloc,
                            "entry_time": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                            "confidence": prob,
                            "predicted_return": exp_ret,
                        }
                        self.state["available_capital"] -= alloc
                        trade_rec = {
                            "action": "BUY",
                            "symbol": symbol,
                            "company": STOCK_UNIVERSE.get(symbol, symbol),
                            "price": curr_price,
                            "qty": qty,
                            "invested": alloc,
                            "confidence": prob,
                            "predicted_return": exp_ret,
                            "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                        }
                        self._log_trade(trade_rec)
                        executed_trades.append(trade_rec)
                            
        self.state["last_scan"] = datetime.now().isoformat()
        self._save_state()
        return {
            "status": "ok",
            "trades_count": len(executed_trades),
            "trades": executed_trades,
            "summary": self.get_portfolio_summary()
        }

    def _evaluate_entry(self, symbol, feature_df, min_confidence=0.50):
        """Evaluate whether to buy a stock."""
        symbol_data = feature_df[feature_df["symbol"] == symbol].sort_values("timestamp")
        if len(symbol_data) < 50:
            return {"symbol": symbol, "action": "SKIP", "reason": "insufficient_data"}

        window = symbol_data.tail(50)
        current_price = float(window["close"].iloc[-1])

        try:
            decision = self.engine.evaluate_state(window)
        except Exception as e:
            return {"symbol": symbol, "action": "SKIP", "reason": f"model_error: {e}"}

        prob = float(decision["final_probability"])
        expected_return = float(decision.get("expected_return", 0))
        confidence = prob  # Use probability as confidence proxy

        signal = {
            "symbol": symbol,
            "company": STOCK_UNIVERSE.get(symbol, symbol),
            "price": current_price,
            "probability": prob,
            "expected_return": expected_return,
            "regime": decision.get("regime", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }

        # Gate 1: Minimum confidence
        if confidence < min_confidence:
            signal["action"] = "SKIP"
            signal["reason"] = f"low_confidence ({prob:.2%})"
            return signal

        # Gate 2: Check if predicted return is viable after all costs
        allocation = self._get_allocation(confidence, min_confidence=min_confidence)
        if allocation <= 0:
            signal["action"] = "SKIP"
            signal["reason"] = "zero_allocation"
            return signal

        viable, cost_breakdown = self.tax_engine.is_trade_viable(
            allocation, expected_return, min_net_pct=MIN_NET_PROFIT_PCT
        )
        signal["cost_breakdown"] = cost_breakdown

        if not viable:
            signal["action"] = "SKIP"
            signal["reason"] = (
                f"not_viable_after_costs (net: {cost_breakdown['net_return_pct']:.4f}%, "
                f"need: {MIN_NET_PROFIT_PCT*100:.1f}%)"
            )
            return signal

        # [OK] ALL GATES PASSED -- EXECUTE BUY
        qty = allocation / current_price
        self.state["positions"][symbol] = {
            "entry_price": current_price,
            "qty": qty,
            "invested": allocation,
            "entry_time": datetime.now().isoformat(),
            "confidence": confidence,
            "predicted_return": expected_return,
        }
        self.state["available_capital"] -= allocation

        signal["action"] = "BUY"
        signal["qty"] = qty
        signal["invested"] = allocation
        signal["reason"] = (
            f"viable (predicted net: {cost_breakdown['net_return_pct']:.4f}%, "
            f"confidence: {confidence:.2%})"
        )

        self._log_trade({
            "action": "BUY",
            "symbol": symbol,
            "company": STOCK_UNIVERSE.get(symbol, symbol),
            "price": current_price,
            "qty": qty,
            "invested": allocation,
            "confidence": confidence,
            "predicted_return": expected_return,
            "timestamp": datetime.now().isoformat(),
        })

        print(f"   BUY {symbol} @ Rs.{current_price:.2f} | Qty: {qty:.2f} | "
              f"Invested: Rs.{allocation:.2f} | Confidence: {confidence:.2%}")

        return signal

    def _check_exit(self, symbol, feature_df):
        """Check if an existing position should be sold."""
        pos = self.state["positions"].get(symbol)
        if not pos:
            return None

        symbol_data = feature_df[feature_df["symbol"] == symbol].sort_values("timestamp")
        if symbol_data.empty:
            return None

        current_price = float(symbol_data["close"].iloc[-1])
        entry_price = pos["entry_price"]
        gross_return = (current_price - entry_price) / entry_price

        # Check STOP-LOSS: exit if gross loss exceeds threshold
        if gross_return < -MAX_GROSS_LOSS_PCT:
            return self._execute_sell(symbol, current_price, gross_return, "STOP_LOSS")

        # Check TAKE-PROFIT: exit if net profit > 0.5% after all costs
        if gross_return > 0:
            is_profitable, breakdown = self.tax_engine.is_exit_profitable(
                pos["invested"], gross_return, min_net_pct=MIN_NET_PROFIT_PCT
            )
            if is_profitable:
                return self._execute_sell(
                    symbol, current_price, gross_return, "TAKE_PROFIT", breakdown
                )

        return None

    def _execute_sell(self, symbol, current_price, gross_return, reason, breakdown=None, exit_time=None):
        """Execute a sell and apply compounding."""
        pos = self.state["positions"][symbol]

        if breakdown is None:
            breakdown = self.tax_engine.calculate_total_cost(pos["invested"], gross_return)

        net_profit = breakdown["net_profit"]
        invested = pos["invested"]

        # -- COMPOUNDING: add net profit back to capital --
        returned_capital = invested + net_profit
        self.state["available_capital"] += returned_capital
        self.state["total_capital"] = (
            self.state["available_capital"]
            + sum(p["invested"] for p in self.state["positions"].values() if p != pos)
        )

        # Track win/loss
        is_win = net_profit > 0
        self.state["total_trades"] += 1
        if is_win:
            self.state["winning_trades"] += 1
        else:
            self.state["losing_trades"] += 1

        # Rolling accuracy for auto-retrain trigger
        predicted_up = pos.get("predicted_return", 0) > 0
        actual_up = gross_return > 0
        self.state["recent_outcomes"].append(predicted_up == actual_up)
        if len(self.state["recent_outcomes"]) > ROLLING_ACCURACY_WINDOW:
            self.state["recent_outcomes"] = self.state["recent_outcomes"][
                -ROLLING_ACCURACY_WINDOW:
            ]

        # Remove position
        del self.state["positions"][symbol]

        # Recalculate total capital after position removal
        invested_in_positions = sum(
            p["invested"] for p in self.state["positions"].values()
        )
        self.state["total_capital"] = self.state["available_capital"] + invested_in_positions

        if exit_time is None:
            exit_dt = datetime.now()
        elif isinstance(exit_time, str):
            try:
                exit_dt = datetime.fromisoformat(exit_time)
            except Exception:
                exit_dt = datetime.now()
        elif hasattr(exit_time, 'to_pydatetime'):
            exit_dt = exit_time.to_pydatetime()
        elif isinstance(exit_time, datetime):
            exit_dt = exit_time
        else:
            exit_dt = datetime.now()

        hold_duration = "15m"
        try:
            raw_entry = pos.get("entry_time")
            if isinstance(raw_entry, str):
                entry_dt = datetime.fromisoformat(raw_entry)
            elif hasattr(raw_entry, 'to_pydatetime'):
                entry_dt = raw_entry.to_pydatetime()
            elif isinstance(raw_entry, datetime):
                entry_dt = raw_entry
            else:
                entry_dt = exit_dt

            delta = exit_dt - entry_dt
            total_seconds = int(delta.total_seconds())
            if total_seconds < 0:
                total_seconds = abs(total_seconds)
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            minutes = (total_seconds % 3600) // 60

            if days > 0:
                hold_duration = f"{days}d {hours}h {minutes}m"
            elif hours > 0:
                hold_duration = f"{hours}h {minutes}m"
            elif minutes > 0:
                hold_duration = f"{minutes}m"
            else:
                hold_duration = f"{total_seconds}s"
        except Exception:
            pass

        trade_record = {
            "action": "SELL",
            "reason": reason,
            "symbol": symbol,
            "company": STOCK_UNIVERSE.get(symbol, symbol),
            "entry_price": pos["entry_price"],
            "exit_price": current_price,
            "qty": pos["qty"],
            "invested": invested,
            "gross_return_pct": round(gross_return * 100, 4),
            "gross_profit": breakdown["gross_profit"],
            "total_cost": breakdown["total_cost"],
            "brokerage": breakdown["brokerage"],
            "stt": breakdown["stt"],
            "gst": breakdown["gst"],
            "tax": breakdown["tax"],
            "net_profit": net_profit,
            "net_return_pct": breakdown["net_return_pct"],
            "capital_after": round(self.state["total_capital"], 2),
            "hold_duration": hold_duration,
            "timestamp": exit_dt.isoformat(),
        }

        self._log_trade(trade_record)

        tag = "[WIN]" if is_win else "[LOSS]"
        print(
            f"  {tag} SELL {symbol} @ Rs.{current_price:.2f} | "
            f"Gross: {gross_return*100:+.2f}% | Net P&L: Rs.{net_profit:+.2f} | "
            f"Hold: {hold_duration} | Reason: {reason} | Capital: Rs.{self.state['total_capital']:.2f}"
        )

        return trade_record

    # -- Performance Metrics --------------------------------------------------

    def get_rolling_accuracy(self):
        """Returns rolling accuracy over recent trades."""
        outcomes = self.state.get("recent_outcomes", [])
        if not outcomes:
            return 1.0  # No trades yet, assume good
        return sum(outcomes) / len(outcomes)

    def needs_retrain(self):
        """Check if model accuracy has dropped and retrain is needed."""
        if len(self.state.get("recent_outcomes", [])) < ROLLING_ACCURACY_WINDOW:
            return False  # Not enough data
        return self.get_rolling_accuracy() < ACCURACY_RETRAIN_THRESHOLD

    def get_portfolio_summary(self):
        """Returns a summary dict for the dashboard."""
        self._reload_state()
        total_invested = sum(
            p["invested"] for p in self.state.get("positions", {}).values()
        )
        return {
            "total_capital": round(self.state["total_capital"], 2),
            "available_capital": round(self.state["available_capital"], 2),
            "invested_capital": round(total_invested, 2),
            "starting_capital": self.state.get("starting_capital", DEFAULT_CAPITAL),
            "total_net_pnl": round(
                self.state["total_capital"]
                - self.state.get("starting_capital", DEFAULT_CAPITAL),
                2,
            ),
            "total_trades": self.state["total_trades"],
            "winning_trades": self.state["winning_trades"],
            "losing_trades": self.state["losing_trades"],
            "win_rate": (
                round(
                    self.state["winning_trades"] / self.state["total_trades"] * 100, 1
                )
                if self.state["total_trades"] > 0
                else 0.0
            ),
            "rolling_accuracy": round(self.get_rolling_accuracy() * 100, 1),
            "active_positions": len(self.state["positions"]),
            "is_trading": self.state.get("is_trading", False),
            "last_scan": self.state.get("last_scan"),
            "last_retrain": self.state.get("last_retrain"),
        }

    def get_live_trade_feed(self, price_map=None):
        """
        Compiles a comprehensive live chronological audit feed of:
        - Currently HOLDING active positions (re-evaluated live against current market prices).
        - Executed BUY and SELL transactions from trade history.
        - Latest AI scan signals.
        """
        self._reload_state()
        price_map = price_map or {}
        feed = []
        now = datetime.now()

        # 1. Active HOLDING positions
        for symbol, pos in self.state.get("positions", {}).items():
            entry_p = float(pos.get("entry_price", 0))
            curr_p = float(price_map.get(symbol, entry_p))
            qty = float(pos.get("qty", 0))
            invested = float(pos.get("invested", 0))

            gross_ret = (curr_p - entry_p) / entry_p if entry_p > 0 else 0.0
            bd = self.tax_engine.calculate_total_cost(invested, gross_ret)

            # Hold duration
            try:
                raw_time = pos.get("entry_time")
                if isinstance(raw_time, str):
                    entry_dt = datetime.fromisoformat(raw_time)
                elif hasattr(raw_time, 'to_pydatetime'):
                    entry_dt = raw_time.to_pydatetime()
                elif isinstance(raw_time, datetime):
                    entry_dt = raw_time
                else:
                    entry_dt = now
                diff = now - entry_dt
                hold_mins = int(diff.total_seconds() / 60)
                if hold_mins >= 60:
                    hold_str = f"{hold_mins // 60}h {hold_mins % 60}m"
                else:
                    hold_str = f"{hold_mins}m"
            except Exception:
                hold_str = "Active"

            tp_price = entry_p * 1.015  # 1.5% TP
            sl_price = entry_p * (1 - MAX_GROSS_LOSS_PCT)

            dist_to_tp = ((tp_price - curr_p) / curr_p) * 100
            dist_to_sl = ((curr_p - sl_price) / curr_p) * 100

            note = f"Trailing to TP ({dist_to_tp:+.2f}% away) • SL Buffer: {dist_to_sl:.2f}%"

            feed.append({
                "Timestamp": pos.get("entry_time", now.isoformat())[:19].replace("T", " "),
                "Symbol": symbol.replace(".NS", ""),
                "Company": STOCK_UNIVERSE.get(symbol, symbol),
                "Action / Status": "🛡 HOLDING",
                "Entry Price": f"₹{entry_p:,.2f}",
                "Current / Exit": f"₹{curr_p:,.2f}",
                "Qty": f"{qty:.2f}",
                "Invested": f"₹{invested:,.2f}",
                "Gross P&L": f"₹{bd['gross_profit']:+,.2f}",
                "Net P&L (Post-Tax)": f"₹{bd['net_profit']:+,.2f}",
                "Net ROI %": f"{bd['net_return_pct']:+.3f}%",
                "Fees & Tax": f"₹{bd['total_cost'] + bd['tax']:,.2f}",
                "Hold Time": hold_str,
                "Target (TP)": f"₹{tp_price:,.2f} (+1.5%)",
                "Stop Loss (SL)": f"₹{sl_price:,.2f} (-0.8%)",
                "AI Conf": f"{pos.get('confidence', 0.8)*100:.1f}%",
                "Details / Reason": note,
                "_status_code": "HOLDING",
                "_sort_ts": str(pos.get("entry_time", now.isoformat())),
            })

        # 2. Executed BUY and SELL transactions
        history = self.get_trade_history()
        for t in history[-50:]:
            action = t.get("action", "")
            reason = t.get("reason", "")
            ts_str = str(t.get("timestamp", now.isoformat()))[:19].replace("T", " ")
            sym = t.get("symbol", "")

            if action == "BUY":
                p = float(t.get("price", t.get("entry_price", 0)))
                inv = float(t.get("invested", 0))
                qty = float(t.get("qty", 0))
                conf = float(t.get("confidence", 0))
                feed.append({
                    "Timestamp": ts_str,
                    "Symbol": sym.replace(".NS", ""),
                    "Company": t.get("company", STOCK_UNIVERSE.get(sym, sym)),
                    "Action / Status": "🟢 BOUGHT",
                    "Entry Price": f"₹{p:,.2f}",
                    "Current / Exit": f"₹{p:,.2f}",
                    "Qty": f"{qty:.2f}",
                    "Invested": f"₹{inv:,.2f}",
                    "Gross P&L": "—",
                    "Net P&L (Post-Tax)": "—",
                    "Net ROI %": "—",
                    "Fees & Tax": "—",
                    "Hold Time": "Entry",
                    "Target (TP)": f"₹{p*1.015:,.2f} (+1.5%)",
                    "Stop Loss (SL)": f"₹{p*(1-MAX_GROSS_LOSS_PCT):,.2f} (-0.8%)",
                    "AI Conf": f"{conf*100:.1f}%",
                    "Details / Reason": "Buy order executed via AI Ensemble Signal",
                    "_status_code": "BOUGHT",
                    "_sort_ts": str(t.get("timestamp", now.isoformat())),
                })
            elif action == "SELL":
                entry_p = float(t.get("entry_price", 0))
                exit_p = float(t.get("exit_price", 0))
                qty = float(t.get("qty", 0))
                inv = float(t.get("invested", 0))
                net_p = float(t.get("net_profit", 0))
                gross_p = float(t.get("gross_profit", 0))
                fees = float(t.get("total_cost", 0)) + float(t.get("tax", 0))
                roi = float(t.get("net_return_pct", 0))

                badge = "🔴 SOLD (TP)" if reason == "TAKE_PROFIT" else "🔴 SOLD (SL)" if reason == "STOP_LOSS" else f"🔴 SOLD ({reason})"

                feed.append({
                    "Timestamp": ts_str,
                    "Symbol": sym.replace(".NS", ""),
                    "Company": t.get("company", STOCK_UNIVERSE.get(sym, sym)),
                    "Action / Status": badge,
                    "Entry Price": f"₹{entry_p:,.2f}",
                    "Current / Exit": f"₹{exit_p:,.2f}",
                    "Qty": f"{qty:.2f}",
                    "Invested": f"₹{inv:,.2f}",
                    "Gross P&L": f"₹{gross_p:+,.2f}",
                    "Net P&L (Post-Tax)": f"₹{net_p:+,.2f}",
                    "Net ROI %": f"{roi:+.3f}%",
                    "Fees & Tax": f"₹{fees:,.2f}",
                    "Hold Time": t.get("hold_duration", "Closed"),
                    "Target (TP)": f"₹{entry_p*1.015:,.2f}",
                    "Stop Loss (SL)": f"₹{entry_p*(1-MAX_GROSS_LOSS_PCT):,.2f}",
                    "AI Conf": "Exited",
                    "Details / Reason": f"Exit triggered: {reason.replace('_', ' ').title()}",
                    "_status_code": "SOLD",
                    "_sort_ts": str(t.get("timestamp", now.isoformat())),
                })

        # 3. Latest AI Scan signals
        for s in self.state.get("last_signals", []):
            sym = s.get("symbol", "")
            if sym in self.state.get("positions", {}):
                continue
            act = s.get("action", "SKIP")
            ts_str = str(s.get("timestamp", now.isoformat()))[:19].replace("T", " ")
            p = float(s.get("price", 0))
            prob = float(s.get("probability", 0))
            exp_ret = float(s.get("expected_return", 0))
            reason = s.get("reason", "")

            badge = "🟢 BUY SIGNAL" if act == "BUY" else "📡 SCAN (WAIT)"

            feed.append({
                "Timestamp": ts_str,
                "Symbol": sym.replace(".NS", ""),
                "Company": STOCK_UNIVERSE.get(sym, sym),
                "Action / Status": badge,
                "Entry Price": "—",
                "Current / Exit": f"₹{p:,.2f}",
                "Qty": "—",
                "Invested": "—",
                "Gross P&L": "—",
                "Net P&L (Post-Tax)": "—",
                "Net ROI %": f"Exp: {exp_ret*100:+.2f}%",
                "Fees & Tax": "—",
                "Hold Time": "Scan",
                "Target (TP)": f"₹{p*1.015:,.2f}",
                "Stop Loss (SL)": f"₹{p*(1-MAX_GROSS_LOSS_PCT):,.2f}",
                "AI Conf": f"{prob*100:.1f}%",
                "Details / Reason": reason.replace("_", " ").title() if reason else "Evaluated in live cycle",
                "_status_code": "SCAN",
                "_sort_ts": str(s.get("timestamp", now.isoformat())),
            })

        # Sort descending by sort timestamp
        feed.sort(key=lambda x: x.get("_sort_ts", ""), reverse=True)
        return feed

    def set_capital(self, amount):
        """Reset capital (only when no positions are open)."""
        if self.state["positions"]:
            return False, "Cannot reset capital while positions are open"
        self.state["total_capital"] = amount
        self.state["available_capital"] = amount
        self.state["starting_capital"] = amount
        self._save_state()
        return True, f"Capital set to Rs.{amount:,.2f}"

    def reset_all(self, capital=None):
        """Full reset: clear all positions, trade history, and start fresh."""
        cap = capital or DEFAULT_CAPITAL
        self.state = {
            "total_capital": cap,
            "available_capital": cap,
            "starting_capital": cap,
            "positions": {},
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "is_trading": False,
            "last_scan": None,
            "last_retrain": None,
            "recent_outcomes": [],
        }
        # Clear trade log
        if os.path.exists(TRADE_LOG_FILE):
            os.remove(TRADE_LOG_FILE)
        self._save_state()

    # -- Continuous Trading Loop ----------------------------------------------

    def run_loop(self):
        """Run the trading loop continuously (called from dashboard background)."""
        self.state["is_trading"] = True
        self._save_state()
        print(f"\n{'='*50}")
        print("NSE Auto-Trader ACTIVE")
        print(f"Capital: Rs.{self.state['total_capital']:,.2f}")
        print(f"Stocks: {len(STOCK_SYMBOLS)}")
        print(f"Min Net Profit: {MIN_NET_PROFIT_PCT*100:.1f}%")
        print(f"{'='*50}\n")

        while self.state.get("is_trading", False):
            try:
                now = datetime.now()
                # Only trade during market hours (9:15 AM - 3:30 PM IST)
                if now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour > 15 or (now.hour == 15 and now.minute > 30):
                    time.sleep(30)
                    if os.path.exists(STATE_FILE):
                        with open(STATE_FILE, "r") as f:
                            fresh = json.load(f)
                            if not fresh.get("is_trading", False):
                                print("Trading stopped by user.")
                                break
                    continue

                print(f"\n[{now.strftime('%H:%M:%S')}] Scanning {len(STOCK_SYMBOLS)} stocks...")
                result = self.run_single_cycle()

                # Check if retrain needed
                if self.needs_retrain():
                    print("  [!] Rolling accuracy below threshold -- retrain recommended")

                time.sleep(SCAN_INTERVAL_SECONDS)

                # Reload state in case dashboard changed it
                if os.path.exists(STATE_FILE):
                    with open(STATE_FILE, "r") as f:
                        fresh = json.load(f)
                        if not fresh.get("is_trading", False):
                            print("Trading stopped by user.")
                            break

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  [ERROR] {e}")
                time.sleep(30)

        self.state["is_trading"] = False
        self._save_state()
        print("Auto-Trader stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NSE Auto-Trader (Paper Trading)")
    parser.add_argument("--capital", type=float, default=None, help="Starting capital (INR)")
    parser.add_argument("--single", action="store_true", help="Run single cycle only")
    args = parser.parse_args()

    trader = NSEAutoTrader(capital=args.capital)
    if args.single:
        result = trader.run_single_cycle()
        print(json.dumps(result, indent=2, default=str))
    else:
        trader.run_loop()
