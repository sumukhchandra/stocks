"""
High-Performance Vectorized Simulation & Backtest Engine.

Simulates algorithmic intraday & swing trading on NSE stocks with:
- Zero look-ahead bias (strict chronological order).
- Real Indian market microstructure costs (Zerodha brokerage, STT, exchange turnover, GST, SEBI, stamp duty, STCG 25% tax).
- Dynamic AI confidence gates and risk-managed position sizing.
- Comprehensive quantitative performance analytics (Sharpe, Drawdown, Win Rate, Compounding Equity vs Buy & Hold).
"""

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from database.stocks_config import STOCK_SYMBOLS, STOCK_UNIVERSE
from backend.risk.india_tax_engine import IndiaTaxEngine
from ml_models.models.final_ensemble_engine import FinalEnsembleEngine


class SimulationEngine:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(PARENT_DIR, "ml_models", "models", "saved_models")
        self.model_dir = model_dir
        self.engine = FinalEnsembleEngine(model_dir=self.model_dir)
        self.tax_engine = IndiaTaxEngine()
        self._dataset_cache = None

    def load_dataset(self):
        """Loads and caches the master feature dataset."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        data_path = os.path.join(PARENT_DIR, "data", "processed", "master_labeled_dataset.parquet")
        if os.path.exists(data_path):
            df = pd.read_parquet(data_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date_only"] = df["timestamp"].dt.date
            self._dataset_cache = df
            return df
        return pd.DataFrame()

    def get_available_dates(self):
        """Returns sorted list of distinct trading dates in dataset."""
        df = self.load_dataset()
        if df.empty:
            return []
        return sorted(df["date_only"].unique())

    def run_simulation(
        self,
        symbols=None,
        n_days=None,
        target_date=None,
        starting_capital=100000.0,
        min_confidence=0.55,
        min_expected_return=0.008,
        profit_target_pct=0.020,
        stop_loss_pct=0.008,
        trailing_activation_pct=0.008,
        trailing_distance_pct=0.005,
        max_open_positions=4,
        max_allocation_pct=0.20,
    ):
        """
        Runs full backtest simulation.
        
        Args:
            symbols: list of stock symbols (None = all in universe)
            n_days: number of recent days to simulate (if target_date is None)
            target_date: specific date to simulate (overrides n_days)
            starting_capital: initial portfolio cash in INR
            min_confidence: minimum AI probability to open trade (e.g. 0.55 = 55%)
            min_expected_return: minimum predicted gross return (e.g. 0.008 = 0.8%)
            profit_target_pct: profit take threshold (e.g. 0.020 = 2.0%)
            stop_loss_pct: stop loss threshold (e.g. 0.008 = 0.8%)
            trailing_activation_pct: gain needed to activate trailing stop (e.g. 0.008 = +0.8%)
            trailing_distance_pct: distance to trail behind peak watermark (e.g. 0.005 = 0.5%)
            max_open_positions: max concurrent active trades
            max_allocation_pct: max % of available capital per trade
        """
        df = self.load_dataset()
        if df.empty:
            return {"status": "error", "message": "No historical dataset found."}

        day_before_target = None
        history_bars_count = 0

        # Date filtering & Pre-Target Data Collection
        if target_date is not None:
            if isinstance(target_date, datetime):
                target_date = target_date.date()
            elif isinstance(target_date, str):
                target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

            # Historical data collected strictly up to the day before target date
            df_history_before = df[df["date_only"] < target_date]
            history_dates = sorted(df_history_before["date_only"].unique())
            day_before_target = history_dates[-1] if history_dates else (target_date - timedelta(days=1))
            history_bars_count = len(df_history_before)

            # Target session to replay
            df_target = df[df["date_only"] == target_date].copy()
            if symbols and len(symbols) > 0 and "ALL" not in symbols:
                df_sim = df_target[df_target["symbol"].isin(symbols)].copy()
            else:
                df_sim = df_target.copy()
        elif n_days is not None:
            if symbols and len(symbols) > 0 and "ALL" not in symbols:
                df_filtered = df[df["symbol"].isin(symbols)].copy()
            else:
                df_filtered = df.copy()
            dates = sorted(df_filtered["date_only"].unique())
            selected_dates = dates[-n_days:] if len(dates) >= n_days else dates
            df_sim = df_filtered[df_filtered["date_only"].isin(selected_dates)].copy()

        if df_sim.empty:
            return {
                "status": "error",
                "message": f"No data found for selected date ({target_date}). Data was available up to {day_before_target}.",
                "day_before_target": day_before_target,
            }

        # Precompute model predictions across all rows using fast vectorization
        X_vec = df_sim.reindex(columns=self.engine.primary_features, fill_value=0)
        df_sim["pred_prob"] = self.engine.predict_universal(X_vec)
        df_sim["pred_ret"] = self.engine.predict_return_ensemble(X_vec)

        # Simulation state
        capital = float(starting_capital)
        cash = float(starting_capital)
        active_positions = {}
        trade_log = []
        equity_records = []

        timestamps = sorted(df_sim["timestamp"].unique())
        if len(timestamps) == 0:
            return {"status": "error", "message": "No timestamps found."}

        # Track benchmark buy & hold (equal weight across universe at start)
        first_ts = timestamps[0]
        first_bars = df_sim[df_sim["timestamp"] == first_ts].set_index("symbol")["close"].to_dict()
        benchmark_shares = {}
        for sym, p0 in first_bars.items():
            if p0 > 0:
                benchmark_shares[sym] = (starting_capital / max(len(first_bars), 1)) / p0

        total_signals_evaluated = 0
        signals_triggered = 0

        for current_time in timestamps:
            current_bars = df_sim[df_sim["timestamp"] == current_time]
            price_map = current_bars.set_index("symbol")["close"].to_dict()
            high_map = current_bars.set_index("symbol")["high"].to_dict()
            low_map = current_bars.set_index("symbol")["low"].to_dict()

            # ── 1. Check Exits on Active Positions ──
            closed_now = []
            for symbol, pos in active_positions.items():
                if symbol not in price_map:
                    continue

                curr_close = float(price_map[symbol])
                curr_high = float(high_map.get(symbol, curr_close))
                curr_low = float(low_map.get(symbol, curr_close))

                # Update peak watermark price reached during trade
                highest = max(pos.get("highest_price", pos["entry_price"]), curr_close, curr_high)
                pos["highest_price"] = highest

                # Dynamic Trailing Stop Ratchet:
                # If price advanced >= trailing_activation_pct (+0.80%), ratchet stop loss
                gain_from_entry = (highest - pos["entry_price"]) / pos["entry_price"]
                if gain_from_entry >= trailing_activation_pct:
                    breakeven_p = pos["entry_price"] * 1.0025  # Breakeven + round-trip taxes & fees
                    trail_p = highest * (1.0 - trailing_distance_pct)
                    pos["sl_price"] = max(pos["sl_price"], trail_p, breakeven_p)

                tp_price = pos["tp_price"]
                sl_price = pos["sl_price"]
                hit_tp = curr_high >= tp_price
                hit_sl = curr_low <= sl_price
                hit_time = current_time >= pos["expiration"]
                is_eod = current_time == timestamps[-1] or (current_time.hour == 15 and current_time.minute >= 20)

                exit_price = None
                exit_reason = None

                if hit_tp:
                    exit_price = max(curr_close, tp_price)
                    exit_reason = "TAKE_PROFIT"
                elif hit_sl:
                    exit_price = min(curr_close, sl_price)
                    exit_reason = "TRAILING_STOP" if gain_from_entry >= trailing_activation_pct else "STOP_LOSS"
                elif is_eod:
                    exit_price = curr_close
                    exit_reason = "EOD_SQUAREOFF"
                elif hit_time:
                    exit_price = curr_close
                    exit_reason = "TIME_EXPIRY"

                if exit_price is not None:
                    gross_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                    cost_breakdown = self.tax_engine.calculate_total_cost(pos["invested"], gross_ret)
                    net_profit = cost_breakdown["net_profit"]
                    
                    cash += pos["invested"] + net_profit
                    capital = cash + sum(p["invested"] for s, p in active_positions.items() if s != symbol)

                    hold_mins = int((current_time - pos["entry_time"]).total_seconds() / 60)

                    trade_log.append({
                        "trade_id": len(trade_log) + 1,
                        "symbol": symbol,
                        "company": STOCK_UNIVERSE.get(symbol, symbol),
                        "entry_time": pos["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        "exit_time": current_time.strftime("%Y-%m-%d %H:%M"),
                        "hold_mins": hold_mins,
                        "entry_price": round(pos["entry_price"], 2),
                        "exit_price": round(exit_price, 2),
                        "qty": round(pos["qty"], 2),
                        "invested": round(pos["invested"], 2),
                        "gross_pnl": round(cost_breakdown["gross_profit"], 2),
                        "total_fees": round(cost_breakdown["total_cost"], 2),
                        "tax": round(cost_breakdown["tax"], 2),
                        "net_pnl": round(net_profit, 2),
                        "net_return_pct": round(cost_breakdown["net_return_pct"], 3),
                        "reason": exit_reason,
                        "p_win": round(pos["prob"] * 100, 1),
                        "capital_after": round(capital, 2),
                    })
                    closed_now.append(symbol)

            for sym in closed_now:
                del active_positions[sym]

            # ── 2. Check Entries for New Setups ──
            if len(active_positions) < max_open_positions and cash > 1000:
                for _, row in current_bars.iterrows():
                    sym = row["symbol"]
                    if sym in active_positions:
                        continue
                    if len(active_positions) >= max_open_positions:
                        break

                    total_signals_evaluated += 1
                    prob = float(row["pred_prob"])
                    exp_ret = float(row["pred_ret"])
                    curr_price = float(row["close"])

                    # Confidence Gate & Expected Return Filter
                    if prob >= min_confidence and exp_ret >= min_expected_return and curr_price > 0:
                        signals_triggered += 1
                        
                        # Size position
                        avail = cash * max_allocation_pct
                        alloc = min(cash * 0.95, max(1000.0, avail))
                        qty = alloc / curr_price

                        tp = curr_price * (1 + profit_target_pct)
                        sl = curr_price * (1 - stop_loss_pct)

                        active_positions[sym] = {
                            "entry_price": curr_price,
                            "highest_price": curr_price,
                            "qty": qty,
                            "invested": alloc,
                            "tp_price": tp,
                            "sl_price": sl,
                            "prob": prob,
                            "exp_ret": exp_ret,
                            "entry_time": current_time,
                            "expiration": current_time + timedelta(minutes=75),
                        }
                        cash -= alloc

            # ── 3. Record Snapshot for Equity Curve ──
            current_portfolio_value = cash
            for sym, pos in active_positions.items():
                p_now = float(price_map.get(sym, pos["entry_price"]))
                current_portfolio_value += pos["qty"] * p_now

            # Benchmark valuation
            benchmark_val = 0.0
            for sym, shares in benchmark_shares.items():
                p_now = float(price_map.get(sym, first_bars.get(sym, 0)))
                benchmark_val += shares * p_now

            equity_records.append({
                "timestamp": current_time,
                "strategy_equity": round(current_portfolio_value, 2),
                "benchmark_equity": round(benchmark_val, 2),
                "cash": round(cash, 2),
                "open_positions": len(active_positions),
            })

        # Close any lingering positions at end of session
        if active_positions:
            last_ts = timestamps[-1]
            last_bars = df_sim[df_sim["timestamp"] == last_ts].set_index("symbol")["close"].to_dict()
            for sym, pos in list(active_positions.items()):
                p_end = float(last_bars.get(sym, pos["entry_price"]))
                gross_ret = (p_end - pos["entry_price"]) / pos["entry_price"]
                cost_breakdown = self.tax_engine.calculate_total_cost(pos["invested"], gross_ret)
                net_profit = cost_breakdown["net_profit"]
                cash += pos["invested"] + net_profit
                trade_log.append({
                    "trade_id": len(trade_log) + 1,
                    "symbol": sym,
                    "company": STOCK_UNIVERSE.get(sym, sym),
                    "entry_time": pos["entry_time"].strftime("%Y-%m-%d %H:%M"),
                    "exit_time": last_ts.strftime("%Y-%m-%d %H:%M"),
                    "hold_mins": int((last_ts - pos["entry_time"]).total_seconds() / 60),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(p_end, 2),
                    "qty": round(pos["qty"], 2),
                    "invested": round(pos["invested"], 2),
                    "gross_pnl": round(cost_breakdown["gross_profit"], 2),
                    "total_fees": round(cost_breakdown["total_cost"], 2),
                    "tax": round(cost_breakdown["tax"], 2),
                    "net_pnl": round(net_profit, 2),
                    "net_return_pct": round(cost_breakdown["net_return_pct"], 3),
                    "reason": "END_OF_SIMULATION",
                    "p_win": round(pos["prob"] * 100, 1),
                    "capital_after": round(cash, 2),
                })
            active_positions.clear()

        final_capital = cash
        net_profit = final_capital - starting_capital
        roi_pct = (net_profit / starting_capital) * 100 if starting_capital > 0 else 0.0

        # Performance analytics
        total_trades = len(trade_log)
        winning_trades = len([t for t in trade_log if t["net_pnl"] > 0])
        losing_trades = len([t for t in trade_log if t["net_pnl"] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        gross_wins = sum(t["net_pnl"] for t in trade_log if t["net_pnl"] > 0)
        gross_losses = abs(sum(t["net_pnl"] for t in trade_log if t["net_pnl"] < 0))
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

        total_fees = sum(t["total_fees"] for t in trade_log)
        total_tax = sum(t["tax"] for t in trade_log)

        # Equity DF & Drawdown
        equity_df = pd.DataFrame(equity_records)
        max_drawdown_pct = 0.0
        max_drawdown_amt = 0.0
        sharpe_ratio = 0.0

        if not equity_df.empty:
            peak = equity_df["strategy_equity"].cummax()
            dd = (equity_df["strategy_equity"] - peak) / peak
            max_drawdown_pct = abs(dd.min()) * 100
            max_drawdown_amt = (peak - equity_df["strategy_equity"]).max()

            returns = equity_df["strategy_equity"].pct_change().dropna()
            if len(returns) > 5 and returns.std() > 0:
                # Annualized for 5m intraday data: 75 bars/day * 250 days = 18750 bars/yr
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(18750)

        # Benchmark comparison
        benchmark_final = equity_df["benchmark_equity"].iloc[-1] if not equity_df.empty else starting_capital
        benchmark_pnl = benchmark_final - starting_capital
        benchmark_roi = (benchmark_pnl / starting_capital) * 100 if starting_capital > 0 else 0.0

        return {
            "status": "success",
            "starting_capital": starting_capital,
            "ending_capital": final_capital,
            "net_profit": net_profit,
            "roi_pct": roi_pct,
            "benchmark_roi": benchmark_roi,
            "alpha_pct": roi_pct - benchmark_roi,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_amt": max_drawdown_amt,
            "sharpe_ratio": sharpe_ratio,
            "total_fees": total_fees,
            "target_date": target_date,
            "day_before_target": day_before_target,
            "history_bars_collected": history_bars_count,
            "total_signals_evaluated": total_signals_evaluated,
            "signals_triggered": signals_triggered,
            "noise_filtered_pct": ((1 - signals_triggered / max(1, total_signals_evaluated)) * 100),
            "trade_log": trade_log,
            "equity_df": equity_df,
        }
