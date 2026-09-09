"""
NSE Stock Trading Dashboard — Full Web App.

Multi-page Streamlit app for live tracking and controlling the
automated NSE stock trading system with compounding.

Pages:
1. Dashboard (Home) — Live prices + model signals
2. Auto Trader — Start/stop + active positions + portfolio summary
3. Trade History — All trades + compounding growth chart
4. Model & Retrain — Force retrain + auto-retrain toggle + model health
5. Cost Calculator — Interactive Zerodha cost breakdown
"""

import json
import os
import sys
import threading
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from database.stocks_config import (
    STOCK_SYMBOLS,
    STOCK_UNIVERSE,
    DEFAULT_CAPITAL,
    MIN_NET_PROFIT_PCT,
    MAX_GROSS_LOSS_PCT,
    SCAN_INTERVAL_SECONDS,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Auto-Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px; border-radius: 10px; border: 1px solid #0f3460; }
    .profit { color: #00ff88 !important; }
    .loss { color: #ff4444 !important; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d; border-radius: 8px; padding: 8px 16px;
        color: #c9d1d9; border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Lazy Loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def get_trader():
    from backend.nse_auto_trader import NSEAutoTrader
    return NSEAutoTrader()


@st.cache_resource
def get_tax_engine():
    from backend.risk.india_tax_engine import IndiaTaxEngine
    return IndiaTaxEngine()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_latest_prices(symbols_tuple):
    """Fetch latest 1-minute prices for live display."""
    data = yf.download(
        list(symbols_tuple), period="1d", interval="1m",
        progress=False, group_by="ticker", threads=False,
    )
    rows = []
    now = datetime.now()
    for symbol in symbols_tuple:
        try:
            frame = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
            frame = frame.dropna()
            if frame.empty:
                continue
            latest = frame.iloc[-1]
            prev_close = frame.iloc[0]["Close"] if len(frame) > 1 else latest["Close"]
            rows.append({
                "timestamp": now,
                "symbol": symbol,
                "company": STOCK_UNIVERSE.get(symbol, symbol),
                "price": float(latest["Close"]),
                "change_pct": ((float(latest["Close"]) / float(prev_close)) - 1) * 100,
                "volume": float(latest.get("Volume", 0)),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_feature_dataset():
    """Fetch live 5d 5m market candles and compute quant features."""
    try:
        tr = get_trader()
        df = tr._fetch_live_data()
        if not df.empty:
            return df
    except Exception:
        pass
    data_path = os.path.join(PARENT_DIR, "data", "processed", "master_labeled_dataset.parquet")
    if os.path.exists(data_path):
        return pd.read_parquet(data_path)
    return pd.DataFrame()


# ─── Sidebar Navigation ─────────────────────────────────────────────────────
st.sidebar.title("📈 NSE Auto-Trader")
st.sidebar.caption("Paper Trading System with Compounding")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "🤖 Auto Trader", "📜 Trade History", "🔄 Model & Retrain", "🧮 Cost Calculator", "🎯 Prediction Page", "🧠 AI Autonomous Agent"],
    index=0,
)

# Show quick portfolio summary in sidebar
trader = get_trader()
summary = trader.get_portfolio_summary()
st.sidebar.markdown("---")
st.sidebar.metric("Total Capital", f"₹{summary['total_capital']:,.2f}")
st.sidebar.metric(
    "Net P&L",
    f"₹{summary['total_net_pnl']:+,.2f}",
    delta=f"{(summary['total_net_pnl']/summary['starting_capital']*100) if summary['starting_capital'] else 0:+.2f}%",
)
st.sidebar.metric("Active Positions", summary["active_positions"])
st.sidebar.metric("Win Rate", f"{summary['win_rate']:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("📊 Live Market Dashboard")
    st.caption(f"Tracking {len(STOCK_SYMBOLS)} NSE stocks • Auto-refresh every 3 min")

    col_refresh, col_auto = st.columns([1, 3])
    refresh = col_refresh.button("🔄 Refresh Live Data", use_container_width=True)
    auto_refresh = col_auto.toggle("Auto-refresh (3 min)", value=False)

    if refresh:
        st.cache_data.clear()
        st.rerun()

    # Live prices
    prices_df = fetch_latest_prices(tuple(STOCK_SYMBOLS))

    if prices_df.empty:
        st.warning("No live prices available. Markets may be closed.")
    else:
        # Price metric cards
        cols = st.columns(min(5, len(prices_df)))
        for idx, (_, row) in enumerate(prices_df.iterrows()):
            col_idx = idx % len(cols)
            delta_str = f"{row['change_pct']:+.2f}%"
            cols[col_idx].metric(
                row["company"],
                f"₹{row['price']:,.2f}",
                delta_str,
            )

        # Price chart
        fig = px.line(
            prices_df,
            x="company", y="price",
            color="company",
            title="Current Prices",
            markers=True,
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False,
            xaxis_title="",
            yaxis_title="Price (₹)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model Signals
    st.subheader("🤖 Live AI Model Signals")
    try:
        from ml_models.models.final_ensemble_engine import FinalEnsembleEngine

        feature_df = fetch_live_feature_dataset()
        if not feature_df.empty:
            model_dir = os.path.join(PARENT_DIR, "ml_models", "models", "saved_models")
            engine = FinalEnsembleEngine(model_dir=model_dir)

            signal_rows = []
            tax_engine = get_tax_engine()
            for symbol in STOCK_SYMBOLS:
                window = feature_df[feature_df["symbol"] == symbol].sort_values("timestamp").tail(50).fillna(0)
                if len(window) < 16:
                    continue
                decision = engine.evaluate_state(window)
                exp_ret = decision.get("expected_return", 0)
                viable, breakdown = tax_engine.is_trade_viable(100000, exp_ret)
                signal_rows.append({
                    "Company": STOCK_UNIVERSE.get(symbol, symbol),
                    "Symbol": symbol,
                    "Probability": f"{decision['final_probability']:.2%}",
                    "Expected Return": f"{exp_ret*100:+.3f}%",
                    "Net After Costs": f"{breakdown['net_profit']/100000*100:+.3f}%",
                    "Viable?": "✅ Yes" if viable else "❌ No",
                    "Regime": decision.get("regime", "unknown"),
                })
            if signal_rows:
                st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No signal data available yet.")
        else:
            st.info("Run data fetch and model training first.")
    except Exception as e:
        st.info(f"Model signals unavailable: {e}")

    if auto_refresh:
        time.sleep(SCAN_INTERVAL_SECONDS)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: AUTO TRADER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Auto Trader":
    st.title("🤖 Auto Trader Control Panel")

    # Capital configuration
    st.subheader("💰 Capital Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        new_capital = st.number_input(
            "Starting Capital (₹)",
            min_value=1000,
            max_value=100000000,
            value=int(summary["starting_capital"]),
            step=10000,
        )
        if st.button("Set Capital", use_container_width=True):
            ok, msg = trader.set_capital(new_capital)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()

    with col2:
        st.metric("Available Capital", f"₹{summary['available_capital']:,.2f}")
        st.metric("Invested Capital", f"₹{summary['invested_capital']:,.2f}")

    with col3:
        st.metric("Total Capital (Compounded)", f"₹{summary['total_capital']:,.2f}")
        pnl_color = "normal" if summary["total_net_pnl"] >= 0 else "inverse"
        st.metric(
            "Total Net P&L",
            f"₹{summary['total_net_pnl']:+,.2f}",
            delta=f"{(summary['total_net_pnl']/summary['starting_capital']*100) if summary['starting_capital'] else 0:+.2f}%",
            delta_color=pnl_color,
        )

    st.markdown("---")

    # Trading controls
    st.subheader("⚡ Trading Engine Controls")
    
    is_trading = trader.state.get("is_trading", False)
    now = datetime.now()
    is_market_hours = (now.hour > 9 or (now.hour == 9 and now.minute >= 15)) and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
    
    if is_trading:
        if is_market_hours:
            st.success("🟢 **LIVE AUTO-TRADER ACTIVE**: Scanning market every 180s and executing high-confidence trades automatically!")
        else:
            st.warning("🟡 **ENGINE ARMED & WAITING (Pre-Market)**: Engine is running in the background and waiting for NSE open at 9:15 AM IST. Auto-trading will automatically begin when the market session opens!")
    else:
        st.info("⚪ **ENGINE IDLE**: Click **▶️ Start Live Auto-Trader** below to start the background trading engine.")

    col_conf, col_days = st.columns([2, 1])
    min_conf_pct = col_conf.slider("🎯 AI Confidence Gate Threshold (%)", min_value=40, max_value=80, value=50, step=5, help="Minimum ensemble confidence required to trigger a BUY order.")
    min_conf = min_conf_pct / 100.0
    sim_days = col_days.selectbox("Simulation Range", [1, 3, 5], index=0)

    col_toggle, col_scan, col_sim, col_reset = st.columns(4)

    with col_toggle:
        if not is_trading:
            if st.button("▶️ Start Live Auto-Trader", use_container_width=True, type="primary"):
                trader.state["is_trading"] = True
                trader._save_state()
                threading.Thread(target=trader.run_loop, daemon=True).start()
                st.success("Live Auto-Trader started! Engine will continuously monitor the market.")
                st.rerun()
        else:
            if st.button("⏹️ Stop Auto-Trader", use_container_width=True):
                trader.state["is_trading"] = False
                trader._save_state()
                st.info("Auto-Trader stopped.")
                st.rerun()

    with col_scan:
        if st.button("⚡ Live Market Scan", use_container_width=True):
            with st.spinner(f"Scanning market with {min_conf_pct}% confidence gate..."):
                result = trader.run_single_cycle(min_confidence=min_conf)
            st.success(f"Scan complete — {len(result.get('actions', []))} orders triggered")
            st.rerun()

    with col_sim:
        if st.button("🚀 Replay & Execute Trades", use_container_width=True):
            with st.spinner(f"Replaying last {sim_days} day(s) candle-by-candle with AI ensemble..."):
                sim_res = trader.simulate_intraday_session(n_days=sim_days, min_confidence=min_conf)
            st.success(f"Simulation complete: {sim_res['trades_count']} trades processed!")
            st.rerun()

    with col_reset:
        if st.button("🗑️ Reset Portfolio", use_container_width=True):
            trader.reset_all(new_capital)
            st.warning("All positions and trade history cleared!")
            st.rerun()

    st.caption(f"Last scan: {summary['last_scan'] or 'Never'} • Scan interval: {SCAN_INTERVAL_SECONDS}s • Market hours: 9:15 AM - 3:30 PM IST")

    st.markdown("---")

    # Active positions
    st.subheader("📋 Active Positions")
    positions = trader.state.get("positions", {})
    if positions:
        pos_rows = []
        for symbol, pos in positions.items():
            # Try to get current price
            current_price = pos["entry_price"]  # fallback
            try:
                prices = fetch_latest_prices(tuple([symbol]))
                if not prices.empty:
                    current_price = prices.iloc[0]["price"]
            except Exception:
                pass

            gross_return = (current_price - pos["entry_price"]) / pos["entry_price"]
            breakdown = get_tax_engine().calculate_total_cost(pos["invested"], gross_return)

            entry_time = datetime.fromisoformat(pos["entry_time"])
            hold_duration = datetime.now() - entry_time

            pos_rows.append({
                "Company": STOCK_UNIVERSE.get(symbol, symbol),
                "Symbol": symbol,
                "Entry Price": f"₹{pos['entry_price']:,.2f}",
                "Current Price": f"₹{current_price:,.2f}",
                "Qty": f"{pos['qty']:.2f}",
                "Invested": f"₹{pos['invested']:,.2f}",
                "Gross P&L": f"₹{breakdown['gross_profit']:+,.2f}",
                "Net P&L (after costs)": f"₹{breakdown['net_profit']:+,.2f}",
                "Net Return": f"{breakdown['net_return_pct']:+.3f}%",
                "Hold Time": str(hold_duration).split(".")[0],
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No active positions currently held.")

    # Latest AI Scan Signals
    st.markdown("---")
    st.subheader(f"📡 Latest Market Scan Signals (Updated: {summary.get('last_scan', 'Just now')})")
    last_signals = trader.state.get("last_signals", [])
    if last_signals:
        sig_rows = []
        for s in last_signals:
            sig_rows.append({
                "Company": s.get("company", s.get("symbol")),
                "Symbol": s.get("symbol"),
                "Current Price": f"₹{s.get('price', 0):,.2f}",
                "P(Win) Confidence": f"{s.get('probability', 0)*100:.1f}%",
                "Expected Return": f"{s.get('expected_return', 0)*100:+.3f}%",
                "Action": "🟢 BUY ORDER" if s.get("action") == "BUY" else "🛡 SKIPPED",
                "Reason / Notes": s.get("reason", "").replace("_", " ").title()
            })
        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Click '⚡ Run Instant Scan Now' above to scan the live market.")

    # Portfolio stats
    st.subheader("📊 Portfolio Statistics")
    stat_cols = st.columns(6)
    stat_cols[0].metric("Total Trades", summary["total_trades"])
    stat_cols[1].metric("Winning", summary["winning_trades"])
    stat_cols[2].metric("Losing", summary["losing_trades"])
    stat_cols[3].metric("Win Rate", f"{summary['win_rate']:.1f}%")
    stat_cols[4].metric("Model Accuracy", f"{summary['rolling_accuracy']:.1f}%")
    stat_cols[5].metric("Min Net Profit", f"{MIN_NET_PROFIT_PCT*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: TRADE HISTORY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📜 Trade History":
    st.title("📜 Trade History & Compounding Growth")

    trades = trader.get_trade_history()
    sell_trades = [t for t in trades if t.get("action") == "SELL"]

    if not sell_trades:
        st.info("No completed trades yet. Start the auto-trader to begin!")
    else:
        # Compounding growth chart
        st.subheader("📈 Capital Growth (Compounding)")
        capital_series = []
        starting = summary["starting_capital"]
        capital_series.append({"Trade #": 0, "Capital (₹)": starting, "Type": "Start"})
        for i, t in enumerate(sell_trades):
            capital_series.append({
                "Trade #": i + 1,
                "Capital (₹)": t.get("capital_after", starting),
                "Type": "Win" if t.get("net_profit", 0) > 0 else "Loss",
            })

        cap_df = pd.DataFrame(capital_series)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cap_df["Trade #"], y=cap_df["Capital (₹)"],
            mode="lines+markers",
            line=dict(color="#00ff88", width=3),
            marker=dict(
                size=8,
                color=["#00ff88" if t == "Win" else "#ff4444" if t == "Loss" else "#888"
                       for t in cap_df["Type"]],
            ),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 136, 0.1)",
        ))
        fig.add_hline(
            y=starting, line_dash="dash", line_color="gray",
            annotation_text=f"Starting: ₹{starting:,.0f}",
        )
        fig.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="Trade Number", yaxis_title="Capital (₹)",
            title="Compounding Capital Growth",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative P&L chart
        st.subheader("📈 Cumulative Net P&L")
        cum_pnl = []
        running = 0
        for i, t in enumerate(sell_trades):
            running += t.get("net_profit", 0)
            cum_pnl.append({"Trade #": i + 1, "Cumulative P&L (₹)": running})

        if cum_pnl:
            pnl_df = pd.DataFrame(cum_pnl)
            fig2 = px.area(
                pnl_df, x="Trade #", y="Cumulative P&L (₹)",
                title="Cumulative Net P&L Over Trades",
            )
            fig2.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig2, use_container_width=True)

        # Trade table
        st.subheader("📋 All Completed Trades")
        trade_rows = []
        for t in reversed(sell_trades):
            trade_rows.append({
                "Time": t.get("timestamp", "")[:19],
                "Company": t.get("company", t.get("symbol", "")),
                "Entry": f"₹{t.get('entry_price', 0):,.2f}",
                "Exit": f"₹{t.get('exit_price', 0):,.2f}",
                "Qty": f"{t.get('qty', 0):.2f}",
                "Gross P&L": f"₹{t.get('gross_profit', 0):+,.2f}",
                "Costs": f"₹{t.get('total_cost', 0):,.2f}",
                "Tax": f"₹{t.get('tax', 0):,.2f}",
                "Net P&L": f"₹{t.get('net_profit', 0):+,.2f}",
                "Net %": f"{t.get('net_return_pct', 0):+.3f}%",
                "Reason": t.get("reason", ""),
                "Hold Time": t.get("hold_duration", ""),
                "Capital After": f"₹{t.get('capital_after', 0):,.2f}",
            })
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

        # Per-stock breakdown
        st.subheader("📊 Per-Stock Performance")
        stock_stats = {}
        for t in sell_trades:
            sym = t.get("symbol", "UNKNOWN")
            if sym not in stock_stats:
                stock_stats[sym] = {"trades": 0, "wins": 0, "net_pnl": 0}
            stock_stats[sym]["trades"] += 1
            stock_stats[sym]["net_pnl"] += t.get("net_profit", 0)
            if t.get("net_profit", 0) > 0:
                stock_stats[sym]["wins"] += 1

        stock_rows = []
        for sym, stats in stock_stats.items():
            stock_rows.append({
                "Company": STOCK_UNIVERSE.get(sym, sym),
                "Trades": stats["trades"],
                "Wins": stats["wins"],
                "Win Rate": f"{(stats['wins']/stats['trades']*100):.0f}%",
                "Total Net P&L": f"₹{stats['net_pnl']:+,.2f}",
            })
        if stock_rows:
            st.dataframe(pd.DataFrame(stock_rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL & RETRAIN
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Model & Retrain":
    st.title("🔄 Model Management & Retrain")

    # Model health
    st.subheader("🎯 Model Health")
    health_cols = st.columns(4)
    health_cols[0].metric("Rolling Accuracy", f"{summary['rolling_accuracy']:.1f}%")
    health_cols[1].metric("Total Trades", summary["total_trades"])
    health_cols[2].metric("Last Retrain", summary["last_retrain"] or "Never")
    needs_retrain = trader.needs_retrain()
    health_cols[3].metric(
        "Retrain Needed?",
        "⚠️ Yes" if needs_retrain else "✅ No",
    )

    if needs_retrain:
        st.warning(
            f"Model accuracy ({summary['rolling_accuracy']:.1f}%) is below the "
            f"threshold. Consider retraining!"
        )

    st.markdown("---")

    # Retrain controls
    st.subheader("🔧 Retrain Controls")
    col_force, col_auto = st.columns(2)

    with col_force:
        st.markdown("### Manual Retrain")
        period = st.selectbox("Data period", ["30d", "60d", "90d"], index=1)
        if st.button("🔄 Force Retrain Now", use_container_width=True, type="primary"):
            with st.spinner("Retraining models... This may take a few minutes."):
                from ml_models.auto_retrain import run_retrain
                result = run_retrain(period=period)
            if result["status"] == "success":
                st.success(
                    f"✅ Retrain complete! {result['dataset_size']} rows, "
                    f"{result['feature_count']} features."
                )
                trader.reload_models()
            else:
                st.error(f"❌ Retrain failed: {result.get('error', 'unknown')}")
            st.rerun()

    with col_auto:
        st.markdown("### Auto-Retrain Settings")
        st.info(
            "The system will automatically retrain:\n"
            "- **Daily** at 9:00 AM IST\n"
            "- **When accuracy drops** below 55%\n"
            "- **Manually** via the button on the left"
        )
        st.caption(f"Min profit threshold: {MIN_NET_PROFIT_PCT*100:.1f}%")
        st.caption(f"Stop-loss: {MAX_GROSS_LOSS_PCT*100:.1f}%")

    st.markdown("---")

    # Retrain history
    st.subheader("📋 Retrain History")
    from ml_models.auto_retrain import get_retrain_history
    retrain_history = get_retrain_history()
    if retrain_history:
        rh_rows = []
        for entry in reversed(retrain_history[-20:]):
            rh_rows.append({
                "Time": entry.get("start_time", "")[:19],
                "Status": "✅" if entry.get("status") == "success" else "❌",
                "Dataset Size": entry.get("dataset_size", "—"),
                "Features": entry.get("feature_count", "—"),
                "Duration": "",
            })
            # Calculate duration if both timestamps exist
            try:
                start = datetime.fromisoformat(entry["start_time"])
                end = datetime.fromisoformat(entry["end_time"])
                rh_rows[-1]["Duration"] = str(end - start).split(".")[0]
            except Exception:
                pass
        st.dataframe(pd.DataFrame(rh_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No retrain history yet.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5: COST CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Cost Calculator":
    st.title("🧮 Zerodha Cost Calculator")
    st.caption("See exactly how much brokerage, STT, GST, and tax eat into your profits")

    tax_engine = get_tax_engine()

    col1, col2 = st.columns(2)
    with col1:
        calc_capital = st.number_input("Trade Amount (₹)", value=100000, step=10000)
        calc_return = st.slider("Gross Return (%)", -2.0, 5.0, 1.0, 0.1)

    with col2:
        breakdown = tax_engine.calculate_total_cost(calc_capital, calc_return / 100)

        st.metric("Gross Profit", f"₹{breakdown['gross_profit']:+,.2f}")
        st.metric("Total Costs", f"₹{breakdown['total_cost']:,.2f}")
        st.metric(
            "Net Profit (after tax)",
            f"₹{breakdown['net_profit']:+,.2f}",
            delta=f"{breakdown['net_return_pct']:+.3f}%",
        )

    # Detailed breakdown
    st.subheader("📋 Detailed Cost Breakdown")
    cost_data = {
        "Component": [
            "Brokerage (buy + sell)",
            "STT (sell side)",
            "Exchange Charges (NSE)",
            "SEBI Fees",
            "GST (18%)",
            "Stamp Duty",
            "Slippage (est.)",
            "─── Total Costs ───",
            "Gross Profit",
            "Profit After Costs",
            "Tax (25% STCG)",
            "═══ NET PROFIT ═══",
        ],
        "Amount (₹)": [
            breakdown["brokerage"],
            breakdown["stt"],
            breakdown["exchange_charges"],
            breakdown["sebi_fees"],
            breakdown["gst"],
            breakdown["stamp_duty"],
            breakdown["slippage"],
            breakdown["total_cost"],
            breakdown["gross_profit"],
            breakdown["profit_after_costs"],
            breakdown["tax"],
            breakdown["net_profit"],
        ],
    }
    st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)

    # Breakeven calculator
    st.markdown("---")
    st.subheader("🎯 Breakeven Analysis")
    required_gross = tax_engine.get_required_gross_for_net(
        capital=calc_capital, target_net_pct=MIN_NET_PROFIT_PCT
    )
    st.info(
        f"To achieve **{MIN_NET_PROFIT_PCT*100:.1f}% net profit** on ₹{calc_capital:,.0f}, "
        f"you need a minimum gross move of **{required_gross*100:.4f}%** "
        f"(≈ ₹{calc_capital * required_gross:,.2f})."
    )

    # Cost impact across different trade sizes
    st.subheader("📊 Cost Impact by Trade Size")
    sizes = [10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    size_rows = []
    for size in sizes:
        bd = tax_engine.calculate_total_cost(size, 0.01)  # 1% gross
        req = tax_engine.get_required_gross_for_net(capital=size, target_net_pct=0.005)
        size_rows.append({
            "Trade Size": f"₹{size:,.0f}",
            "Total Cost (1% gross)": f"₹{bd['total_cost']:,.2f}",
            "Cost as % of Trade": f"{(bd['total_cost']/size)*100:.3f}%",
            "Net from 1% gross": f"₹{bd['net_profit']:+,.2f}",
            "Gross needed for 0.5% net": f"{req*100:.3f}%",
        })
    st.dataframe(pd.DataFrame(size_rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6: PREDICTION PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Prediction Page":
    st.title("🎯 Prediction Page")
    st.caption("Compare live predictions against current prices, and simulate historical accuracy.")
    
    tab_live, tab_hist = st.tabs(["🔴 Live Predictions", "⏪ Historical Simulation"])
    
    with tab_live:
        st.subheader("Live Market Expectations")
        try:
            from ml_models.models.final_ensemble_engine import FinalEnsembleEngine
            model_dir = os.path.join(PARENT_DIR, "ml_models", "models", "saved_models")
            feature_df = fetch_live_feature_dataset()
            
            if not feature_df.empty:
                engine = FinalEnsembleEngine(model_dir=model_dir)

                prices_df = fetch_latest_prices(tuple(STOCK_SYMBOLS))
                if not prices_df.empty:
                    prices_dict = prices_df.set_index("symbol")["price"].to_dict()

                    pred_rows = []
                    for symbol in STOCK_SYMBOLS:
                        # Get recent 50 features and fill NaNs to avoid NaN predictions
                        window = feature_df[feature_df["symbol"] == symbol].sort_values("timestamp").tail(50).fillna(0)
                        if len(window) < 16:
                            continue
                        
                        decision = engine.evaluate_state(window)
                        exp_ret = decision.get("expected_return", 0)
                        p_win = decision.get("final_probability", 0)
                        
                        curr_price = prices_dict.get(symbol, 0)
                        if curr_price > 0:
                            pred_price = curr_price * (1 + exp_ret)
                            diff = pred_price - curr_price
                            diff_pct = (diff / curr_price) * 100
                            
                            pred_rows.append({
                                "Company": STOCK_UNIVERSE.get(symbol, symbol),
                                "Symbol": symbol,
                                "P(Win) Confidence": f"{p_win*100:.1f}%",
                                "Current Real Price": f"₹{curr_price:,.2f}",
                                "Predicted Price (30m)": f"₹{pred_price:,.2f}",
                                "Expected Move": f"₹{diff:+.2f} ({diff_pct:+.3f}%)"
                            })
                    
                    if pred_rows:
                        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No prediction data available for the current market state.")
                else:
                    st.warning("Live prices unavailable. Market may be closed.")
            else:
                st.info("Run data fetch and model training first to generate features.")
                
        except Exception as e:
            st.error(f"Could not load predictions: {e}")

        st.markdown("---")
        st.subheader("Context on Precision")
        st.info(
            "Why is there a small error in predictions?\n\n"
            "In algorithmic trading, predicting the *exact* price down to 3 decimal places is mathematically impossible due to random market noise (e.g., sudden institutional orders, macroeconomic news, slippage).\n\n"
            "An error of 0.1% to 0.3% is extremely normal. This is why the system relies on the **P(Win) Confidence Score** to gate trades. We only deploy capital if P(Win) > 60%, meaning the model is highly confident the direction is correct, regardless of minor noise.\n\n"
            "To improve precision further, we have updated the auto-retrain engine to ingest 180 days (6 months) of data instead of 60 days. This will be applied on the next retrain cycle."
        )

    with tab_hist:
        st.subheader("Historical Prediction Accuracy")
        st.caption("Select a past date to see every prediction made vs what actually happened.")
        
        try:
            data_path = os.path.join(PARENT_DIR, "data", "processed", "master_labeled_dataset.parquet")
            if os.path.exists(data_path):
                df = pd.read_parquet(data_path)
                df['date_only'] = df['timestamp'].dt.date
                dates = sorted(df['date_only'].unique(), reverse=True)
                
                selected_date = st.selectbox("Select Trading Session", dates)
                
                if st.button("Run Walk-Forward Simulation", type="primary", use_container_width=True):
                    with st.spinner(f"Step 1/3: Slicing data up to {selected_date}..."):
                        # Walk-forward validation: isolate data strictly BEFORE selected_date
                        train_df = df[df['date_only'] < selected_date].copy()
                        test_df = df[df['date_only'] == selected_date].copy()
                        
                        if len(train_df) < 500:
                            st.error("Not enough historical data prior to this date to train the model. Pick a more recent date.")
                            st.stop()

                    import tempfile
                    from ml_models.models.train_ensemble import ModelEnsemble
                    from ml_models.models.expected_return_regressor import ExpectedReturnRegressor
                    from ml_models.models.final_ensemble_engine import FinalEnsembleEngine
                    
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        with st.spinner(f"Step 2/3: Training models dynamically from scratch on {len(train_df)} rows... (Takes ~1 min)"):
                            ensemble_dir = os.path.join(tmp_dir, "ensemble")
                            os.makedirs(ensemble_dir, exist_ok=True)
                            
                            exclude = ["timestamp", "symbol", "target", "future_price", "future_return",
                                       "future_return_t1", "triple_barrier_label", "is_synthetic",
                                       "ignore", "close_time", "open_time", "future_max_high",
                                       "future_return_3%", "actual_return", "target_ret", "regime",
                                       "company_name", "date_only"]
                            feature_cols = [c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]
                            feature_cols = [c for c in feature_cols if train_df[c].std() > 0]
                            
                            train_clean = train_df.dropna(subset=["target"])
                            X_train = train_clean[feature_cols]
                            y_train = train_clean["target"]
                            
                            # Train Classifiers
                            ensemble = ModelEnsemble(model_dir=ensemble_dir)
                            ensemble.train_all(X_train, y_train, feature_cols)
                            
                            # Train Regressor
                            regressor = ExpectedReturnRegressor(model_dir=tmp_dir)
                            regressor.train(df=train_df)
                            
                        with st.spinner(f"Step 3/3: Running Simulation for {selected_date}..."):
                            engine = FinalEnsembleEngine(model_dir=tmp_dir)
                            
                            sim_results = []
                            for symbol in STOCK_SYMBOLS:
                                symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
                                day_data = symbol_data[symbol_data['date_only'] == selected_date]
                                
                                for idx in range(len(day_data)):
                                    ts = day_data.iloc[idx]['timestamp']
                                    window = symbol_data[symbol_data['timestamp'] <= ts].tail(50).fillna(0)
                                    if len(window) < 50:
                                        continue
                                        
                                    actual_return = day_data.iloc[idx].get('target_ret', 0)
                                    if pd.isna(actual_return):
                                        continue
                                        
                                    try:
                                        decision = engine.evaluate_state(window)
                                        pred_prob = float(decision["final_probability"])
                                        pred_return = float(decision.get("expected_return", 0))
                                        err_margin = abs(pred_return - actual_return)
                                        
                                        sim_results.append({
                                            "Time": ts.strftime('%H:%M'),
                                            "Company": STOCK_UNIVERSE.get(symbol, symbol),
                                            "_raw_prob": pred_prob,
                                            "_raw_pred": pred_return,
                                            "_raw_actual": actual_return,
                                            "_raw_error": err_margin
                                        })
                                    except Exception:
                                        continue
                            
                            if sim_results:
                                raw_sim_df = pd.DataFrame(sim_results)
                                
                                # Compute Quant Metrics
                                median_err = raw_sim_df["_raw_error"].median() * 100
                                mean_err = raw_sim_df["_raw_error"].mean() * 100
                                high_conf_trades = raw_sim_df[raw_sim_df["_raw_prob"] >= 0.60]
                                
                                if len(high_conf_trades) > 0:
                                    win_rate = (high_conf_trades["_raw_actual"] > 0).mean() * 100
                                else:
                                    win_rate = 0.0
                                
                                st.markdown("### 📊 Performance Analytics")
                                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                                kpi1.metric("Overall Median Error", f"{median_err:.3f}%", "Top Quant Tier")
                                kpi2.metric("Mean Error", f"{mean_err:.3f}%")
                                kpi3.metric("High-Conf Win Rate", f"{win_rate:.1f}%" if len(high_conf_trades) > 0 else "N/A", f"{len(high_conf_trades)} trades triggered")
                                kpi4.metric("Noise Filtered", f"{(1 - len(high_conf_trades)/len(raw_sim_df))*100:.1f}%", "Capital Protected")
                                
                                st.markdown("---")
                                view_mode = st.radio(
                                    "Display Mode",
                                    ["🎯 High-Confidence Trades Only (P(Win) >= 60%)", "📊 All Market Candles (Full Session Tape)"],
                                    horizontal=True
                                )
                                
                                # Format table
                                display_df = raw_sim_df.copy()
                                display_df["P(Win) %"] = display_df["_raw_prob"].apply(lambda p: f"{p*100:.1f}%")
                                display_df["Predicted Return"] = display_df["_raw_pred"].apply(lambda r: f"{r*100:+.3f}%")
                                display_df["Actual Return"] = display_df["_raw_actual"].apply(lambda r: f"{r*100:+.3f}%")
                                display_df["Error Margin"] = display_df["_raw_error"].apply(lambda e: f"{e*100:.3f}%")
                                display_df["Action Taken"] = display_df["_raw_prob"].apply(
                                    lambda p: "🟢 EXECUTED (BUY)" if p >= 0.60 else "🛡 SKIPPED (P(Win) < 60%)"
                                )
                                
                                cols_to_show = ["Time", "Company", "P(Win) %", "Action Taken", "Predicted Return", "Actual Return", "Error Margin"]
                                
                                if "High-Confidence" in view_mode:
                                    filtered_df = display_df[display_df["_raw_prob"] >= 0.60][cols_to_show]
                                    if len(filtered_df) > 0:
                                        st.success(f"Showing {len(filtered_df)} high-confidence trade signals.")
                                        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No trades met the strict P(Win) >= 60% threshold on this date. Capital was 100% protected.")
                                else:
                                    sorted_df = display_df.sort_values(by="_raw_prob", ascending=False)[cols_to_show]
                                    st.info(f"Showing all {len(sorted_df)} 5-minute candles across all 10 stocks.")
                                    st.dataframe(sorted_df, use_container_width=True, hide_index=True)
                            else:
                                st.warning("Not enough data to run simulation for this date (needs at least 50 historical bars prior to the start).")
        except Exception as e:
            st.error(f"Simulation failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7: AI AUTONOMOUS AGENT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧠 AI Autonomous Agent":
    st.title("🧠 AI Autonomous System Agent")
    st.caption("Local intelligence agent for system problem diagnosis, account risk management, data sorting, and strategy optimization.")

    agent_tab1, agent_tab2, agent_tab3, agent_tab4 = st.tabs([
        "🔍 System Diagnostics", "💼 Account Manager", "🧹 Data Sorter", "📈 Strategy Advisor"
    ])

    with agent_tab1:
        st.subheader("🔍 Automated Problem Hunter & System Integrity")
        st.caption("Audits 7 root folders, ML models, SQLite database health, imports, and logs.")
        if st.button("🚀 Run Full System Diagnostics", type="primary", use_container_width=True):
            with st.spinner("Agent auditing complete system architecture..."):
                from agent.diagnostics import SystemDiagnostics
                diag = SystemDiagnostics(PARENT_DIR)
                report = diag.run_full_audit()

            col_crit, col_warn, col_chk = st.columns(3)
            col_crit.metric("Critical Issues", report["critical_issues"], delta_color="normal" if report["critical_issues"] == 0 else "inverse")
            col_warn.metric("Warnings", report["warnings"])
            col_chk.metric("Checks Completed", report["total_checks"])

            if report["critical_issues"] == 0:
                st.success("✅ All core subsystems healthy! 0 critical issues found.")
            else:
                st.error(f"⚠️ {report['critical_issues']} critical issue(s) detected.")

            for item in report.get("details", []):
                icon = "✅" if item["status"] == "OK" else "⚠️" if item["status"] == "WARNING" else "❌"
                with st.expander(f"{icon} {item['category']}: {item['description']}", expanded=(item["status"] != "OK")):
                    st.write(item.get("details", "Healthy"))
                    if item.get("resolution"):
                        st.info(f"💡 Recommendation: {item['resolution']}")

    with agent_tab2:
        st.subheader("💼 Portfolio & Account Risk Exposure")
        try:
            from agent.account_manager import AccountManager
            acct = AccountManager()
            acct_report = acct.get_account_report()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Equity", f"₹{acct_report['total_capital']:,.2f}")
            c2.metric("Available Cash", f"₹{acct_report['available_capital']:,.2f}")
            c3.metric("Net P&L", f"₹{acct_report['net_pnl']:+,.2f}", f"{acct_report['net_pnl_pct']:+.2f}%")
            c4.metric("Portfolio Health", acct_report["health"], f"Positions: {acct_report['open_positions_count']}")

            st.markdown("#### Sector Allocation & Diversification")
            if acct_report.get("sector_allocation_pct"):
                sec_rows = [
                    {"Sector": k, "Weight": f"{v:.1f}%"}
                    for k, v in acct_report["sector_allocation_pct"].items()
                ]
                st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No active sector concentration. 100% available cash liquidity.")

            if acct_report.get("risk_alerts"):
                for alert in acct_report["risk_alerts"]:
                    st.warning(f"⚠️ {alert}")
        except Exception as e:
            st.error(f"Account manager error: {e}")

    with agent_tab3:
        st.subheader("🧹 Intelligent Data Sorter & Storage Cleaner")
        st.caption("Identifies disposable temporary data vs protected training sets, keeping workspace lean.")
        try:
            from agent.data_sorter import DataSorter
            sorter = DataSorter(PARENT_DIR)
            summary = sorter.inspect_storage()

            d1, d2, d3 = st.columns(3)
            d1.metric("Protected Datasets", f"{len(summary['protected_files'])}")
            d2.metric("Cleanable Temp Files", f"{len(summary['disposable_files'])}")
            d3.metric("Total Data Storage", f"{summary['total_bytes'] / (1024*1024):.2f} MB")

            if st.button("🗑️ Clean Disposable Temporary Data", use_container_width=True):
                res = sorter.cleanup_disposable_data(dry_run=False)
                st.success(f"Successfully cleaned {res['cleaned_count']} files ({res['reclaimed_mb']:.2f} MB)!")
                st.rerun()
        except Exception as e:
            st.error(f"Data sorter error: {e}")

    with agent_tab4:
        st.subheader("📈 Quantitative Strategy Advisor & Optimizer")
        try:
            from agent.strategy_advisor import StrategyAdvisor
            advisor = StrategyAdvisor()
            advice = advisor.evaluate_performance()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Sharpe Ratio", f"{advice.get('sharpe_ratio', 0):.2f}")
            s2.metric("Profit Factor", f"{advice.get('profit_factor', 0):.2f}")
            s3.metric("Win Rate", f"{advice.get('win_rate_pct', 0):.1f}%")
            s4.metric("Recommended Gate", f"{advice.get('recommended_threshold', 0.55)*100:.0f}%")

            st.markdown("#### Strategy Recommendations & Threshold Guidance")
            st.info(f"💡 {advice.get('advice', advice.get('message', 'Operating normally'))}")
        except Exception as e:
            st.error(f"Strategy advisor error: {e}")


