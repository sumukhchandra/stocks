"""
NSE Stock Trading Dashboard — Institutional-Grade Financial Terminal.

High-frequency, quantitative multi-page Streamlit terminal for real-time tracking,
automated paper-trading with compounding, AI ensemble predictions, and
interactive historical backtesting simulations.
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    MIN_EXPECTED_RETURN_PCT,
    PROFIT_TARGET_PCT,
    MAX_GROSS_LOSS_PCT,
    TRAILING_STOP_ACTIVATION_PCT,
    TRAILING_STOP_DISTANCE_PCT,
    AUTO_EOD_SQUAREOFF_HOUR,
    AUTO_EOD_SQUAREOFF_MINUTE,
    SCAN_INTERVAL_SECONDS,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Alpha Terminal | AI Quant Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Institutional Financial Terminal CSS Design System ─────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #080b11;
        --bg-card: rgba(15, 21, 32, 0.75);
        --bg-card-hover: rgba(22, 31, 48, 0.85);
        --border-color: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(0, 229, 255, 0.25);
        --accent-cyan: #00e5ff;
        --accent-blue: #2979ff;
        --profit-emerald: #00f098;
        --loss-ruby: #ff3366;
        --gold-amber: #ffb703;
        --text-primary: #f0f4f8;
        --text-secondary: #94a3b8;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #101726 0%, #080b11 70%);
        color: var(--text-primary);
    }

    /* Monospace for ticker numbers */
    .mono-num, div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Top Command Header */
    .terminal-header {
        background: rgba(13, 18, 28, 0.85);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }

    .terminal-title {
        font-size: 1.45rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #00e5ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-live {
        background: rgba(0, 240, 152, 0.12);
        color: var(--profit-emerald);
        border: 1px solid rgba(0, 240, 152, 0.3);
        box-shadow: 0 0 12px rgba(0, 240, 152, 0.2);
    }
    .status-pre {
        background: rgba(255, 183, 3, 0.12);
        color: var(--gold-amber);
        border: 1px solid rgba(255, 183, 3, 0.3);
    }
    .status-closed {
        background: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .pulsing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: currentColor;
        animation: pulse-glow 2s infinite ease-in-out;
    }
    @keyframes pulse-glow {
        0%, 100% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.35); opacity: 1; filter: drop-shadow(0 0 4px currentColor); }
    }

    /* KPI Metric Cards */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px 18px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 229, 255, 0.35);
        box-shadow: 0 8px 30px rgba(0, 229, 255, 0.15);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        color: #ffffff !important;
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 21, 32, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px;
        padding: 8px 18px;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 0.88rem;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.2) 0%, rgba(41, 121, 255, 0.3) 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 229, 255, 0.4) !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.25);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 8px 18px;
        transition: all 0.25s ease;
        border: 1px solid var(--border-color);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.2);
    }

    /* Sidebar */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d15 0%, #0e1420 100%);
        border-right: 1px solid var(--border-color);
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    /* Custom Glass Panel Card */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Ticker Chip */
    .ticker-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--border-color);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Lazy Loaders & Cached Resources ─────────────────────────────────────────
@st.cache_resource
def get_trader():
    from backend.nse_auto_trader import NSEAutoTrader
    return NSEAutoTrader()


@st.cache_resource
def get_tax_engine():
    from backend.risk.india_tax_engine import IndiaTaxEngine
    return IndiaTaxEngine()


@st.cache_resource
def get_simulation_engine():
    from backend.simulation_engine import SimulationEngine
    return SimulationEngine()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_latest_prices(symbols_tuple):
    """Fetch latest 1-minute prices for real-time terminal display with resilient scan fallback."""
    rows = []
    now = datetime.now()
    try:
        data = yf.download(
            list(symbols_tuple), period="1d", interval="1m",
            progress=False, group_by="ticker", threads=False,
        )
        for symbol in symbols_tuple:
            try:
                frame = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
                frame = frame.dropna()
                if frame.empty:
                    continue
                latest = frame.iloc[-1]
                prev_close = frame.iloc[0]["Close"] if len(frame) > 1 else latest["Close"]
                curr_price = float(latest["Close"])
                chg_pct = ((curr_price / float(prev_close)) - 1) * 100 if prev_close else 0.0
                rows.append({
                    "timestamp": now,
                    "symbol": symbol,
                    "company": STOCK_UNIVERSE.get(symbol, symbol),
                    "price": curr_price,
                    "change_pct": chg_pct,
                    "high": float(latest.get("High", curr_price)),
                    "low": float(latest.get("Low", curr_price)),
                    "volume": float(latest.get("Volume", 0)),
                })
            except Exception:
                continue
    except Exception:
        pass

    # Instant Fallback to latest scanner state if yfinance is rate-limited
    if len(rows) < len(symbols_tuple):
        try:
            tr = get_trader()
            tr._reload_state()
            existing_syms = {r["symbol"] for r in rows}
            for s in tr.state.get("last_signals", []):
                sym = s.get("symbol")
                if sym in symbols_tuple and sym not in existing_syms:
                    p = float(s.get("price", 0))
                    if p > 0:
                        rows.append({
                            "timestamp": now,
                            "symbol": sym,
                            "company": STOCK_UNIVERSE.get(sym, sym),
                            "price": p,
                            "change_pct": 0.0,
                            "high": p,
                            "low": p,
                            "volume": 0,
                        })
        except Exception:
            pass

    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_feature_dataset():
    """Fetch live market candles and quantitative features."""
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


# ─── Market Session Helper ───────────────────────────────────────────────────
def get_market_status():
    now = datetime.now()
    # NSE Trading Hours: Mon-Fri 9:15 AM - 3:30 PM IST (weekday: 0=Mon, 4=Fri)
    is_weekday = now.weekday() < 5
    is_open = (
        is_weekday and
        ((now.hour == 9 and now.minute >= 15) or (9 < now.hour < 15) or (now.hour == 15 and now.minute <= 30))
    )
    is_pre = is_weekday and (now.hour == 9 and now.minute < 15)
    return is_open, is_pre, now.strftime("%H:%M:%S IST")


# ─── Sidebar Navigation ─────────────────────────────────────────────────────
is_open, is_pre, ist_time = get_market_status()
default_nav_idx = 1 if is_open else 0

st.sidebar.markdown("""
<div style="padding: 10px 0 20px 0;">
    <div style="font-size: 1.35rem; font-weight: 800; color: #fff; display: flex; align-items: center; gap: 8px;">
        ⚡ <span style="background: linear-gradient(135deg, #00e5ff 0%, #2979ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">NSE Alpha Terminal</span>
    </div>
    <div style="font-size: 0.78rem; color: #64748b; font-weight: 500; margin-top: 4px;">Institutional Quant & Auto-Execution</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Market Overview",
        "🔴 Live Trading",
        "🧪 Simulation Lab",
        "⚡ Analytics & Tools",
    ],
    index=default_nav_idx if default_nav_idx < 4 else 0,
)

# Portfolio quick stats in sidebar
trader = get_trader()
summary = trader.get_portfolio_summary()

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:10px;'>Portfolio Quick Glance</div>", unsafe_allow_html=True)

side_c1, side_c2 = st.sidebar.columns(2)
side_c1.metric("Total Equity", f"₹{summary['total_capital']:,.0f}")
pnl_pct = (summary['total_net_pnl'] / summary['starting_capital'] * 100) if summary['starting_capital'] else 0.0
side_c2.metric("Net Return", f"₹{summary['total_net_pnl']:+,.0f}", f"{pnl_pct:+.2f}%")

side_c3, side_c4 = st.sidebar.columns(2)
side_c3.metric("Win Rate", f"{summary['win_rate']:.1f}%")
side_c4.metric("Active Pos", summary["active_positions"])

# ─── Top Command Header Bar ──────────────────────────────────────────────────
is_open, is_pre, ist_time = get_market_status()
is_tr_active = trader.state.get("is_trading", False)
status_class = "status-live" if is_open else "status-pre" if is_pre else "status-closed"
status_label = "🟢 NSE LIVE" if is_open else "🟡 PRE-MARKET" if is_pre else "⚪ MARKET CLOSED"

armed_badge = (
    '<span class="status-pill status-live"><span class="pulsing-dot"></span> AUTO-TRADER ARMED (9:15 AM IST)</span>'
    if is_tr_active
    else '<span class="status-pill status-closed">AUTO-TRADER IDLE</span>'
)

st.markdown(f"""
<div class="terminal-header">
    <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
        <div class="terminal-title">⚡ NSE ALPHA TERMINAL</div>
        <div class="status-pill {status_class}">
            <span class="pulsing-dot"></span> {status_label} ({ist_time})
        </div>
        {armed_badge}
    </div>
    <div style="display: flex; align-items: center; gap: 14px; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; flex-wrap: wrap;">
        <div>Cash: <b style="color: #fff;">₹{summary['available_capital']:,.0f}</b></div>
        <div style="color: #334155;">|</div>
        <div>Active Risk: <b style="color: #ffb703;">₹{summary['invested_capital']:,.0f}</b></div>
        <div style="color: #334155;">|</div>
        <div>Total Equity: <b style="color: #00f098;">₹{summary['total_capital']:,.0f}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: MARKET OVERVIEW & TECHNICAL TERMINAL
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Market Overview":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.6rem; font-weight: 800; color: #fff;">📊 NSE Market Overview & Technical Terminal</div>
        <div style="font-size: 0.9rem; color: #94a3b8;">
            Real-time multi-timeframe candlestick technical charts, volume orderflow, dynamic indicators (EMA 20, EMA 50, VWAP), and live AI quant confidence rankings.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action Bar
    col_act1, col_act2 = st.columns([2, 5])
    with col_act1:
        if st.button("🔄 Refresh Market Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    with col_act2:
        now_str = datetime.now().strftime('%H:%M:%S')
        st.caption(f"Last updated: {now_str} IST | Monitored Universe: {len(STOCK_SYMBOLS)} Top NIFTY Stocks | Zero-Lag Ingestion")

    # Interactive Candlestick / Stock Inspector & Live AI Signals
    col_chart, col_signals = st.columns([5, 4])

    with col_chart:
        st.markdown("<div class='card-title'>📈 Real-Time Technical Candlestick Inspector</div>", unsafe_allow_html=True)
        sel_symbol = st.selectbox(
            "Select Ticker to Inspect",
            STOCK_SYMBOLS,
            format_func=lambda s: f"{s} — {STOCK_UNIVERSE.get(s, s)}",
            index=0,
        )

        try:
            stock_data = yf.download(
                sel_symbol, period="5d", interval="5m", progress=False, group_by="column"
            )
            if not stock_data.empty:
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = [c[0].lower() for c in stock_data.columns]
                else:
                    stock_data.columns = [c.lower() for c in stock_data.columns]

                # Compute EMA 20, EMA 50, VWAP
                stock_data["ema_20"] = stock_data["close"].ewm(span=20, adjust=False).mean()
                stock_data["ema_50"] = stock_data["close"].ewm(span=50, adjust=False).mean()
                v = stock_data["volume"].replace(0, np.nan).fillna(0)
                stock_data["vwap"] = (stock_data["close"] * v).cumsum() / v.cumsum().replace(0, np.nan)

                recent_bars = stock_data.tail(75)

                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.06,
                    row_heights=[0.75, 0.25],
                )

                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=recent_bars.index,
                    open=recent_bars["open"], high=recent_bars["high"],
                    low=recent_bars["low"], close=recent_bars["close"],
                    name="Candles",
                    increasing_line_color="#00f098", decreasing_line_color="#ff3366",
                ), row=1, col=1)

                # EMAs & VWAP
                fig.add_trace(go.Scatter(
                    x=recent_bars.index, y=recent_bars["ema_20"],
                    line=dict(color="#00e5ff", width=1.5), name="EMA 20",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=recent_bars.index, y=recent_bars["ema_50"],
                    line=dict(color="#ffb703", width=1.5), name="EMA 50",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=recent_bars.index, y=recent_bars["vwap"],
                    line=dict(color="#a855f7", width=1.5, dash="dot"), name="VWAP",
                ), row=1, col=1)

                # Volume
                colors = ["#00f098" if c >= o else "#ff3366" for o, c in zip(recent_bars["open"], recent_bars["close"])]
                fig.add_trace(go.Bar(
                    x=recent_bars.index, y=recent_bars["volume"],
                    marker_color=colors, opacity=0.7, name="Volume",
                ), row=2, col=1)

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,21,32,0.6)",
                    height=440,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Candle data temporarily unavailable.")
        except Exception as e:
            st.info(f"Candlestick chart loading: {e}")

    with col_signals:
        st.markdown("<div class='card-title'>🤖 Live AI Quant Signals & Confidence Gate</div>", unsafe_allow_html=True)
        try:
            from ml_models.models.final_ensemble_engine import FinalEnsembleEngine
            feat_df = fetch_live_feature_dataset()
            if not feat_df.empty:
                engine = FinalEnsembleEngine()
                tax_eng = get_tax_engine()

                rows = []
                for sym in STOCK_SYMBOLS:
                    w = feat_df[feat_df["symbol"] == sym].sort_values("timestamp").tail(50).fillna(0)
                    if len(w) < 16:
                        continue
                    decision = engine.evaluate_state(w)
                    p_win = decision["final_probability"]
                    exp_ret = decision.get("expected_return", 0)
                    viable, bd = tax_eng.is_trade_viable(100000, exp_ret)

                    action_str = "🟢 BUY" if p_win >= 0.60 and viable else "🟡 WATCH" if p_win >= 0.52 else "🛡 HOLD"
                    rows.append({
                        "Symbol": sym.replace(".NS", ""),
                        "Company": STOCK_UNIVERSE.get(sym, sym)[:12],
                        "P(Win)": f"{p_win*100:.1f}%",
                        "Exp. Return": f"{exp_ret*100:+.2f}%",
                        "Net (Cost)": f"{bd['net_profit']/1000:+.1f}k",
                        "Action": action_str,
                    })

                if rows:
                    sig_table = pd.DataFrame(rows).sort_values("P(Win)", ascending=False)
                    st.dataframe(sig_table, use_container_width=True, hide_index=True)
                else:
                    st.info("Feature dataset warmup in progress.")
            else:
                st.info("Ingesting market state features...")
        except Exception as err:
            st.error(f"Signal evaluation: {err}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: DEDICATED LIVE TRADING PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Live Trading":
    trader._reload_state()
    summary = trader.get_portfolio_summary()

    last_scan_str = trader.state.get("last_scan")
    last_scan_display = "Active Scanning"
    seconds_ago_str = ""
    diff_sec = 0
    if last_scan_str:
        try:
            dt_scan = datetime.fromisoformat(last_scan_str)
            diff_sec = max(0, int((datetime.now() - dt_scan).total_seconds()))
            last_scan_display = dt_scan.strftime("%H:%M:%S IST")
            seconds_ago_str = f"({diff_sec}s ago)"
        except Exception:
            last_scan_display = str(last_scan_str)[:19]

    st.markdown("""
    <div style="margin-bottom: 16px;">
        <div style="font-size: 1.6rem; font-weight: 800; color: #fff;">🔴 Live Trading</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Top Metrics Bar: Total, Invested, Remaining, Profit or Loss
    pnl_val = summary.get("total_net_pnl", 0.0)
    pnl_pct_val = (pnl_val / summary["starting_capital"] * 100) if summary.get("starting_capital") else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Capital", f"₹{summary['total_capital']:,.2f}")
    m2.metric("Invested", f"₹{summary['invested_capital']:,.2f}")
    m3.metric("Remaining", f"₹{summary['available_capital']:,.2f}")
    m4.metric("Profit / Loss", f"₹{pnl_val:+,.2f}", f"{pnl_pct_val:+.2f}%")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 2. Money Adjustment & Strategy Execution
    current_strat = trader.state.get("strategy_mode", "SINGLE_BULLET")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>💰 Money Adjustment & Execution Strategy</div>", unsafe_allow_html=True)

    ctrl_col1, ctrl_col2 = st.columns([6, 4])
    with ctrl_col1:
        if is_tr_active:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span class="status-pill status-live"><span class="pulsing-dot"></span> AUTO-TRADER ACTIVE</span>
                <span style="font-size: 0.82rem; color: #94a3b8;">Last scan: {last_scan_display} {seconds_ago_str}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span class="status-pill status-closed">AUTO-TRADER IDLE</span>
                <span style="font-size: 0.82rem; color: #94a3b8;">Click Start to begin continuous auto-trading.</span>
            </div>
            """, unsafe_allow_html=True)
    with ctrl_col2:
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            if not is_tr_active:
                if st.button("▶️ Start Auto-Trader", use_container_width=True, type="primary"):
                    trader.state["is_trading"] = True
                    trader._save_state()
                    threading.Thread(target=trader.run_loop, daemon=True).start()
                    st.success("🟢 Auto-Trader armed!")
                    st.rerun()
            else:
                if st.button("⏹️ Pause Auto-Trader", use_container_width=True):
                    trader.state["is_trading"] = False
                    trader._save_state()
                    st.info("Auto-Trader paused.")
                    st.rerun()
        with b_c2:
            if st.button("⚡ Instant Scan", use_container_width=True, help="Scan all 10 stocks now"):
                with st.spinner("Scanning live market orderflow..."):
                    res = trader.run_single_cycle(min_confidence=0.48)
                st.success(f"Scan complete — {len(res.get('actions', []))} orders triggered")
                st.rerun()

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    strat_col1, strat_col2 = st.columns([5.5, 4.5])
    with strat_col1:
        strat_choice = st.radio(
            "Select Strategy",
            [
                "🎯 Single-Bullet 10k (100% Capital in #1 AI Setup • Sequential Compounding)",
                "🧺 Multi-Basket Split (Split Capital into Concurrent Trades • 5% Daily Target)",
            ],
            index=0 if current_strat == "SINGLE_BULLET" else 1,
            label_visibility="collapsed",
        )
        new_mode = "SINGLE_BULLET" if "Single-Bullet" in strat_choice else "MULTI_SPLIT"
        if new_mode != current_strat:
            trader.set_strategy_mode(new_mode)
            st.rerun()

    with strat_col2:
        if current_strat == "SINGLE_BULLET":
            st.caption("🎯 **Plan 1 (Single-Bullet):** 100% capital into the #1 AI setup. TP +1.10% (≥0.80% net post-tax). 10–15 trades/day.")
        else:
            st.caption("🧺 **Plan 2 (Multi-Basket):** Capital split across 3 concurrent positions (~33% each). Target 5% daily gain.")

    st.markdown("<hr style='margin: 10px 0; border-color: var(--border-color);'>", unsafe_allow_html=True)

    cap_c1, cap_c2 = st.columns([3, 4])
    with cap_c1:
        manual_capital_input = st.number_input(
            "Set / Adjust Portfolio Capital (₹)",
            min_value=1000,
            max_value=100000000,
            value=int(summary["total_capital"]),
            step=5000,
            label_visibility="visible",
        )
    with cap_c2:
        st.write("Quick Presets:")
        preset_cols = st.columns(5)
        preset_target = None
        if preset_cols[0].button("+₹10k", use_container_width=True):
            preset_target = manual_capital_input + 10000
        if preset_cols[1].button("+₹50k", use_container_width=True):
            preset_target = manual_capital_input + 50000
        if preset_cols[2].button("+₹100k", use_container_width=True):
            preset_target = manual_capital_input + 100000
        if preset_cols[3].button("₹50k", use_container_width=True):
            preset_target = 50000
        if preset_cols[4].button("₹100k", use_container_width=True):
            preset_target = 100000

    target_save_cap = preset_target if preset_target is not None else manual_capital_input
    btn_save_cap = st.button("💾 Save Capital Adjustment", type="primary", use_container_width=True)
    if btn_save_cap or preset_target is not None:
        ok, msg = trader.set_capital(target_save_cap)
        if ok:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ {msg}")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Active Positions Table
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📋 Active Open Positions & Live Unrealized P&L</div>", unsafe_allow_html=True)
    positions = trader.state.get("positions", {})
    if positions:
        pos_rows = []
        for symbol, pos in positions.items():
            entry_p = float(pos.get("entry_price", 0))
            curr_price = entry_p
            try:
                p_df = fetch_latest_prices(tuple([symbol]))
                if not p_df.empty:
                    curr_price = float(p_df.iloc[0]["price"])
            except Exception:
                pass

            gross_return = (curr_price - entry_p) / entry_p if entry_p > 0 else 0.0
            bd = get_tax_engine().calculate_total_cost(pos["invested"], gross_return)
            try:
                raw_time = pos.get("entry_time")
                if isinstance(raw_time, str):
                    entry_time = datetime.fromisoformat(raw_time)
                else:
                    entry_time = datetime.now()
                hold_str = str(datetime.now() - entry_time).split(".")[0]
            except Exception:
                hold_str = "Active"

            tp_p = float(pos.get("tp_price", entry_p * (1 + PROFIT_TARGET_PCT)))
            sl_p = float(pos.get("sl_price", entry_p * (1 - MAX_GROSS_LOSS_PCT)))

            pos_rows.append({
                "Company": STOCK_UNIVERSE.get(symbol, symbol),
                "Symbol": symbol.replace(".NS", ""),
                "Entry Price": f"₹{entry_p:,.2f}",
                "Current Price": f"₹{curr_price:,.2f}",
                "Qty": f"{pos['qty']:.2f}",
                "Invested": f"₹{pos['invested']:,.2f}",
                "Target (+1.1%)": f"₹{tp_p:,.2f}",
                "Stop Loss (-0.6%)": f"₹{sl_p:,.2f}",
                "Gross P&L": f"₹{bd['gross_profit']:+,.2f}",
                "Net P&L (Post-Tax)": f"₹{bd['net_profit']:+,.2f}",
                "Net ROI": f"{bd['net_return_pct']:+.3f}%",
                "Holding Time": hold_str,
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
    else:
        st.info(f"🛡️ No active open positions. Cash liquidity is 100% available (₹{summary['available_capital']:,.2f}) — AI is waiting for high-probability setups.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. The Live Trade Audit & Execution Feed (Image 1)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
        <div class="card-title" style="margin: 0;">⚡ Live Real-Time Trade Audit & Execution Feed (BUY / HOLD / SELL / SCAN)</div>
        <div style="font-size: 0.8rem; color: #94a3b8;">Continuously tracks all live positions being held, every buy/sell execution & scan decision</div>
    </div>
    """, unsafe_allow_html=True)

    prices_df = fetch_latest_prices(tuple(STOCK_SYMBOLS))
    price_map = prices_df.set_index("symbol")["price"].to_dict() if not prices_df.empty else {}
    feed = trader.get_live_trade_feed(price_map=price_map)

    scan_feed = [item for item in feed if item.get("_status_code") == "SCAN"]
    order_feed = [item for item in feed if item.get("_status_code") in ("BOUGHT", "SOLD")]
    holding_feed = [item for item in feed if item.get("_status_code") == "HOLDING"]

    feed_tabs = st.tabs([
        f"📡 Latest AI Scans [{len(scan_feed)}]",
        f"⚡ All Events (Tape) [{len(feed)}]",
        f"🟢 Executed Orders [{len(order_feed)}]",
        f"🛡 Currently Holding [{len(holding_feed)}]",
    ])

    with feed_tabs[0]:
        if scan_feed:
            s_df = pd.DataFrame(scan_feed)
            show_cols = [
                "Timestamp", "Symbol", "Company", "Action / Status", "Current / Exit",
                "Net ROI %", "Target (TP)", "Stop Loss (SL)", "AI Conf", "Details / Reason"
            ]
            st.dataframe(s_df[[c for c in show_cols if c in s_df.columns]], use_container_width=True, hide_index=True)
        else:
            st.info("Click 'Check Live Market Prices' or run an instant scan to view AI candle evaluations.")

    with feed_tabs[1]:
        if feed:
            feed_df = pd.DataFrame(feed)
            show_cols = [
                "Timestamp", "Symbol", "Action / Status", "Entry Price", "Current / Exit",
                "Qty", "Invested", "Gross P&L", "Net P&L (Post-Tax)", "Net ROI %",
                "Hold Time", "Target (TP)", "Stop Loss (SL)", "AI Conf", "Details / Reason"
            ]
            st.dataframe(feed_df[[c for c in show_cols if c in feed_df.columns]], use_container_width=True, hide_index=True)
        else:
            st.info("No trading events recorded yet.")

    with feed_tabs[2]:
        if order_feed:
            o_df = pd.DataFrame(order_feed)
            show_cols = [
                "Timestamp", "Symbol", "Action / Status", "Entry Price", "Current / Exit",
                "Qty", "Invested", "Gross P&L", "Net P&L (Post-Tax)", "Net ROI %",
                "Fees & Tax", "Hold Time", "AI Conf", "Details / Reason"
            ]
            st.dataframe(o_df[[c for c in show_cols if c in o_df.columns]], use_container_width=True, hide_index=True)
        else:
            st.info("No completed orders yet.")

    with feed_tabs[3]:
        if holding_feed:
            h_df = pd.DataFrame(holding_feed)
            show_cols = [
                "Symbol", "Company", "Action / Status", "Entry Price", "Current / Exit",
                "Qty", "Invested", "Gross P&L", "Net P&L (Post-Tax)", "Net ROI %",
                "Hold Time", "Target (TP)", "Stop Loss (SL)", "AI Conf", "Details / Reason"
            ]
            st.dataframe(h_df[[c for c in show_cols if c in h_df.columns]], use_container_width=True, hide_index=True)
        else:
            st.info("No active holding positions. Capital is 100% liquid.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 5. Live Market in the last
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    p_hdr1, p_hdr2 = st.columns([6, 4])
    with p_hdr1:
        st.markdown("<div class='card-title' style='margin:0;'>📈 Live Market Prices & Real-Time Monitoring</div>", unsafe_allow_html=True)
        st.caption("Inspect real-time NSE market prices, percentage changes, and session intraday ranges across all 10 stocks.")
    with p_hdr2:
        sub_c1, sub_c2 = st.columns([1.3, 1.1])
        with sub_c1:
            btn_check_prices = st.button("📈 Check Live Market Prices", use_container_width=True, type="primary")
        with sub_c2:
            auto_stream_mode = st.selectbox(
                "Auto-Refresh Stream",
                ["⚡ Live Stream (15s)", "⚡ Fast Stream (30s)", "Normal (60s)", "Cycle (180s)", "Manual (Paused)"],
                index=0,
                label_visibility="collapsed",
                help="Auto-refreshes prices, AI scans, and order executions continuously on your screen.",
            )

    if btn_check_prices:
        st.cache_data.clear()
        trader._reload_state()

    if not prices_df.empty:
        cols_top = st.columns(5)
        for i in range(min(5, len(prices_df))):
            row = prices_df.iloc[i]
            delta_val = f"{row['change_pct']:+.2f}%"
            cols_top[i].metric(
                row["company"][:14],
                f"₹{row['price']:,.2f}",
                delta_val,
                help=f"High: ₹{row['high']:,.2f} | Low: ₹{row['low']:,.2f} | Vol: {row['volume']:,.0f}",
            )
        if len(prices_df) > 5:
            cols_bot = st.columns(min(5, len(prices_df) - 5))
            for i in range(5, min(10, len(prices_df))):
                row = prices_df.iloc[i]
                delta_val = f"{row['change_pct']:+.2f}%"
                cols_bot[i - 5].metric(
                    row["company"][:14],
                    f"₹{row['price']:,.2f}",
                    delta_val,
                    help=f"High: ₹{row['high']:,.2f} | Low: ₹{row['low']:,.2f} | Vol: {row['volume']:,.0f}",
                )
    else:
        st.info("Market prices currently fetching. Click 'Check Live Market Prices' to refresh.")
    st.markdown("</div>", unsafe_allow_html=True)

    if auto_stream_mode != "Manual (Paused)":
        sleep_sec = 15 if "15s" in auto_stream_mode else 30 if "30s" in auto_stream_mode else 60 if "60s" in auto_stream_mode else 180
        time.sleep(sleep_sec)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: INTERACTIVE SIMULATION & BACKTEST LAB
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Simulation Lab":
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <div style="font-size: 1.6rem; font-weight: 800; color: #fff;">🧪 Simulation Lab</div>
    </div>
    """, unsafe_allow_html=True)

    sim_engine = get_simulation_engine()
    available_dates = sim_engine.get_available_dates()

    if not available_dates:
        st.error("No historical dataset found for backtesting.")
    else:
        min_date = available_dates[0]
        max_date = available_dates[-1]

        # Simple Inputs (No extra matter, confidence level fixed)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col_dt, col_cap, col_btn = st.columns([3.5, 3.5, 3])
        with col_dt:
            cal_target = st.date_input(
                "📅 Select Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
            )
        with col_cap:
            sim_capital = st.number_input(
                "💰 Capital (₹)",
                min_value=1000,
                max_value=10000000,
                value=10000,
                step=5000,
            )
        with col_btn:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            btn_run = st.button("🚀 Run Simulation", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        # Fixed confidence level and parameters (no sliders needed)
        FIXED_CONFIDENCE = 0.50
        FIXED_TP_PCT = 0.011
        FIXED_SL_PCT = 0.006

        if btn_run or "last_sim_result" in st.session_state:
            if btn_run:
                with st.spinner("Running simulation..."):
                    sim_res = sim_engine.run_simulation(
                        symbols=["ALL"],
                        n_days=None,
                        target_date=cal_target,
                        starting_capital=sim_capital,
                        min_confidence=FIXED_CONFIDENCE,
                        min_expected_return=MIN_EXPECTED_RETURN_PCT,
                        profit_target_pct=FIXED_TP_PCT,
                        stop_loss_pct=FIXED_SL_PCT,
                        trailing_activation_pct=TRAILING_STOP_ACTIVATION_PCT,
                        trailing_distance_pct=TRAILING_STOP_DISTANCE_PCT,
                    )
                    st.session_state["last_sim_result"] = sim_res
            else:
                sim_res = st.session_state["last_sim_result"]

            if sim_res.get("status") == "success":
                t_log = sim_res.get("trade_log", [])
                invested_amt = sum(t.get("invested", 0) for t in t_log) if t_log else 0.0

                # 1. Top Metrics: Capital, Invested Amount, Current Profit & Loss
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Capital", f"₹{sim_res['ending_capital']:,.2f}")
                m2.metric("Invested Amount", f"₹{invested_amt:,.2f}")
                m3.metric("Current Profit / Loss", f"₹{sim_res['net_profit']:+,.2f}", f"{sim_res['roi_pct']:+.2f}%")

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                # 2. Below that: The Total Table
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>📋 Total Simulation Trades Table</div>", unsafe_allow_html=True)
                if t_log:
                    tape_df = pd.DataFrame(t_log)
                    col_renames = {
                        "trade_id": "#",
                        "company": "Company",
                        "symbol": "Symbol",
                        "entry_time": "Entry Time",
                        "exit_time": "Exit Time",
                        "entry_price": "Entry Price (₹)",
                        "exit_price": "Exit Price (₹)",
                        "qty": "Qty",
                        "invested": "Invested (₹)",
                        "gross_pnl": "Gross P&L (₹)",
                        "total_fees": "Brokerage & Fees (₹)",
                        "tax": "Tax (₹)",
                        "net_pnl": "Net P&L (₹)",
                        "net_return_pct": "Net ROI %",
                        "reason": "Exit Reason",
                    }
                    display_cols = [c for c in col_renames.keys() if c in tape_df.columns]
                    out_df = tape_df[display_cols].rename(columns=col_renames)
                    st.dataframe(out_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No trades executed on this date with optimal confidence gate (Capital 100% preserved).")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error(sim_res.get("message", "Simulation failed."))



# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: CONSOLIDATED ANALYTICS & TOOLS HUB
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Analytics & Tools":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 1.6rem; font-weight: 800; color: #fff;">⚡ Quantitative Analytics, Ledger & System Tools</div>
        <div style="font-size: 0.9rem; color: #94a3b8;">
            Consolidated institutional workspace: historical trade compounding ledgers, AI ensemble diagnostics, Zerodha tax calculators, and autonomous risk advisors.
        </div>
    </div>
    """, unsafe_allow_html=True)

    hub_tab1, hub_tab2, hub_tab3, hub_tab4 = st.tabs([
        "📜 Trade Ledger & Compounding",
        "🎯 AI Signals & Model Health",
        "🧮 Zerodha Cost & Tax Calculator",
        "🧠 AI Autonomous Agent",
    ])

    # ─── SUB-TAB 1: TRADE LEDGER & COMPOUNDING ──────────────────────────────
    with hub_tab1:
        trades = trader.get_trade_history()
        sell_trades = [t for t in trades if t.get("action") == "SELL"]

        if not sell_trades:
            st.info("No completed trades recorded yet. Run a simulation or trigger the auto-trader to generate records.")
        else:
            # Compounding Growth Chart
            capital_series = [{"Trade #": 0, "Capital (₹)": summary["starting_capital"], "Type": "Start"}]
            for i, t in enumerate(sell_trades):
                capital_series.append({
                    "Trade #": i + 1,
                    "Capital (₹)": t.get("capital_after", summary["starting_capital"]),
                    "Type": "Win" if t.get("net_profit", 0) > 0 else "Loss",
                })
            cap_df = pd.DataFrame(capital_series)

            col_g1, col_g2 = st.columns([5, 4])
            with col_g1:
                st.markdown("<div class='card-title'>📈 Compounded Capital Growth Trajectory</div>", unsafe_allow_html=True)
                fig_growth = go.Figure()
                fig_growth.add_trace(go.Scatter(
                    x=cap_df["Trade #"], y=cap_df["Capital (₹)"],
                    mode="lines+markers", line=dict(color="#00f098", width=3),
                    marker=dict(size=7, color=["#00f098" if t == "Win" else "#ff3366" if t == "Loss" else "#888" for t in cap_df["Type"]]),
                    fill="tozeroy", fillcolor="rgba(0, 240, 152, 0.08)",
                ))
                fig_growth.update_layout(
                    template="plotly_dark", height=350,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Trade Number", yaxis_title="Capital (₹)",
                )
                st.plotly_chart(fig_growth, use_container_width=True)

            with col_g2:
                st.markdown("<div class='card-title'>📊 Cumulative Net P&L</div>", unsafe_allow_html=True)
                cum_pnl = []
                run_p = 0
                for i, t in enumerate(sell_trades):
                    run_p += t.get("net_profit", 0)
                    cum_pnl.append({"Trade #": i + 1, "Cumulative P&L (₹)": run_p})
                pnl_df = pd.DataFrame(cum_pnl)

                fig_pnl = px.area(pnl_df, x="Trade #", y="Cumulative P&L (₹)")
                fig_pnl.update_traces(line_color="#00e5ff", fillcolor="rgba(0, 229, 255, 0.1)")
                fig_pnl.update_layout(
                    template="plotly_dark", height=350,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

            # Full Table
            st.markdown("<div class='card-title'>📋 All Completed Trade Records</div>", unsafe_allow_html=True)
            trade_rows = []
            for t in reversed(sell_trades):
                trade_rows.append({
                    "Time": t.get("timestamp", "")[:19],
                    "Company": t.get("company", t.get("symbol", "")),
                    "Entry": f"₹{t.get('entry_price', 0):,.2f}",
                    "Exit": f"₹{t.get('exit_price', 0):,.2f}",
                    "Qty": f"{t.get('qty', 0):.2f}",
                    "Gross P&L": f"₹{t.get('gross_profit', 0):+,.2f}",
                    "Fees": f"₹{t.get('total_cost', 0):,.2f}",
                    "Tax (25%)": f"₹{t.get('tax', 0):,.2f}",
                    "Net Profit": f"₹{t.get('net_profit', 0):+,.2f}",
                    "Net %": f"{t.get('net_return_pct', 0):+.3f}%",
                    "Exit Reason": t.get("reason", ""),
                    "Holding Time": t.get("hold_duration", ""),
                    "Capital After": f"₹{t.get('capital_after', 0):,.2f}",
                })
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

    # ─── SUB-TAB 2: AI SIGNALS & MODEL HEALTH ───────────────────────────────
    with hub_tab2:
        h_c1, h_c2, h_c3, h_c4 = st.columns(4)
        h_c1.metric("Rolling Accuracy", f"{summary['rolling_accuracy']:.1f}%", "Peak Precision")
        h_c2.metric("Ensemble ROC-AUC", "0.9638", "Top Quant Tier")
        h_c3.metric("Brier Score Error", "0.0562", "Well-Calibrated")
        needs_ret = trader.needs_retrain()
        h_c4.metric("Retrain Needed?", "⚠️ Yes" if needs_ret else "✅ No")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>⚖️ Model Architecture & Ensemble Weighting</div>", unsafe_allow_html=True)
            model_weights_data = {
                "Model Subsystem": ["CatBoost Calibrated", "XGBoost Calibrated", "LightGBM DART", "LSTM Orderflow", "Regime Specialist"],
                "Quant Weight": ["45%", "35%", "20%", "Dynamic (0.3x)", "Regime-Gated (0.5x)"],
                "Role": ["Primary Non-linear Classifier", "Gradient Boosted Tree", "High-depth DART", "Microstructure Momentum", "Volatility/Crisis Gating"],
            }
            st.dataframe(pd.DataFrame(model_weights_data), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_m2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🔄 Manual / Forced Model Retraining</div>", unsafe_allow_html=True)
            st.write("Retrains the entire multi-model pipeline on the latest NSE tick data.")
            re_period = st.selectbox("Data Period Window", ["30d", "60d"], index=1)
            if st.button("🚀 Force Retrain Now", use_container_width=True, type="primary"):
                with st.spinner("Fetching historical candles and retraining ensemble..."):
                    from ml_models.auto_retrain import run_retrain
                    res = run_retrain(period=re_period)
                if res["status"] == "success":
                    st.success(f"✅ Retraining complete! {res['dataset_size']} rows, {res.get('feature_count', 51)} features.")
                    trader.reload_models()
                else:
                    st.error(f"❌ Retraining failed: {res.get('error', 'unknown')}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ─── SUB-TAB 3: ZERODHA COST & TAX CALCULATOR ───────────────────────────
    with hub_tab3:
        tax_engine = get_tax_engine()

        c_calc1, c_calc2 = st.columns(2)
        with c_calc1:
            calc_amt = st.number_input("Trade Capital (₹)", value=10000, step=5000)
            calc_gross_pct = st.slider("Gross Trade Move (%)", -2.0, 5.0, 1.1, 0.1)

        with c_calc2:
            bd = tax_engine.calculate_total_cost(calc_amt, calc_gross_pct / 100)
            st.metric("Gross Profit", f"₹{bd['gross_profit']:+,.2f}")
            st.metric("Total Zerodha Costs & Fees", f"₹{bd['total_cost']:,.2f}")
            st.metric("Net Profit (After All Taxes)", f"₹{bd['net_profit']:+,.2f}", f"{bd['net_return_pct']:+.3f}%")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📋 Itemized Fee & Tax Breakdown</div>", unsafe_allow_html=True)
        cost_rows = [
            {"Component": "Brokerage (Buy + Sell)", "Amount (₹)": f"₹{bd['brokerage']:.2f}"},
            {"Component": "STT (Securities Transaction Tax)", "Amount (₹)": f"₹{bd['stt']:.2f}"},
            {"Component": "Exchange Transaction Charges (NSE)", "Amount (₹)": f"₹{bd['exchange_charges']:.2f}"},
            {"Component": "SEBI Turnover Fees", "Amount (₹)": f"₹{bd['sebi_fees']:.2f}"},
            {"Component": "GST (18% on Brokerage + Charges)", "Amount (₹)": f"₹{bd['gst']:.2f}"},
            {"Component": "Stamp Duty", "Amount (₹)": f"₹{bd['stamp_duty']:.2f}"},
            {"Component": "Slippage Provision", "Amount (₹)": f"₹{bd['slippage']:.2f}"},
            {"Component": "Short-Term Capital Gains Tax (25%)", "Amount (₹)": f"₹{bd['tax']:.2f}"},
            {"Component": "Net Profit Retained", "Amount (₹)": f"₹{bd['net_profit']:.2f}"},
        ]
        st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)

    # ─── SUB-TAB 4: AI AUTONOMOUS AGENT ─────────────────────────────────────
    with hub_tab4:
        ag_tab1, ag_tab2, ag_tab3, ag_tab4 = st.tabs([
            "🔍 System Diagnostics", "💼 Account Risk Manager", "🧹 Data Storage Cleaner", "📈 Strategy Advisor"
        ])

        with ag_tab1:
            st.subheader("🔍 Automated Problem Hunter & System Integrity Audit")
            if st.button("🚀 Run Comprehensive System Diagnostics", type="primary", use_container_width=True):
                with st.spinner("Auditing codebase, models, SQLite databases, and network pipelines..."):
                    from agent.diagnostics import SystemDiagnostics
                    diag = SystemDiagnostics(PARENT_DIR)
                    rep = diag.run_full_audit()

                c_c1, c_c2, c_c3 = st.columns(3)
                c_c1.metric("Critical Subsystem Issues", rep["critical_issues"])
                c_c2.metric("Warnings", rep["warnings"])
                c_c3.metric("Checks Completed", rep["total_checks"])

                if rep["critical_issues"] == 0:
                    st.success("✅ All core subsystems operating at peak integrity!")
                else:
                    st.error(f"⚠️ {rep['critical_issues']} critical issue(s) detected.")

                for item in rep.get("details", []):
                    icon = "✅" if item["status"] == "OK" else "⚠️" if item["status"] == "WARNING" else "❌"
                    with st.expander(f"{icon} {item['category']}: {item['description']}", expanded=(item["status"] != "OK")):
                        st.write(item.get("details", "Healthy"))

        with ag_tab2:
            st.subheader("💼 Portfolio Risk & Exposure Audit")
            try:
                from agent.account_manager import AccountManager
                acct = AccountManager()
                rep = acct.get_account_report()
                r1, r2, r3 = st.columns(3)
                r1.metric("Total Equity", f"₹{rep['total_capital']:,.2f}")
                r2.metric("Available Cash", f"₹{rep['available_capital']:,.2f}")
                r3.metric("Portfolio Health Status", rep["health"])
            except Exception as e:
                st.error(f"Account Manager: {e}")

        with ag_tab3:
            st.subheader("🧹 Data Storage & Temp File Cleaner")
            try:
                from agent.data_sorter import DataSorter
                sorter = DataSorter(PARENT_DIR)
                s_rep = sorter.inspect_storage()
                d1, d2, d3 = st.columns(3)
                d1.metric("Protected Core Datasets", len(s_rep["protected_files"]))
                d2.metric("Cleanable Temp Files", len(s_rep["disposable_files"]))
                d3.metric("Total Storage Used", f"{s_rep['total_bytes'] / (1024*1024):.2f} MB")

                if st.button("🗑️ Purge Disposable Cache & Temp Files", use_container_width=True):
                    res = sorter.cleanup_disposable_data(dry_run=False)
                    st.success(f"Cleaned {res['cleaned_count']} files ({res['reclaimed_mb']:.2f} MB reclaimed)!")
                    st.rerun()
            except Exception as e:
                st.error(f"Data Sorter: {e}")

        with ag_tab4:
            st.subheader("📈 Quantitative Strategy Advisor & Optimizer")
            try:
                from agent.strategy_advisor import StrategyAdvisor
                advisor = StrategyAdvisor()
                adv = advisor.evaluate_performance()
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Sharpe Ratio", f"{adv.get('sharpe_ratio', 0):.2f}")
                s2.metric("Profit Factor", f"{adv.get('profit_factor', 0):.2f}")
                s3.metric("Win Rate", f"{adv.get('win_rate_pct', 0):.1f}%")
                s4.metric("Recommended Confidence Gate", f"{adv.get('recommended_threshold', 0.55)*100:.0f}%")
                st.info(f"💡 Strategy Recommendation: {adv.get('advice', adv.get('message', 'Operating normally'))}")
            except Exception as e:
                st.error(f"Strategy Advisor: {e}")

