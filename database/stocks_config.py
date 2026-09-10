"""Shared stock universe and trading configuration for the NSE trading system."""

# --- Stock Universe (10 high-liquidity NSE stocks) ---------------------------
STOCK_UNIVERSE = {
    # Original 5
    "RELIANCE.NS": "Reliance Industries",
    "HDFCBANK.NS": "HDFC Bank",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ICICIBANK.NS": "ICICI Bank",
    "TATACAP.NS": "Tata Capital",
    # Added 5 (high-liquidity Nifty 50)
    "INFY.NS": "Infosys",
    "TCS.NS": "TCS",
    "SBIN.NS": "State Bank of India",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
}

STOCK_SYMBOLS = list(STOCK_UNIVERSE.keys())

# --- Strategy Modes & Compounding Configuration ------------------------------
STRATEGY_MODE = "SINGLE_BULLET"      # Default mode: "SINGLE_BULLET" (100% in #1 setup) or "MULTI_SPLIT" (split into N baskets)
SINGLE_BULLET_ALLOCATION_PCT = 0.98  # 98% of available capital into the single highest-conviction trade
MULTI_SPLIT_POSITIONS = 3            # Split capital into 3 concurrent trades in MULTI_SPLIT mode

# --- Trading Thresholds & Profit Optimization (Post-Tax Guarantee) -----------
TARGET_NET_PROFIT_PCT = 0.008        # 0.80% minimum NET profit after ALL Zerodha fees and STT
MIN_NET_PROFIT_PCT = 0.0005         # 0.05% absolute hurdle
MIN_EXPECTED_RETURN_PCT = 0.0010    # 0.10% (10 bps) minimum predicted return from ensemble
PROFIT_TARGET_PCT = 0.011            # 1.10% Take-Profit target (guarantees >= 0.80% net post-tax)
MAX_GROSS_LOSS_PCT = 0.006           # 0.60% Stop-Loss limit (strict 2:1 reward-to-risk ratio)
TRAILING_STOP_ACTIVATION_PCT = 0.006  # Ratchet stop-loss to Breakeven (+0.25%) once trade gains +0.60%
TRAILING_STOP_DISTANCE_PCT = 0.004   # Trails 0.40% below peak watermark
AUTO_EOD_SQUAREOFF_HOUR = 15         # Auto square-off at 3:20 PM IST to eliminate overnight gap-down risk
AUTO_EOD_SQUAREOFF_MINUTE = 20
DEFAULT_CAPITAL = 10000              # Default starting capital (INR 10,000)

TARGET_PROFIT_PER_TRADE = 80.0       # Target Rs.80-100 net profit per trade in single-bullet mode
TARGET_DAILY_PROFIT = 500.0          # Target Rs.500 daily net profit (10-15 trades)
INTRADAY_LEVERAGE = 5.0              # 5x Margin Intraday Square-off (MIS on Zerodha)
MIN_POSITION_SIZE_INR = 3000.0       # Minimum size per trade in split mode

# --- Position Sizing Dictionary (Legacy / Multi-Basket reference) ------------
POSITION_SIZING = {
    "high":   0.30,   # High conviction
    "medium": 0.20,   # Moderate
    "low":    0.15,   # Low
}

# --- Scanning & Timing -------------------------------------------------------
SCAN_INTERVAL_SECONDS = 60          # Active 60s scanner to capture 10-15 intraday trade setups/day
MARKET_OPEN_HOUR = 9                # NSE opens at 9:15 AM IST
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15              # NSE closes at 3:30 PM IST
MARKET_CLOSE_MINUTE = 30

# --- Auto-Retrain ------------------------------------------------------------
RETRAIN_HOUR = 9                    # Daily auto-retrain at 9:00 AM IST
RETRAIN_MINUTE = 0
ACCURACY_RETRAIN_THRESHOLD = 0.55   # Retrain if rolling accuracy drops below 55%
ROLLING_ACCURACY_WINDOW = 20        # Number of recent trades to compute rolling accuracy
DATA_FETCH_PERIOD = "60d"           # Fetch 60 days historical data (Yahoo Finance 5m maximum limit)
DATA_FETCH_INTERVAL = "5m"          # 5-minute candles

# --- Tax Configuration -------------------------------------------------------
PROFIT_TAX_PCT = 0.25               # 25% short-term capital gains tax (user chose higher buffer)
