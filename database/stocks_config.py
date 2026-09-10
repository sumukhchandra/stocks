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

# --- Trading Thresholds & High-Yield Target Optimization ----------------------
TARGET_PROFIT_PER_TRADE = 50.0       # Target Rs.50 net profit per successful trade
TARGET_DAILY_PROFIT = 500.0         # Target Rs.500 daily net profit (10-15 successful trades)
INTRADAY_LEVERAGE = 5.0             # 5x Margin Intraday Square-off (MIS standard on Zerodha/Upstox)
MIN_POSITION_SIZE_INR = 4500.0      # Minimum size per trade (~Rs.4,500 - Rs.5,000) to yield Rs.50+ net after all taxes
MIN_NET_PROFIT_PCT = 0.0005         # 0.05% minimum net profit after ALL costs
MIN_EXPECTED_RETURN_PCT = 0.0010    # 0.10% (10 bps) minimum predicted return from ensemble
PROFIT_TARGET_PCT = 0.012           # 1.20% Take-Profit target (optimized for 5m intraday completion to hit 10-15 trades/day)
MAX_GROSS_LOSS_PCT = 0.006          # 0.60% initial Stop-Loss limit (strict 2:1 reward-to-risk ratio)
TRAILING_STOP_ACTIVATION_PCT = 0.006 # Ratchet stop-loss to Breakeven (+0.25%) once trade reaches +0.60%
TRAILING_STOP_DISTANCE_PCT = 0.004  # Trails 0.40% below peak watermark
AUTO_EOD_SQUAREOFF_HOUR = 15        # Auto square-off at 3:20 PM IST to eliminate overnight gap-down risk
AUTO_EOD_SQUAREOFF_MINUTE = 20
DEFAULT_CAPITAL = 10000             # Default starting capital (INR) -- overridable from UI

# --- Position Sizing (dynamic, confidence-based) -----------------------------
# Percentage of AVAILABLE (uninvested) capital to allocate per trade
POSITION_SIZING = {
    "high":   0.25,   # High conviction -> larger allocation
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
