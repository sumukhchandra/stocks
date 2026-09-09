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

# --- Trading Thresholds ------------------------------------------------------
MIN_NET_PROFIT_PCT = 0.0005         # 0.05% minimum net profit after ALL costs to enter/exit
MAX_GROSS_LOSS_PCT = 0.003          # 0.3% max gross loss before stop-loss triggers
DEFAULT_CAPITAL = 100000            # Default starting capital (INR) -- overridable from UI

# --- Position Sizing (dynamic, confidence-based) -----------------------------
# Percentage of AVAILABLE (uninvested) capital to allocate per trade
POSITION_SIZING = {
    "high":   0.15,   # Confidence > 0.8  -> 15% of available capital
    "medium": 0.10,   # Confidence 0.7-0.8 -> 10%
    "low":    0.05,   # Confidence 0.6-0.7 -> 5%
}

# --- Scanning & Timing -------------------------------------------------------
SCAN_INTERVAL_SECONDS = 180         # Scan market every 3 minutes
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
