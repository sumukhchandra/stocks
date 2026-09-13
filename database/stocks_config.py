"""Shared stock universe and trading configuration for the NSE trading system."""

# --- Stock Universe (30 high-liquidity NSE stocks for scalping) ---------------
STOCK_UNIVERSE = {
    # --- Tier 1: Banking & Finance (Highest intraday liquidity) ---
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "AXISBANK.NS": "Axis Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "INDUSINDBK.NS": "IndusInd Bank",
    # --- Tier 2: IT & Technology ---
    "INFY.NS": "Infosys",
    "TCS.NS": "TCS",
    "HCLTECH.NS": "HCL Technologies",
    "WIPRO.NS": "Wipro",
    "TECHM.NS": "Tech Mahindra",
    # --- Tier 3: Energy & Industrials ---
    "RELIANCE.NS": "Reliance Industries",
    "BHARTIARTL.NS": "Bharti Airtel",
    "LT.NS": "Larsen & Toubro",
    "TATASTEEL.NS": "Tata Steel",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "POWERGRID.NS": "Power Grid Corp",
    "NTPC.NS": "NTPC",
    # --- Tier 4: FMCG & Pharma ---
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "SUNPHARMA.NS": "Sun Pharma",
    "DRREDDY.NS": "Dr. Reddy's Labs",
    # --- Tier 5: Auto & Metals ---
    "MARUTI.NS": "Maruti Suzuki",
    "M&M.NS": "Mahindra & Mahindra",
    "JSWSTEEL.NS": "JSW Steel",
    "HINDALCO.NS": "Hindalco Industries",
    "COALINDIA.NS": "Coal India",
}

STOCK_SYMBOLS = list(STOCK_UNIVERSE.keys())

# --- Strategy Modes & Compounding Configuration ------------------------------
STRATEGY_MODE = "SINGLE_BULLET"      # "SINGLE_BULLET" | "MULTI_SPLIT" | "SCALP_COMPOUND"
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

# --- Scalp Compounding Mode Configuration ------------------------------------
SCALP_MIN_RETURN_PCT = 0.010         # 1.0% gross target per trade (guarantees >0.5% net after all costs)
SCALP_NET_PROFIT_TARGET = 0.005      # 0.5% net overall daily target after all taxes
SCALP_MIN_PROBABILITY = 0.62         # Minimum ML probability to enter a scalp trade
SCALP_MIN_TRADES_PER_SESSION = 10    # Find minimum 10 compound trades per day
SCALP_MAX_TRADES_PER_SESSION = 15    # Cap at 15 trades to prevent overtrading
SCALP_MAX_HOLD_BARS = 3              # Max 3 × 5-min = 15 mins per scalp trade
SCALP_TP_PCT = 0.012                 # 1.2% take-profit for scalp trades
SCALP_SL_PCT = 0.004                 # 0.4% stop-loss for scalp (tight, 3:1 R:R ratio)
SCALP_CHAIN_STOP_ON_LOSS = True      # Pause compounding chain for 10 min after a stop-loss hit
SCALP_CHAIN_PAUSE_MINUTES = 10       # Cooldown after a loss before resuming chain
SCALP_DRAWDOWN_HALT_PCT = 0.02       # Halt session if pool drops 2% below starting capital
COMPOUND_REINVEST_PCT = 1.0          # 100% of capital+profit reinvested into next trade
SCALP_SCAN_INTERVAL_SECONDS = 30     # Faster 30s scanning for scalp mode

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
