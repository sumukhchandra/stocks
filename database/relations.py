"""
database.relations: Relational mappings for stocks, sectors, and cross-asset dependencies.
"""

STOCK_METADATA = {
    "RELIANCE.NS": {
        "name": "Reliance Industries Ltd",
        "sector": "Energy / Conglomerate",
        "nifty_weight": 0.098,
        "beta": 1.05,
        "lot_liquidity": "Ultra-High"
    },
    "HDFCBANK.NS": {
        "name": "HDFC Bank Ltd",
        "sector": "Banking & Financials",
        "nifty_weight": 0.115,
        "beta": 1.10,
        "lot_liquidity": "Ultra-High"
    },
    "BHARTIARTL.NS": {
        "name": "Bharti Airtel Ltd",
        "sector": "Telecommunications",
        "nifty_weight": 0.045,
        "beta": 0.85,
        "lot_liquidity": "High"
    },
    "ICICIBANK.NS": {
        "name": "ICICI Bank Ltd",
        "sector": "Banking & Financials",
        "nifty_weight": 0.078,
        "beta": 1.15,
        "lot_liquidity": "Ultra-High"
    },
    "TATACAP.NS": {
        "name": "Tata Capital Ltd",
        "sector": "Non-Banking Financials (NBFC)",
        "nifty_weight": 0.015,
        "beta": 1.25,
        "lot_liquidity": "Medium-High"
    },
    "INFY.NS": {
        "name": "Infosys Ltd",
        "sector": "Information Technology",
        "nifty_weight": 0.062,
        "beta": 0.95,
        "lot_liquidity": "Ultra-High"
    },
    "TCS.NS": {
        "name": "Tata Consultancy Services Ltd",
        "sector": "Information Technology",
        "nifty_weight": 0.048,
        "beta": 0.80,
        "lot_liquidity": "Ultra-High"
    },
    "SBIN.NS": {
        "name": "State Bank of India",
        "sector": "Banking & Financials",
        "nifty_weight": 0.035,
        "beta": 1.20,
        "lot_liquidity": "Ultra-High"
    },
    "AXISBANK.NS": {
        "name": "Axis Bank Ltd",
        "sector": "Banking & Financials",
        "nifty_weight": 0.033,
        "beta": 1.15,
        "lot_liquidity": "High"
    },
    "BAJFINANCE.NS": {
        "name": "Bajaj Finance Ltd",
        "sector": "Non-Banking Financials (NBFC)",
        "nifty_weight": 0.027,
        "beta": 1.30,
        "lot_liquidity": "High"
    },
}

SECTORS = {
    "Banking & Financials": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS"],
    "Information Technology": ["INFY.NS", "TCS.NS"],
    "Non-Banking Financials (NBFC)": ["TATACAP.NS", "BAJFINANCE.NS"],
    "Energy / Conglomerate": ["RELIANCE.NS"],
    "Telecommunications": ["BHARTIARTL.NS"],
}

STOCK_SECTOR_MAP = {s: meta["sector"] for s, meta in STOCK_METADATA.items()}

def get_stock_sector(symbol):
    """Return the sector name for a given stock symbol."""
    return STOCK_METADATA.get(symbol, {}).get("sector", "Unknown")

def get_sector_stocks(sector):
    """Return all stock symbols in a given sector."""
    return SECTORS.get(sector, [])

def get_correlated_peers(symbol):
    """Return peers in the same sector."""
    sector = get_stock_sector(symbol)
    peers = get_sector_stocks(sector)
    return [p for p in peers if p != symbol]
