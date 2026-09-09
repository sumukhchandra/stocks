import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Focus on the primary market drivers and the user's focus coin
SYMBOLS = ['BTCUSDT', 'XRPUSDT']
LOG_FILE = 'QA/logs/pattern_analysis.csv'

def fetch_order_book(symbol, limit=20):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    try:
        data = requests.get(url).json()
        bids = np.array(data['bids'], dtype=float)
        asks = np.array(data['asks'], dtype=float)
        return bids, asks
    except Exception as e:
        print(f"Error fetching order book for {symbol}: {e}")
        return None, None

def analyze_patterns():
    print(f" Enhanced Pattern Analyzer Started. Monitoring {SYMBOLS}...")
    
    last_btc_price = None
    
    while True:
        timestamp = datetime.now().isoformat()
        results = []
        
        for symbol in SYMBOLS:
            bids, asks = fetch_order_book(symbol)
            if bids is None or asks is None:
                continue
                
            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            bid_vol = bids[:, 1].sum()
            ask_vol = asks[:, 1].sum()
            
            # 1. Order Book Imbalance (OBI)
            obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)
            
            # 2. Identify "Whale Walls" (Top 5 levels)
            top_bid_wall = bids[np.argmax(bids[:5, 1]), 1]
            top_ask_wall = asks[np.argmax(asks[:5, 1]), 1]
            
            # 3. Microprice
            microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
            
            results.append({
                'symbol': symbol,
                'price': (best_bid + best_ask) / 2,
                'obi': obi,
                'microprice': microprice,
                'bid_vol': bid_vol,
                'ask_vol': ask_vol,
                'top_bid_wall': top_bid_wall,
                'top_ask_wall': top_ask_wall
            })
            
        # 4. Correlation Check (BTC vs XRP)
        if len(results) == 2:
            btc = results[0]
            xrp = results[1]
            
            # Log the patterns
            log_data = {
                'timestamp': timestamp,
                'btc_price': btc['price'],
                'xrp_price': xrp['price'],
                'btc_obi': btc['obi'],
                'xrp_obi': xrp['obi'],
                'btc_microprice': btc['microprice'],
                'xrp_microprice': xrp['microprice'],
                'btc_bid_vol': btc['bid_vol'],
                'btc_ask_vol': btc['ask_vol']
            }
            
            df = pd.DataFrame([log_data])
            file_exists = pd.io.common.file_exists(LOG_FILE)
            df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
            
            # Provide high-signal output to the terminal
            print(f" {datetime.now().strftime('%H:%M:%S')} | BTC: ${btc['price']:.2f} (OBI: {btc['obi']:+.2f}) | XRP: ${xrp['price']:.4f} (OBI: {xrp['obi']:+.2f})")
            
            if abs(btc['obi']) > 0.5:
                side = "BUYERS" if btc['obi'] > 0 else "SELLERS"
                print(f"[!] WHALE ALERT: {side} dominating BTC order book!")
            
            if xrp['obi'] > 0.6 and btc['obi'] > 0.2:
                print(f" PATTERN MATCH: High-conviction long setup detected for XRP.")

        time.sleep(60) # Analyze every minute

if __name__ == "__main__":
    analyze_patterns()
