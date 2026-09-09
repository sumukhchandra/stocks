import requests
import pandas as pd
import time
import os
from datetime import datetime

# Expanded Universe based on user request
MONITOR_UNIVERSE = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ZECUSDT', 
    'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'LINKUSDT', 
    'MATICUSDT', 'SHIBUSDT', 'PEPEUSDT', 'XLMUSDT', 'AVAXUSDT',
    'LTCUSDT', 'TRXUSDT', 'NEARUSDT'
]
LOG_FILE = 'QA/logs/background_market_knowledge.csv'

def fetch_ticker(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        data = requests.get(url).json()
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'price': float(data['lastPrice']),
            'change_24h_pct': float(data['priceChangePercent']),
            'volume': float(data['volume']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice'])
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def run_monitor():
    print(f" Background Monitor Started. Tracking {len(MONITOR_UNIVERSE)} coins.")
    if not os.path.exists('QA/logs'):
        os.makedirs('QA/logs')
        
    while True:
        results = []
        for symbol in MONITOR_UNIVERSE:
            ticker = fetch_ticker(symbol)
            if ticker:
                results.append(ticker)
        
        df = pd.DataFrame(results)
        
        # Append to CSV
        file_exists = os.path.isfile(LOG_FILE)
        df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
        
        print(f"[OK] {datetime.now().strftime('%H:%M:%S')} | Logged data for {len(results)} assets.")
        
        # Wait 15 minutes for next knowledge update
        time.sleep(900)

if __name__ == "__main__":
    run_monitor()
