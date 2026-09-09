import pandas as pd
import numpy as np
import os
import sys
import requests
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.final_ensemble_engine import FinalEnsembleEngine
from features.technical import calculate_rsi, calculate_vwap, calculate_atr

def fetch_live_market_context():
    """Fetch 1H data for top coins to check for breakouts."""
    universe = ['BTCUSDT', 'ETHUSDT']
    data = []
    print(f"\n[VOLATILITY ALERT] Scanning {len(universe)} coins for breakout pressure...")
    
    for symbol in universe:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=50"
        try:
            res = requests.get(url).json()
            df = pd.DataFrame(res, columns=['ot','o','h','l','c','v','ct','qv','nt','tbb','tbq','i'])
            close = df['c'].astype(float)
            high = df['h'].astype(float)
            low = df['l'].astype(float)
            vol = df['v'].astype(float)
            
            # Volatility Pressure = (Current Range / Avg Range) * (Current Vol / Avg Vol)
            current_range = high.iloc[-1] - low.iloc[-1]
            avg_range = (high - low).rolling(24).mean().iloc[-1]
            
            vol_surge = vol.iloc[-1] / vol.rolling(24).mean().iloc[-1]
            range_expansion = current_range / avg_range
            
            # Pressure Score (Higher = closer to breakout)
            pressure_score = (vol_surge * 0.6) + (range_expansion * 0.4)
            
            data.append({
                'symbol': symbol,
                'price': close.iloc[-1],
                'pressure': pressure_score,
                'vol_surge': vol_surge,
                'range_exp': range_expansion
            })
        except: continue
    return data

def main():
    print("="*50)
    print("GEMINI BREAKOUT RADAR & VOLATILITY ALERT")
    print(f"Update Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    alerts = fetch_live_market_context()
    if not alerts:
        print("No market data available.")
        return

    # Sort by Pressure Score
    alerts = sorted(alerts, key=lambda x: x['pressure'], reverse=True)

    print(f"\n{'SYMBOL':<10} | {'PRICE':<12} | {'VOL SURGE':<10} | {'PRESSURE'}")
    print("-" * 50)
    for a in alerts:
        status = " HIGH" if a['pressure'] > 1.5 else "[*] LOW"
        print(f"{a['symbol']:<10} | ${a['price']:<11.4f} | {a['vol_surge']:<10.2f} | {a['pressure']:.2f} [{status}]")

    top = alerts[0]
    print("\n" + "="*50)
    if top['pressure'] > 1.5:
        print(f" ALERT: {top['symbol']} is under HEAVY pressure ({top['pressure']:.2f}).")
        print(f"A breakout of 5% - 15% is becoming LIKELY in the next 4-12 hours.")
    else:
        print(f"NOTICE: {top['symbol']} is the most active, but the market is still calm.")
        print("Recommendation: Stay in cash. No high-growth breakout detected yet.")
    print("="*50)

if __name__ == "__main__":
    main()
