import requests
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# High-volatility / Trending Universe
VOLATILE_UNIVERSE = [
    'ZECUSDT', 'RNDRUSDT', 'FETUSDT', 'INJUSDT', 'TIAUSDT', 
    'LDOUSDT', 'OPUSDT', 'ARBUSDT', 'SOLUSDT', 'PEPEUSDT'
]

def fetch_data(symbol, interval='1h', limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url).json()
        df = pd.DataFrame(res, columns=['ot','open','high','low','close','v','ct','qv','nt','tbb','tbq','i'])
        for col in ['open','high','low','close','v']: df[col] = df[col].astype(float)
        return df[['close', 'v']]
    except: return None

def fetch_sentiment():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1").json()
        return (int(res['data'][0]['value']) - 50) / 50.0
    except: return 0.0

def analyze_high_growth():
    print(f" Scanning High-Volatility Universe for 1-3% Targets...")
    sentiment_score = fetch_sentiment()
    
    results = []
    for symbol in VOLATILE_UNIVERSE:
        df = fetch_data(symbol)
        if df is None or len(df) < 100: continue
        
        # Features
        df['ret'] = df['close'].pct_change()
        df['volatility'] = df['ret'].rolling(24).std()
        df['ema_24'] = df['close'].ewm(span=24).mean()
        
        df = df.dropna()
        features = ['close', 'v', 'ret', 'volatility', 'ema_24']
        
        # Target: 24h Return
        df['target'] = df['close'].shift(-24)
        train_df = df.dropna()
        
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(train_df[features], train_df['target'])
        
        current_state = df.iloc[-1:]
        pred_24h = model.predict(current_state[features])[0]
        
        # Sentiment Adjustment (Simulation)
        shock = sentiment_score * current_state['volatility'].iloc[0] * 10
        pred_24h = pred_24h * (1 + shock)
        
        change_pct = ((pred_24h - current_state['close'].iloc[0]) / current_state['close'].iloc[0]) * 100
        
        results.append({
            'symbol': symbol,
            'current': current_state['close'].iloc[0],
            'pred_24h': pred_24h,
            'change': change_pct
        })
        print(f"   -> {symbol}: {change_pct:+.2f}% predicted")

    print("\n" + "="*50)
    print(f"{'COIN':<10} | {'PREDICTED 24H MOVE':<20}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x['change'], reverse=True):
        status = " TARGET MET" if 1.0 <= r['change'] <= 3.5 else ""
        print(f"{r['symbol'].replace('USDT',''):<10} | {r['change']:>+18.2f}% {status}")
    print("="*50)

if __name__ == "__main__":
    analyze_high_growth()
