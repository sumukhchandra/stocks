import requests
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# The requested universe
UNIVERSE = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ZECUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
IMPLIED_RATE = 100.83 # USDT to INR Reference

def fetch_and_clean(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=1000"
    try:
        data = requests.get(url).json()
        df = pd.DataFrame(data, columns=['ot','open','high','low','close','v','ct','qv','nt','tbb','tbq','i'])
        df['timestamp'] = pd.to_datetime(df['ot'], unit='ms') + timedelta(hours=5, minutes=30)
        for col in ['open','high','low','close','v']: df[col] = df[col].astype(float)
        
        # Rigorous Cleaning (Wipe out unwanted data/noise)
        df['returns'] = df['close'].pct_change()
        df = df.dropna()
        # Filter price spikes (Z > 3) and volume anomalies (Z > 5)
        df = df[(np.abs(stats.zscore(df['returns'])) < 3)]
        df = df[(stats.zscore(df['v']) < 5)]
        
        return df[['timestamp', 'close', 'v', 'returns']].reset_index(drop=True)
    except:
        return None

def forecast_coin(symbol):
    df = fetch_and_clean(symbol)
    if df is None or len(df) < 500: return None
    
    # Feature Engineering
    df['volatility'] = df['returns'].rolling(12).std()
    df['ema_1h'] = df['close'].ewm(span=12).mean()
    df['ema_3h'] = df['close'].ewm(span=36).mean()
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().where(df['close'].diff() > 0, 0).rolling(14).mean() / 
                                  (df['close'].diff().where(df['close'].diff() < 0, 0).abs().rolling(14).mean() + 1e-9))))
    df = df.dropna()
    
    features = ['close', 'v', 'returns', 'volatility', 'ema_1h', 'ema_3h', 'rsi']
    current_state = df.iloc[-1:]
    current_price = current_state['close'].iloc[0]
    
    horizons = [12, 24, 36, 48, 60] # 60, 120, 180, 240, 300 mins
    results = {}
    
    for step in horizons:
        df['target'] = df['close'].shift(-step)
        train_df = df.dropna(subset=features + ['target'])
        
        model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(train_df[features], train_df['target'])
        
        pred = model.predict(current_state[features])[0]
        results[f'{step*5}m'] = ((pred - current_price) / current_price) * 100
        
    return {'symbol': symbol, 'price_inr': current_price * IMPLIED_RATE, 'moves': results}

def main():
    print(f" Starting Global Opportunity Scan (Local Time: {datetime.now().strftime('%H:%M:%S')})")
    print(" Cleaning market noise and analyzing 9 coins one-by-one...")
    
    all_results = []
    for symbol in UNIVERSE:
        print(f"   -> Analyzing {symbol}...")
        res = forecast_coin(symbol)
        if res: all_results.append(res)
    
    # Ranking
    print("\n" + "="*85)
    print(f"{'COIN':<10} | {'PRICE(INR)':<12} | {'60m':<8} | {'120m':<8} | {'180m':<8} | {'240m':<8} | {'300m':<8}")
    print("-" * 85)
    
    for r in all_results:
        m = r['moves']
        print(f"{r['symbol'].replace('USDT',''):<10} | Rs.{r['price_inr']:<11.2f} | {m['60m']:>+6.2f}% | {m['120m']:>+6.2f}% | {m['180m']:>+6.2f}% | {m['240m']:>+6.2f}% | {m['300m']:>+6.2f}%")
    
    print("="*85)
    
    # Recommendation
    best_short = max(all_results, key=lambda x: x['moves']['60m'])
    best_long = max(all_results, key=lambda x: x['moves']['300m'])
    
    print(f"\n BEST SHORT-TERM (60m): {best_short['symbol'].replace('USDT','')} ({best_short['moves']['60m']:+.2f}%)")
    print(f" BEST MID-TERM (300m): {best_long['symbol'].replace('USDT','')} ({best_long['moves']['300m']:+.2f}%)")
    
    inv_amount = 1000
    expected_profit = (inv_amount * best_long['moves']['300m'] / 100)
    print(f"\n INVESTMENT STRATEGY FOR 1000:")
    print(f"   Allocation: Buy {best_long['symbol'].replace('USDT','')} at current price.")
    print(f"   Target Profit (300m): ~{expected_profit:+.2f} (Total: {inv_amount + expected_profit:.2f})")

if __name__ == "__main__":
    main()
