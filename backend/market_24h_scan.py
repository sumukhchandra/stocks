import requests
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def fetch_crypto_data(symbol, interval='1h', limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    data = res.json()
    df = pd.DataFrame(data, columns=['ot','open','high','low','close','v','ct','qv','nt','tbb','tbq','i'])
    df['timestamp'] = pd.to_datetime(df['ot'], unit='ms')
    for col in ['open','high','low','close','v']: 
        df[col] = df[col].astype(float)
    return df[['timestamp', 'close', 'v']]

def fetch_sentiment():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1").json()
        fng_value = int(res['data'][0]['value'])
        sentiment_score = (fng_value - 50) / 50.0 
        return sentiment_score
    except:
        return 0.0

def run_market_scan():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ZECUSDT', 'DOGEUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    print(f" Scanning Universe: {', '.join([s.replace('USDT','') for s in symbols])}")
    
    sentiment_score = fetch_sentiment()
    print(f" Global Market Sentiment Score: {sentiment_score:+.2f}")
    
    df_btc = fetch_crypto_data('BTCUSDT', limit=2000)
    df_btc = df_btc[['timestamp', 'close']].rename(columns={'close':'btc_close'})
    
    results = []
    
    for symbol in symbols:
        df_coin = fetch_crypto_data(symbol, limit=2000)
        if df_coin is None: continue
        
        df_coin = df_coin.rename(columns={'close':'coin_close', 'v':'coin_vol'})
        df = pd.merge(df_coin, df_btc, on='timestamp', how='inner')
        
        # Noise reduction
        df['coin_ret'] = df['coin_close'].pct_change()
        std = df['coin_ret'].std()
        mean = df['coin_ret'].mean()
        df['coin_ret'] = np.clip(df['coin_ret'], mean - 3*std, mean + 3*std)
        
        df['btc_ret'] = df['btc_close'].pct_change()
        df['rel_strength'] = df['coin_close'] / df['btc_close']
        df['rs_trend'] = df['rel_strength'].pct_change(24)
        df['sma_24'] = df['coin_close'].rolling(24).mean()
        df['volatility_24'] = df['coin_ret'].rolling(24).std()
        
        df.dropna(inplace=True)
        
        current_rs_trend = df['rs_trend'].iloc[-1]
        tech_bullish = current_rs_trend > 0
        sent_bullish = sentiment_score > 0
        
        simulation_active = False
        if not ((tech_bullish and sent_bullish) or (not tech_bullish and not sent_bullish)):
            simulation_active = True
            
        # Target 24h
        df['target_24h'] = df['coin_close'].shift(-24)
        features = ['coin_close', 'coin_vol', 'btc_close', 'coin_ret', 'btc_ret', 'rel_strength', 'rs_trend', 'sma_24', 'volatility_24']
        train_df = df.dropna(subset=features + ['target_24h'])
        
        X = train_df[features]
        y = train_df['target_24h']
        
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(X, y)
        
        current_state = df.iloc[-1:]
        current_price = current_state['coin_close'].iloc[0]
        pred_24h = model.predict(current_state[features])[0]
        
        if simulation_active:
            shock_factor = sentiment_score * current_state['volatility_24'].iloc[0] * 10
            pred_24h = pred_24h * (1 + shock_factor)
            
        change_pct = ((pred_24h - current_price) / current_price) * 100
        
        results.append({
            'Coin': symbol.replace('USDT', ''),
            'Current_Price': current_price,
            'Pred_24h': pred_24h,
            'Expected_Move': change_pct,
            'Tech_Strength': current_rs_trend * 100
        })
        
    res_df = pd.DataFrame(results).sort_values(by='Expected_Move', ascending=False)
    
    print("\n" + "="*60)
    print("24-HOUR MARKET SCAN (SENTIMENT ADJUSTED)")
    print("="*60)
    for idx, row in res_df.iterrows():
        direction = " UP" if row['Expected_Move'] > 0 else " DOWN"
        print(f"{row['Coin']:<5} | Move: {row['Expected_Move']:>+6.2f}% {direction} | Tech Str: {row['Tech_Strength']:>+6.2f}% | Price: ${row['Current_Price']:.4f}")
    print("="*60)
    
    best_coin = res_df.iloc[0]
    print(f"\n BEST 24-HOUR PICK: {best_coin['Coin']}")
    if best_coin['Expected_Move'] < 0:
        print("[!] WARNING: Even the 'best' coin is projected to lose money. Cash is the safest position.")

if __name__ == '__main__':
    run_market_scan()
