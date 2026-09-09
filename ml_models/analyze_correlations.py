import pandas as pd
import numpy as np
import os
import requests

def fetch_klines(symbol, interval='5m', limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url).json()
    df = pd.DataFrame(res, columns=['ot','o','h','l','c','v','ct','qv','nt','tbb','tbq','i'])
    for col in ['o','h','l','c','v','tbb']: df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['ot'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume','tbb':'taker_buy_base_asset_volume'})
    return df

def analyze_correlations(symbol='BTCUSDT'):
    print(f"--- Feature-Target Correlation Analysis for {symbol} ---")
    df = fetch_klines(symbol)
    
    # Target
    df['target_ret'] = df['close'].shift(-1) / df['close'] - 1
    
    # Features (simplified from pipeline)
    df['rsi'] = pd.Series(df['close']).diff().apply(lambda x: max(x,0)).rolling(14).mean() / \
               df['close'].diff().abs().rolling(14).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']
    df['trend'] = df['close'].ewm(span=20).mean() - df['close'].ewm(span=50).mean()
    
    v_taker = df['taker_buy_base_asset_volume']
    v_total = df['volume']
    df['vol_delta'] = v_taker - (v_total - v_taker)
    
    features = ['rsi', 'vwap_dist', 'trend', 'vol_delta', 'volume']
    
    corrs = df[features + ['target_ret']].corr()['target_ret'].drop('target_ret')
    print(corrs)

if __name__ == "__main__":
    analyze_correlations('BTCUSDT')
    analyze_correlations('ETHUSDT')
