import pandas as pd
import numpy as np
import os
import sys
import requests
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.final_ensemble_engine import FinalEnsembleEngine
from features.technical import calculate_rsi, calculate_vwap, calculate_atr

def fetch_recent_klines(symbol, interval='5m', limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']:
            df[col] = df[col].astype(float)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['symbol'] = symbol
        return df
    except:
        return None

def build_live_features(df):
    group = df.copy().sort_values('timestamp')
    group['rsi_14'] = calculate_rsi(group, period=14, price_col='close')
    group['vwap'] = calculate_vwap(group, price_col='close', volume_col='volume')
    group['vwap_dist'] = (group['close'] - group['vwap']) / group['vwap']
    group['atr_14'] = calculate_atr(group, period=14)
    group['ema_20'] = group['close'].ewm(span=20).mean()
    group['ema_50'] = group['close'].ewm(span=50).mean()
    group['trend_strength'] = (group['ema_20'] - group['ema_50']) / group['ema_50']
    group['volatility'] = group['close'].pct_change().rolling(24).std()
    
    v_taker = group['taker_buy_base_asset_volume']
    v_total = group['volume']
    group['volume_delta'] = v_taker - (v_total - v_taker)
    group['vpin'] = group['volume_delta'].abs().rolling(24).sum() / (v_total.rolling(24).sum() + 1e-8)

    group['imbalance'] = 0.0
    group['spread'] = 0.0
    group['ofi'] = 0.0
    group['crisis_flag'] = 0
    group['last_funding_rate'] = 0.0
    group['last_macro_severity'] = 0
    group['macro_risk_factor'] = 1.0
    group['days_until_macro'] = 10
    group['days_since_macro'] = 10
    
    group['regime'] = 'sideways'
    group.loc[group['trend_strength'] > 0.015, 'regime'] = 'trending_bull'
    group.loc[group['trend_strength'] < -0.015, 'regime'] = 'trending_bear'
    
    return group.dropna().tail(50)

def main():
    print("="*50)
    print("GEMINI 5M SCALP MONITOR")
    print("="*50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models', 'saved_models')
    engine = FinalEnsembleEngine(model_dir=model_dir)
    
    universe = ['BTCUSDT', 'ETHUSDT']
    threshold = 0.36
    
    for symbol in universe:
        raw_data = fetch_recent_klines(symbol, interval='5m')
        if raw_data is None: continue
        features = build_live_features(raw_data)
        if len(features) < 1: continue
        
        try:
            decision = engine.evaluate_state(features)
            prob = decision['final_probability']
            trend = features['trend_strength'].iloc[-1]
            score = prob + (abs(trend) * 5)
            print(f"{symbol:<10} | Prob: {prob:.2%} | Trend: {trend:.4f} | Score: {score:.4f}")
        except: continue

if __name__ == "__main__":
    main()
