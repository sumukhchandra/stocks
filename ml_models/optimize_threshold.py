import pandas as pd
import numpy as np
import os
import sys
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_models.final_ensemble_engine import FinalEnsembleEngine
from features.technical import calculate_rsi, calculate_vwap, calculate_atr

def fetch_data(symbol, limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit={limit}"
    res = requests.get(url).json()
    df = pd.DataFrame(res, columns=['ot','o','h','l','c','v','ct','qv','nt','tbb','tbq','i'])
    for col in ['o','h','l','c','v','tbb']: df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['ot'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume','tbb':'taker_buy_base_asset_volume', 'nt':'number_of_trades'})
    return df

def build_features(df):
    group = df.copy().sort_values('timestamp')
    group['rsi_14'] = calculate_rsi(group, period=14, price_col='close')
    group['vwap'] = calculate_vwap(group, price_col='close', volume_col='volume')
    group['vwap_dist'] = (group['close'] - group['vwap']) / group['vwap']
    group['atr_14'] = calculate_atr(group, period=14)
    group['ema_20'] = group['close'].ewm(span=20).mean()
    group['ema_50'] = group['close'].ewm(span=50).mean()
    group['trend_strength'] = (group['ema_20'] - group['ema_50']) / group['ema_50']
    group['volatility'] = group['close'].pct_change().rolling(24).std()
    
    v_taker = group['taker_buy_base_asset_volume'].astype(float)
    v_total = group['volume'].astype(float)
    group['volume_delta'] = v_taker - (v_total - v_taker)
    group['vpin'] = group['volume_delta'].abs().rolling(24).sum() / (v_total.rolling(24).sum() + 1e-8)

    for col in ['imbalance','spread','ofi','crisis_flag','last_funding_rate','last_macro_severity','macro_risk_factor','days_until_macro','days_since_macro']:
        group[col] = 0.0
    group['regime'] = 'sideways'
    return group.dropna()

def optimize_threshold():
    engine = FinalEnsembleEngine(model_dir='ml_models/saved_models')
    universe = ['BTCUSDT', 'ETHUSDT']
    
    for symbol in universe:
        print(f"\n--- Optimizing for {symbol} ---")
        df = fetch_data(symbol)
        features = build_features(df)
        
        probs = []
        returns = []
        
        for i in range(50, len(features)-1):
            window = features.iloc[:i+1]
            try:
                decision = engine.evaluate_state(window)
                probs.append(decision['final_probability'])
                returns.append(features.iloc[i+1]['close'] / features.iloc[i]['close'] - 1)
            except: continue
            
        if not probs: continue
        results = pd.DataFrame({'prob': probs, 'ret': returns})
        
        best_thresh = 0.5
        best_pnl = -999
        
        for t in np.arange(0.4, 0.7, 0.01):
            trades = results[results['prob'] >= t]
            if len(trades) < 5: continue
            
            net_pnl = (trades['ret'] - 0.001).sum() # 10bps fee
            if net_pnl > best_pnl:
                best_pnl = net_pnl
                best_thresh = t
                
        print(f"Optimal Threshold for {symbol}: {best_thresh:.2f} (Net PnL: {best_pnl:.4f}, Trade Count: {len(results[results['prob'] >= best_thresh])})")

if __name__ == "__main__":
    optimize_threshold()
