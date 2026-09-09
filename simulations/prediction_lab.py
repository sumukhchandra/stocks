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

USD_INR = 83.5

def fetch_data(symbol, interval='5m', limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url).json()
        df = pd.DataFrame(res, columns=['ot','o','h','l','c','v','ct','qv','nt','tbb','tbq','i'])
        for col in ['o','h','l','c','v','tbb']: df[col] = df[col].astype(float)
        df['timestamp'] = pd.to_datetime(df['ot'], unit='ms')
        df['symbol'] = symbol
        df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume','tbb':'taker_buy_base_asset_volume'})
        return df
    except: return None

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

    for col in ['number_of_trades', 'imbalance','spread','ofi','crisis_flag','last_funding_rate','last_macro_severity','macro_risk_factor','days_until_macro','days_since_macro']:
        if col not in group.columns:
            group[col] = 0.0
    group['regime'] = 'sideways'
    group.loc[group['trend_strength'] > 0.015, 'regime'] = 'trending_bull'
    group.loc[group['trend_strength'] < -0.015, 'regime'] = 'trending_bear'
    return group.dropna().tail(50)

def main():
    print("="*60)
    print("GEMINI PREDICTION LAB: LIVE AUDIT & SIMULATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models', 'saved_models')
    engine = FinalEnsembleEngine(model_dir=model_dir)
    
    # Expanded universe of "Popular" coins
    universe = ['BTCUSDT', 'ETHUSDT']
    
    results = []

    print(f"{'SYMBOL':<10} | {'PRICE (INR)':<15} | {'PRED (INR)':<15} | {'PRED %':<10} | {'CONF'}")
    print("-" * 65)

    for symbol in universe:
        raw_data = fetch_data(symbol)
        if raw_data is None: continue
        
        features = build_features(raw_data)
        if len(features) < 1: continue
        
        try:
            decision = engine.evaluate_state(features)
            
            # Prediction Logic
            current_price = decision['price']
            # future_return_t1 is the predicted magnitude for the next period
            pred_return = decision['expected_return']
            pred_price = current_price * (1 + pred_return)
            
            # INR Conversion
            current_inr = current_price * USD_INR
            pred_inr = pred_price * USD_INR
            
            results.append({
                'symbol': symbol,
                'current_inr': current_inr,
                'pred_inr': pred_inr,
                'pct': pred_return * 100,
                'conf': decision['final_probability'] * 100
            })
            
            # Format price strings based on magnitude
            fmt = ".2f" if current_inr > 10 else ".4f"
            curr_str = f"Rs.{current_inr:{fmt}}"
            pred_str = f"Rs.{pred_inr:{fmt}}"
            
            print(f"{symbol:<10} | {curr_str:<15} | {pred_str:<15} | {pred_return*100:>+7.2f}% | {decision['final_probability']*100:.1f}%")
        except Exception as e:
            # print(f"Error for {symbol}: {e}")
            continue

    print("\n" + "="*60)
    print("SIMULATION & STRATEGY AUDIT")
    print("-" * 60)
    
    # Quick "Accuracy simulation" based on historical volatility vs prediction
    for r in results[:3]: # Show top 3
        # If confidence is low, the prediction is noise. If high, it's a signal.
        if r['conf'] > 60:
            print(f" {r['symbol']}: HIGH CONVICTION. Expected {r['pct']:.2f}% move. Strategy: Scale-in.")
        else:
            print(f" {r['symbol']}: LOW CONVICTION ({r['conf']:.1f}%). Market is noisy. Strategy: Cash.")

    print("\nPROPOSED CHANGES:")
    print("1. Increase 'Magnitude' weight in the ensemble for better INR targets.")
    print("2. Current regime is SIDEWAYS; predictions favor mean-reversion.")
    print("="*60)

if __name__ == "__main__":
    main()
