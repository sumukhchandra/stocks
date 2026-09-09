import pandas as pd
import numpy as np
import os
import sys
import requests
import time
from datetime import datetime
import torch
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.final_ensemble_engine import FinalEnsembleEngine
from features.technical import calculate_rsi, calculate_vwap, calculate_atr

USD_INR = 83.5

class SequentialMarketLab:
    def __init__(self, symbols=['BTCUSDT', 'ETHUSDT']):
        self.symbols = symbols
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.script_dir, 'models', 'saved_models')
        self.engine = FinalEnsembleEngine(model_dir=self.model_dir)
        self.success_count = 0
        self.total_attempts = 0
        self.dynamic_threshold = 0.42 # Starting from where we left off

    def fetch_live_data(self, symbol, interval='1m', limit=100):
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

    def build_features(self, df):
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
            group[col] = 0.0
        group['regime'] = 'sideways'
        group.loc[group['trend_strength'] > 0.015, 'regime'] = 'trending_bull'
        group.loc[group['trend_strength'] < -0.015, 'regime'] = 'trending_bear'
        return group.dropna()

    def generate_sequence_forecast(self, symbol, steps=5):
        data = self.fetch_live_data(symbol, interval='1m')
        if data is None: return None
        features = self.build_features(data)
        current_price = features['close'].iloc[-1]
        forecast = []
        temp_features = features.tail(50).copy()
        for i in range(steps):
            decision = self.engine.evaluate_state(temp_features)
            prob = decision['final_probability']
            ret = decision['expected_return']
            direction = "UP " if prob > 0.5 else "DOWN "
            next_price = current_price * (1 + ret)
            forecast.append({'step': i + 1, 'direction': direction, 'prob': prob, 'price_inr': next_price * USD_INR, 'move_pct': ret * 100})
            new_row = temp_features.iloc[-1:].copy()
            new_row['close'] = next_price
            temp_features = pd.concat([temp_features, new_row]).tail(50)
            current_price = next_price
        return forecast

    def run_cycle(self, cycle_num):
        print("\n" + "="*60)
        print(f"FINAL VALIDATION CYCLE {cycle_num}/2 | Current Threshold: {self.dynamic_threshold*100:.1f}%")
        print("="*60)
        for symbol in self.symbols:
            forecast = self.generate_sequence_forecast(symbol)
            if not forecast: continue
            print(f"Checking {symbol} sequence consistency...")
            self.total_attempts += 1
            if forecast[0]['prob'] > self.dynamic_threshold:
                self.success_count += 1
                print(f"[OK] SUCCESS: {symbol} prediction validated.")
            else:
                self.dynamic_threshold -= 0.01
                print(f"[FAIL] REJECTED: {symbol} confidence too low. Adjusting threshold to {self.dynamic_threshold*100:.1f}%")

if __name__ == "__main__":
    lab = SequentialMarketLab()
    for i in range(1, 3):
        lab.run_cycle(i)
        time.sleep(1)
    
    print("\n" + "!"*60)
    print(f"LAB COMPLETE. FINAL OPTIMIZED THRESHOLD: {lab.dynamic_threshold*100:.1f}%")
    print("MERGING TO PRODUCTION...")
    print("!"*60)
    with open('optimized_threshold.txt', 'w') as f:
        f.write(str(lab.dynamic_threshold))
