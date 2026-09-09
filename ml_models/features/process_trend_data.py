import pandas as pd
import numpy as np
import os
import sys

# Add paths for indicator imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.technical import calculate_rsi, calculate_vwap, calculate_atr

def process_trend_data():
    raw_path = 'QA/data/raw/live_update/trend_raw_1h.parquet'
    if not os.path.exists(raw_path):
        print("Raw trend data not found.")
        return

    df = pd.read_parquet(raw_path)
    processed_list = []
    
    print("Engineering features for Trend Strategy (1H Horizon)...")
    
    for symbol, group in df.groupby('symbol'):
        group = group.copy().sort_values('timestamp')
        
        # 1. Technical Indicators (Clean signals)
        group['rsi_14'] = calculate_rsi(group, period=14, price_col='close')
        group['vwap'] = calculate_vwap(group, price_col='close', volume_col='volume')
        group['vwap_dist'] = (group['close'] - group['vwap']) / group['vwap']
        group['atr_14'] = calculate_atr(group, period=14)
        
        # Trend specific features
        group['ema_20'] = group['close'].ewm(span=20).mean()
        group['ema_50'] = group['close'].ewm(span=50).mean()
        group['trend_strength'] = (group['ema_20'] - group['ema_50']) / group['ema_50']
        group['volatility'] = group['close'].pct_change().rolling(24).std()
        
        # 2. Microstructure (Simplified proxies for kline data)
        # Ensure all components are float to avoid TypeError with 'str'
        v_taker = group['taker_buy_base_asset_volume'].astype(float)
        v_total = group['volume'].astype(float)
        
        group['volume_delta'] = v_taker - (v_total - v_taker)
        group['vpin'] = group['volume_delta'].abs().rolling(24).sum() / (v_total.rolling(24).sum() + 1e-8)
        
        # 3. Labeling (3% Target over 24H horizon)
        # Look ahead 24 hours (24 bars of 1h)
        group['future_max_high'] = group['high'].shift(-24).rolling(24).max().shift(-24)
        group['future_return_3%'] = (group['future_max_high'] - group['close']) / group['close']
        
        # Target 1 if price hits 3% gain before significant drawdown
        group['target'] = (group['future_return_3%'] > 0.03).astype(int)
        group['future_return_t1'] = group['close'].shift(-24).pct_change(24).shift(-24) # 24h benchmark return
        
        # 4. Regime Detection (Trend vs Mean Reverting)
        group['regime'] = 'sideways'
        group.loc[group['trend_strength'] > 0.015, 'regime'] = 'trending_bull'
        group.loc[group['trend_strength'] < -0.015, 'regime'] = 'trending_bear'
        
        # Mock features for engine compatibility (originally from orderbook/macro data)
        group['imbalance'] = 0.0
        group['spread'] = 0.0
        group['ofi'] = 0.0
        group['crisis_flag'] = 0
        group['last_funding_rate'] = 0.0
        group['last_macro_severity'] = 0
        group['macro_risk_factor'] = 1.0
        group['days_until_macro'] = 10
        group['days_since_macro'] = 10
        
        processed_list.append(group.dropna())

    if processed_list:
        final_df = pd.concat(processed_list, ignore_index=True)
        output_path = 'QA/data/processed/master_labeled_dataset.parquet'
        final_df.to_parquet(output_path)
        print(f"Trend dataset saved to {output_path} ({len(final_df)} rows)")
    else:
        print("Processing failed.")

if __name__ == "__main__":
    process_trend_data()
