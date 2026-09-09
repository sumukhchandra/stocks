import pandas as pd
import numpy as np
import os
import sys

# Add paths for indicator imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.technical import calculate_rsi, calculate_vwap, calculate_atr

def process_data():
    raw_path = 'QA/data/raw/live_update/multi_asset_raw.parquet'
    if not os.path.exists(raw_path):
        print("Raw multi-asset data not found.")
        return

    df = pd.read_parquet(raw_path)
    processed_list = []
    
    print("Processing features and labels for multi-asset dataset...")
    
    for symbol, group in df.groupby('symbol'):
        group = group.copy().sort_values('timestamp')
        
        # 1. Technical Indicators
        group['rsi_14'] = calculate_rsi(group, period=14, price_col='close')
        group['vwap'] = calculate_vwap(group, price_col='close', volume_col='volume')
        group['vwap_dist'] = (group['close'] - group['vwap']) / group['vwap']
        group['atr_14'] = calculate_atr(group, period=14)
        group['volatility'] = group['close'].pct_change().rolling(24).std()
        
        # 2. Microstructure (Simplified proxies for kline data)
        group['volume_delta'] = group['taker_buy_base_asset_volume'] - (group['volume'] - group['taker_buy_base_asset_volume'])
        group['vpin'] = group['volume_delta'].abs().rolling(24).sum() / group['volume'].rolling(24).sum()
        
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
        # 1-hour lookahead (12 * 5m)
        group['future_return_t1'] = group['close'].shift(-12).pct_change(12).shift(-12)
        
        # Target 0.8% move for scalping
        group['target'] = (group['future_return_t1'] > 0.008).astype(int)
        
        # 4. Simple Regime Proxy
        group['returns_24h'] = group['close'].pct_change(288)
        group['regime'] = 'sideways'
        group.loc[(group['returns_24h'] > 0.02) & (group['volatility'] < group['volatility'].median()), 'regime'] = 'trending_bull'
        group.loc[(group['returns_24h'] < -0.02), 'regime'] = 'trending_bear'
        group.loc[group['volatility'] > group['volatility'].quantile(0.75), 'regime'] = 'high_volatility'
        
        # Placeholder cols for engine compatibility
        processed_list.append(group.dropna())

    if processed_list:
        final_df = pd.concat(processed_list, ignore_index=True)
        output_path = 'QA/data/processed/master_labeled_dataset.parquet'
        final_df.to_parquet(output_path)
        print(f"Processed dataset saved to {output_path} ({len(final_df)} rows)")
    else:
        print("Processing failed - no data.")

if __name__ == "__main__":
    process_data()
