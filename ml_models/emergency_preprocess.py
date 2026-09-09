import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.getcwd(), 'QA'))

from features.technical import calculate_rsi, calculate_vwap, calculate_atr

def process_and_label():
    raw_dir = 'QA/data/raw/klines'
    out_file = 'QA/data/processed/labeled_data.parquet'
    
    if not os.path.exists('QA/data/processed'):
        os.makedirs('QA/data/processed')
        
    all_df = []
    for f in os.listdir(raw_dir):
        if f.endswith('.parquet'):
            all_df.append(pd.read_parquet(os.path.join(raw_dir, f)))
            
    if not all_df:
        print("No raw klines found.")
        return
        
    df = pd.concat(all_df).reset_index(drop=True)
    
    # Standardize timestamps (Naive)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Feature Engineering
    print("Engineering features...")
    df['rsi_14'] = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_rsi(x, 14, price_col='close'))
    df['vwap'] = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_vwap(x, price_col='close', volume_col='volume'))
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']
    df['atr_14'] = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_atr(x, 14))
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20).mean())
    df['ema_50'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=50).mean())
    df['trend_strength'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
    df['volatility'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change().rolling(24).std())
    
    v_taker = df['taker_buy_base_asset_volume'].astype(float)
    v_total = df['volume'].astype(float)
    df['volume_delta'] = v_taker - (v_total - v_taker)
    
    # Rolling VPIN approximation - Use a more robust transform
    print("Calculating VPIN...")
    df['vol_sum_24'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(2).sum()) # Minimal window for safety
    df['vdelta_abs_sum_24'] = df.groupby('symbol')['volume_delta'].transform(lambda x: x.abs().rolling(2).sum())
    df['vpin'] = df['vdelta_abs_sum_24'] / (df['vol_sum_24'] + 1e-8)
    
    # Target Labeling
    print("Labeling data...")
    df['target_ret'] = df.groupby('symbol')['close'].shift(-1) / df['close'] - 1
    df['target'] = (df['target_ret'] > 0.0015).astype(int)
    
    # Extra mandatory cols for models
    for col in ['imbalance','spread','ofi','crisis_flag','last_funding_rate','last_macro_severity','macro_risk_factor','days_until_macro','days_since_macro']:
        df[col] = 0.0
    df['regime'] = 'sideways'
    
    print(f"Before dropna: {len(df)}")
    # Drop only rows where target or essential features are missing
    essential_cols = ['rsi_14', 'vwap_dist', 'target']
    df = df.dropna(subset=essential_cols)
    print(f"After dropna: {len(df)}")
    
    # Cast all potential numeric columns to float
    for col in df.columns:
        if col not in ['timestamp', 'symbol', 'regime', 'target']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
    # Drop non-essential columns that might cause parquet issues
    df = df.drop(columns=['ot', 'ct', 'qv', 'i', 'tbq'], errors='ignore')
    
    df.to_parquet(out_file)
    print(f"Successfully processed {len(df)} rows to {out_file}")

if __name__ == "__main__":
    process_and_label()
