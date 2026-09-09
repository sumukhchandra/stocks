import pandas as pd
import numpy as np
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.macro_features import MacroFeatureEngineer
from labels.triple_barrier import apply_triple_barrier_labels
from technical import calculate_rsi, calculate_atr, calculate_vwap

def build_final_master():
    real_path = 'data/processed/master_labeled_dataset.parquet'
    synth_path = 'data/processed/synthetic_monte_carlo.parquet'
    output_path = 'data/processed/master_labeled_dataset.parquet'

    if not os.path.exists(real_path):
        print("Real master dataset not found. Run MasterFeaturePipeline first.")
        return

    print("Loading real dataset...")
    real_df = pd.read_parquet(real_path)
    real_df['is_synthetic'] = 0
    real_df['crisis_flag'] = (real_df['regime'] == 'high_volatility').astype(int)
    
    # Ensure OFI exists (as a proxy or placeholder if data is missing)
    if 'ofi' not in real_df.columns:
        print("Adding OFI proxy (derived from volume delta)...")
        # OFI is normally from orderbook, but we use volume delta as a proxy
        real_df['ofi'] = real_df['volume_delta'].rolling(5).mean().fillna(0)

    if os.path.exists(synth_path):
        print("Loading synthetic dataset...")
        synth_df = pd.read_parquet(synth_path)
        
        # Calculate missing features for synthetic data to match real_df schema
        print("Calculating missing features for synthetic data...")
        synth_df['rsi_14'] = calculate_rsi(synth_df, period=14, price_col='close')
        synth_df['atr_14'] = calculate_atr(synth_df, period=14, close_col='close')
        synth_df['vwap'] = calculate_vwap(synth_df, volume_col='volume', price_col='close')
        synth_df['vwap_dist'] = (synth_df['close'] - synth_df['vwap']) / (synth_df['vwap'] + 1e-8)
        
        # Macro features for synthetic (placeholders since it's synthetic)
        synth_df['days_since_macro'] = 365
        synth_df['days_until_macro'] = 365
        synth_df['last_macro_severity'] = 0
        synth_df['macro_risk_factor'] = 0
        
        # Funding rate for synthetic
        synth_df['last_funding_rate'] = 0.0001 # Neutral
        
        # Microstructure
        synth_df['volume_delta'] = synth_df['mean_imbalance'] * synth_df['volume']
        synth_df['ofi'] = synth_df['volume_delta'] # Simplified
        synth_df['imbalance'] = synth_df['mean_imbalance']
        synth_df['spread'] = synth_df['mean_spread']
        
        # Labeling synthetic data
        print("Labeling synthetic data...")
        synth_df = apply_triple_barrier_labels(synth_df, pt_sl=[1.5, 1.0], t1=5, min_ret=0.001)
        
        synth_df['is_synthetic'] = 1
        synth_df['crisis_flag'] = (synth_df['regime'].isin(['flash_crash', 'high_volatility'])).astype(int)
        
        # Align columns
        common_cols = list(set(real_df.columns) & set(synth_df.columns))
        print(f"Concatenating real and synthetic data on {len(common_cols)} common columns...")
        
        final_df = pd.concat([real_df[common_cols], synth_df[common_cols]], ignore_index=True)
    else:
        print("Synthetic dataset not found. Using real data only.")
        final_df = real_df

    # Final sort
    if 'timestamp' in final_df.columns:
        final_df = final_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    final_df.to_parquet(output_path, engine='pyarrow')
    print(f"Final Master Dataset built: {len(final_df)} rows.")
    print(f"Regime distribution:\n{final_df['regime'].value_counts()}")
    print(f"Crisis flags: {final_df['crisis_flag'].sum()}")

if __name__ == "__main__":
    build_final_master()
