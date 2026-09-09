import pandas as pd
import numpy as np
import os

def apply_micro_horizon_labels(df, horizons=None):
    """
    Applies binary labels (Up/Down) for multiple short horizons.
    horizons: dict of {name: periods}
    """
    if horizons is None:
        horizons = {
            '1s': 10,
            '5s': 50,
            '15s': 150,
            '60s': 600
        }
        
    df = df.copy()
    if 'midprice' not in df.columns:
        df['midprice'] = (df['best_bid'] + df['best_ask']) / 2.0
        
    for name, periods in horizons.items():
        # Label is 1 if future price > current price, else 0
        df[f'target_{name}'] = (df['midprice'].shift(-periods) > df['midprice']).astype(int)
        
    # Drop the last max(periods) to remove NaNs
    max_p = max(horizons.values())
    return df.iloc[:-max_p]

if __name__ == "__main__":
    data_path = 'data/processed/orderbook_depth_dataset.parquet'
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        labeled_df = apply_micro_horizon_labels(df)
        labeled_df.to_parquet(data_path.replace('.parquet', '_labeled.parquet'))
        print(f"Multi-horizon labels applied to {len(labeled_df)} rows.")
        for col in [c for c in labeled_df.columns if 'target_' in c]:
            print(f"{col} distribution:\n{labeled_df[col].value_counts(normalize=True)}")
