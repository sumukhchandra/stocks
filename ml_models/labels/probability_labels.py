import pandas as pd
import numpy as np

def generate_labels(df, price_col='close', threshold=0.0015, horizon=1):
    """
    Generates probability labels for classification.
    Target: 1 if price moves up by > `threshold` (e.g. 0.3%) in `horizon` steps (e.g. 5 min if 5m timeframe).
    Otherwise 0.
    """
    if df is None or df.empty or price_col not in df.columns:
        return df
        
    df = df.copy()
    
    # Calculate future return
    df['future_price'] = df.groupby('symbol')[price_col].shift(-horizon)
    df['future_return'] = (df['future_price'] - df[price_col]) / df[price_col]
    
    # Create binary label
    df['target'] = (df['future_return'] > threshold).astype(int)
    
    # Drop rows where future price is NaN (the last `horizon` rows per symbol)
    df = df.dropna(subset=['future_price'])
    
    return df

if __name__ == "__main__":
    # Example usage
    import os
    
    input_file = 'data/processed/features_latest.parquet'
    output_file = 'data/processed/labeled_data.parquet'
    
    if os.path.exists(input_file):
        df = pd.read_parquet(input_file)
        df_labeled = generate_labels(df)
        df_labeled.to_parquet(output_file, engine='pyarrow')
        print(f"Generated labels and saved to {output_file}")
    else:
        print(f"Input file not found: {input_file}")
