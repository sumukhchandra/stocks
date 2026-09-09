import pandas as pd
import numpy as np

def apply_triple_barrier_labels(df, pt_sl=[1, 1], t1=5, min_ret=0.0005):
    """
    Institutional Triple Barrier Labeling.
    df: DataFrame with OHLC data.
    pt_sl: list of 2 multipliers for take-profit and stop-loss barriers.
    t1: number of candles for the vertical barrier (time).
    min_ret: minimum return required for the barriers to be active.
    """
    if df.empty:
        return df
        
    df = df.copy()
    # Sort to ensure temporal order
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate volatility (rolling standard deviation of returns)
    # Used to set dynamic barriers if desired, but here we'll use fixed for simplicity 
    # as a first step or use the provided pt_sl as percentage multipliers.
    # For HFT, 0.001 (10bps) or 0.002 (20bps) is common.
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    labels = np.zeros(len(df))
    future_returns = np.zeros(len(df))
    max_future_returns = np.zeros(len(df))
    
    # Simple implementation: iterate (optimized for performance if needed later)
    # Barrier width (example: 0.15% = 0.0015)
    width = min_ret
    
    for i in range(len(df) - t1):
        price_start = close[i]
        upper_barrier = price_start * (1 + width * pt_sl[0])
        lower_barrier = price_start * (1 - width * pt_sl[1])
        
        # Check future windows
        label = 0 # Default: vertical barrier (time) hit
        for j in range(1, t1 + 1):
            if high[i+j] >= upper_barrier:
                label = 1
                break
            elif low[i+j] <= lower_barrier:
                label = -1
                break
        
        labels[i] = label
        # Future return until first barrier hit or end of window
        # For simplicity, future return at t1
        future_returns[i] = (close[i+t1] - price_start) / price_start

        # -- Upgrade 6: Max favorable return in the window ------------
        # The maximum return achievable within t+1..t+t1 using highs
        max_high_in_window = max(high[i+1 : i+t1+1])
        max_future_returns[i] = (max_high_in_window - price_start) / price_start

    df['triple_barrier_label'] = labels
    df['future_return_t1'] = future_returns
    df['max_future_return'] = max_future_returns
    
    # We want to predict if label == 1 (Up)
    df['target'] = (df['triple_barrier_label'] == 1).astype(int)
    
    # Use max_future_return as the regressor target (more realistic for trading)
    df['target_ret'] = df['max_future_return']
    
    # Drop the last t1 rows as they don't have full future lookahead
    return df.iloc[:-t1]

if __name__ == "__main__":
    import os
    input_file = 'data/processed/features_latest.parquet'
    output_file = 'data/processed/labeled_data.parquet'
    
    if os.path.exists(input_file):
        df = pd.read_parquet(input_file)
        # Apply triple barrier: 1.5x take profit, 1.0x stop loss, 5 period horizon
        df_labeled = apply_triple_barrier_labels(df, pt_sl=[1.5, 1.0], t1=5, min_ret=0.001)
        df_labeled.to_parquet(output_file, engine='pyarrow')
        print(f"Triple Barrier Labeling complete. Class distribution:")
        print(df_labeled['target'].value_counts(normalize=True))
        print(f"Saved to {output_file}")
    else:
        print(f"Input file not found: {input_file}")
