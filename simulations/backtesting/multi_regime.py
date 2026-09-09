import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def categorize_regimes(df):
    if 'atr_14' not in df.columns:
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['atr_14'] = calculate_atr(df)
        else:
            df['atr_14'] = 0.0
            
    # Moving Average Slope
    if 'close' in df.columns:
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_50_slope'] = (df['sma_50'] - df['sma_50'].shift(10)) / (df['sma_50'].shift(10) + 1e-8)
    else:
        df['sma_50_slope'] = 0.0
        
    atr_median = df['atr_14'].median()
    if pd.isna(atr_median):
        atr_median = 0
        
    conditions = [
        (df['sma_50_slope'] > 0.002) & (df['atr_14'] > atr_median),
        (df['sma_50_slope'] < -0.002) & (df['atr_14'] > atr_median * 1.5),
        (df['sma_50_slope'].between(-0.002, 0.002)) & (df['atr_14'] < atr_median)
    ]
    choices = ['Trending', 'Panic', 'Ranging']
    df['regime'] = np.select(conditions, choices, default='Mixed')
    
    # Calculate regime transitions
    df['prev_regime'] = df['regime'].shift(1)
    df['regime_transition'] = df['prev_regime'] + " -> " + df['regime']
    
    return df

def analyze_regimes(data_path='../data/processed/labeled_data.parquet', prob_threshold=0.6, output_dir='../experiments/multi_regime'):
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(data_path)
    df = categorize_regimes(df)
    
    model_path = '../models/saved_models/calibrated_xgb_model.joblib'
    if not os.path.exists(model_path):
        print("Model not found. Please train first.")
        return
        
    model = joblib.load(model_path)
    feature_cols = joblib.load('../models/saved_models/feature_cols.joblib')
    
    X = df[feature_cols].fillna(0)
    df['pred_prob'] = model.predict_proba(X)[:, 1]
    
    trades = df[df['pred_prob'] >= prob_threshold].copy()
    
    total_cost_rate = (0.001 + 0.0005) * 2 # fees + slippage round trip
    trades['simulated_return'] = trades['future_return'] - total_cost_rate
    
    # Base regime stats
    regime_stats = trades.groupby('regime').agg(
        total_trades=('target', 'count'),
        win_rate=('simulated_return', lambda x: (x > 0).mean()),
        avg_return=('simulated_return', 'mean'),
        sharpe=('simulated_return', lambda x: np.mean(x)/np.std(x)*np.sqrt(252*288) if np.std(x) > 0 else 0)
    ).reset_index()
    
    print("\n--- Multi-Regime Walk-Forward Analysis ---")
    print(regime_stats.to_string(index=False))
    
    regime_stats.to_csv(os.path.join(output_dir, 'regime_performance.csv'), index=False)
    
    # Transition stats (only where transition happened right before trade)
    transition_trades = trades[trades['regime'] != trades['prev_regime']]
    if not transition_trades.empty:
        trans_stats = transition_trades.groupby('regime_transition').agg(
            total_trades=('target', 'count'),
            win_rate=('simulated_return', lambda x: (x > 0).mean()),
            avg_return=('simulated_return', 'mean')
        ).reset_index()
        trans_stats = trans_stats[trans_stats['total_trades'] >= 5] # Only meaningful transitions
        
        print("\n--- Regime Transition Analysis ---")
        print(trans_stats.to_string(index=False))
        trans_stats.to_csv(os.path.join(output_dir, 'transition_performance.csv'), index=False)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(regime_stats['regime'], regime_stats['win_rate'] * 100)
    plt.axhline(50, color='red', linestyle='--', label='Break-Even Win Rate')
    plt.title('Win Rate by Market Regime (Post-Slippage)')
    plt.ylabel('Win Rate (%)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'regime_win_rates.png'))
    plt.close()
    
    print(f"Analysis saved to {output_dir}")

if __name__ == "__main__":
    analyze_regimes()
