import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtesting.full_system_backtest import FullSystemBacktest

def run_stress_test():
    data_path = 'data/processed/master_labeled_dataset.parquet'
    if not os.path.exists(data_path):
        print("Master dataset not found.")
        return
        
    df = pd.read_parquet(data_path)
    
    # Filter for crisis regimes
    crisis_regimes = ['flash_crash', 'high_volatility']
    stress_df = df[df['regime'].isin(crisis_regimes)].copy()
    
    if stress_df.empty:
        print("No crisis regimes found in dataset.")
        return
        
    print(f"Running Stress Test on {len(stress_df)} crisis bars...")
    
    # We use the FullSystemBacktest but pass the stress_df
    # We need to save it to a temp file because FullSystemBacktest loads from disk
    temp_path = 'data/processed/temp_stress_test.parquet'
    stress_df.to_parquet(temp_path)
    
    bt = FullSystemBacktest(data_path=temp_path)
    bt.run()
    bt.print_results()
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

if __name__ == "__main__":
    run_stress_test()
