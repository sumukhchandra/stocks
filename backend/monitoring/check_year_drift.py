import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from monitoring.feature_drift import FeatureDriftDetector

def check_year_drift(data_path='data/processed/labeled_data.parquet'):
    if not os.path.exists(data_path):
        print("Data not found.")
        return
        
    df = pd.read_parquet(data_path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    ref = df[df['timestamp'].dt.year == 2023]
    curr = df[df['timestamp'].dt.year == 2024]
    
    print(f"Checking drift between 2023 (n={len(ref)}) and 2024 (n={len(curr)})...")
    
    detector = FeatureDriftDetector(ref, curr)
    features = ['close', 'volume', 'vpin', 'mean_imbalance', 'mean_spread', 'volume_delta', 'ofi']
    # Filter only existing features
    features = [f for f in features if f in df.columns]
    
    drift_report = detector.run_drift_analysis(features)
    print("\n--- Feature Drift Report (2023 vs 2024) ---")
    print(drift_report)

if __name__ == "__main__":
    check_year_drift()
