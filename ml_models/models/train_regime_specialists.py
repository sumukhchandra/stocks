import pandas as pd
import numpy as np
import joblib
import os
import sys
from xgboost import XGBClassifier

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_models.validation.purged_cv import PurgedWalkForwardCV

def train_regime_specialists(data_path='QA/data/processed/labeled_data.parquet', model_dir='ml_models/saved_models/regime_specialists'):
    if not os.path.exists(data_path):
        print("Data not found.")
        return
        
    df = pd.read_parquet(data_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define regime mappings to user requests
    regime_map = {
        'strong_bull': 'trending_bull',
        'strong_bear': 'trending_bear',
        'sideways_chop': 'sideways',
        'flash_crash': 'crisis_regime',
        'high_volatility': 'high_volatility',
        'sideways': 'sideways',
        'overheated_bull': 'overheated_bull'
    }
    
    df['regime_group'] = df['regime'].map(regime_map).fillna(df['regime'])
    
    exclude = ['timestamp', 'symbol', 'target', 'future_price', 'future_return', 'future_return_t1', 'triple_barrier_label', 'regime', 'regime_group', 'is_synthetic', 'ignore', 'close_time', 'open_time', 'future_max_high', 'future_return_3%', 'target_ret', 'actual_return']
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # --- NOISE CLEANING ---
    # Remove constant features
    feature_cols = [c for c in feature_cols if df[c].std() > 0]
    
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.joblib'))
    
    regimes = df['regime_group'].unique()
    print(f"Training specialists for regimes: {regimes}")
    
    for regime in regimes:
        regime_df = df[df['regime_group'] == regime]
        if len(regime_df) < 500:
            print(f"Skipping {regime} due to insufficient samples ({len(regime_df)})")
            continue
            
        print(f"Training specialist for: {regime} ({len(regime_df)} samples)...")
        X = regime_df[feature_cols]
        y = regime_df['target']
        
        # Specialist model: more aggressive since it's regime-focused
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.8,
            random_state=42,
            scale_pos_weight=(len(y) - y.sum()) / (y.sum() + 1e-8)
        )
        
        # Simple walk-forward for specialist validation
        cv = PurgedWalkForwardCV(n_splits=3, purge_horizon=10)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            
        # Final fit
        model.fit(X, y)
        joblib.dump(model, os.path.join(model_dir, f'specialist_{regime}.joblib'))
        print(f"Specialist for {regime} saved.")

if __name__ == "__main__":
    train_regime_specialists()
