import pandas as pd
import numpy as np
import joblib
import os
import sys
from xgboost import XGBClassifier

# Add project root to path for imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from ml_models.validation.purged_cv import PurgedWalkForwardCV


def train_regime_specialists(
    data_path=os.path.join(PARENT_DIR, 'data', 'processed', 'master_labeled_dataset.parquet'),
    model_dir=os.path.join(PARENT_DIR, 'ml_models', 'models', 'saved_models', 'regime_specialists'),
):
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define regime mappings
    regime_map = {
        'strong_bull': 'trending_bull',
        'strong_bear': 'trending_bear',
        'sideways_chop': 'sideways',
        'flash_crash': 'high_volatility',
        'crisis_regime': 'high_volatility',
        'high_volatility': 'high_volatility',
        'sideways': 'sideways',
        'overheated_bull': 'trending_bull',
        'trending_bull': 'trending_bull',
        'trending_bear': 'trending_bear',
    }
    
    df['regime_group'] = df['regime'].map(regime_map).fillna(df['regime'])
    
    # Align features with ensemble primary features
    ens_feat_path = os.path.join(PARENT_DIR, 'ml_models', 'models', 'saved_models', 'ensemble', 'feature_cols.joblib')
    if os.path.exists(ens_feat_path):
        primary_features = joblib.load(ens_feat_path)
        feature_cols = [c for c in primary_features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        exclude = [
            'timestamp', 'symbol', 'target', 'future_price', 'future_return', 'future_return_t1',
            'triple_barrier_label', 'regime', 'regime_group', 'is_synthetic', 'ignore', 'close_time',
            'open_time', 'future_max_high', 'future_return_3%', 'target_ret', 'actual_return'
        ]
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        feature_cols = [c for c in feature_cols if df[c].std() > 0]
    
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.joblib'))
    
    regimes = [r for r in df['regime_group'].unique() if pd.notna(r)]
    print(f"Training specialists for regimes: {regimes}")
    print(f"Using {len(feature_cols)} features for specialist models.")
    
    for regime in regimes:
        regime_df = df[df['regime_group'] == regime].copy()
        if len(regime_df) < 500:
            print(f"Skipping {regime} due to insufficient samples ({len(regime_df)})")
            continue
            
        print(f"Training specialist for: {regime} ({len(regime_df)} samples)...")
        X = regime_df[feature_cols].fillna(0)
        y = regime_df['target'].astype(int)
        
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        scale_weight = float(neg_count / (pos_count + 1e-8))
        
        # Regime-specialist classifier tuned for high precision
        model = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=scale_weight,
            eval_metric="logloss",
            n_jobs=-1,
        )
        
        # Purged walk-forward validation fold fitting
        cv = PurgedWalkForwardCV(n_splits=3, purge_horizon=10)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            
        # Final fit on entire regime partition
        model.fit(X, y)
        out_path = os.path.join(model_dir, f'specialist_{regime}.joblib')
        joblib.dump(model, out_path)
        print(f"Specialist for {regime} saved to {out_path}")

    # Copy sideways backup specialist if available
    backup_sideways = os.path.join(PARENT_DIR, 'ml_models', 'models', 'saved_models', 'backup', 'specialist_sideways.joblib')
    target_sideways = os.path.join(model_dir, 'specialist_sideways.joblib')
    if os.path.exists(backup_sideways) and not os.path.exists(target_sideways):
        import shutil
        shutil.copyfile(backup_sideways, target_sideways)
        print(f"Copied backup sideways specialist to {target_sideways}")


if __name__ == "__main__":
    train_regime_specialists()

