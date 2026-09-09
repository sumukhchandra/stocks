import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score

import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_models.feature_registry import load_feature_cols, save_feature_cols

from ml_models.validation.purged_cv import PurgedWalkForwardCV

def create_meta_dataset(ensemble_dir, data_path, threshold=0.5):
    if not os.path.exists(ensemble_dir) or not os.path.exists(data_path):
        print(f"Paths not found: {ensemble_dir} or {data_path}")
        return None
        
    df = pd.read_parquet(data_path)
    
    # Use ensemble models to get average probability
    models = []
    for name in ['xgboost', 'lightgbm', 'catboost']:
        path = os.path.join(ensemble_dir, f'{name}_calibrated.joblib')
        if os.path.exists(path):
            models.append(joblib.load(path))
    
    if not models:
        print("No models found in ensemble dir.")
        return None
        
    feature_cols = joblib.load(os.path.join(ensemble_dir, 'feature_cols.joblib'))
    X = df[feature_cols]
    
    # Get average probability
    probas = []
    for model in models:
        probas.append(model.predict_proba(X)[:, 1])
    
    df['pred_prob'] = np.mean(probas, axis=0)
    
    # Meta label: 1 if trade would have been successful, 0 otherwise
    # We define success as simulated_return > 0 (including costs)
    costs = 0.0006 # Adjusted to typical short-term trading costs
    df['meta_target'] = (df['future_return_t1'] > costs).astype(int)
    
    return df

def train_meta_model(df, model_dir='models/saved_models'):
    # Features for meta-model: Primary features + the primary prediction probability
    ensemble_dir = os.path.join(model_dir, 'ensemble')
    primary_feature_cols = joblib.load(os.path.join(ensemble_dir, 'feature_cols.joblib'))
    meta_feature_cols = primary_feature_cols + ['pred_prob']
    
    print(f"Meta-features used for training: {meta_feature_cols}")
    save_feature_cols(meta_feature_cols, 'meta')
    
    X = df[meta_feature_cols]
    y = df['meta_target']
    
    cv = PurgedWalkForwardCV(n_splits=5, purge_horizon=10)
    
    print("Training Meta Model...")
    meta_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.01,
        random_state=42
    )
    
    # Walk-forward for meta-model
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Balance classes for meta-training if needed
        scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
        meta_model.set_params(scale_pos_weight=scale_pos_weight)
        
        meta_model.fit(X_train, y_train)
        preds = meta_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Meta-Fold {fold+1} | Accuracy: {acc:.4f}")

    # Final fit
    meta_model.fit(X, y)
    joblib.dump(meta_model, os.path.join(model_dir, 'meta_xgb_model.joblib'))
    print("Meta model saved.")

if __name__ == "__main__":
    df_meta = create_meta_dataset('ml_models/saved_models/ensemble', 'QA/data/processed/master_labeled_dataset.parquet')
    if df_meta is not None:
        train_meta_model(df_meta, model_dir='ml_models/saved_models')
