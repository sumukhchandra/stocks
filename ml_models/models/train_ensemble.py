import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_models.validation.purged_cv import PurgedWalkForwardCV

class ModelEnsemble:
    def __init__(self, model_dir='models/saved_models/ensemble'):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}

    def train_all(self, X, y, feature_cols):
        print(f"Saving models to: {os.path.abspath(self.model_dir)}")
        # 1. XGBoost
        print("Training XGBoost...")
        xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.01, subsample=0.8, random_state=42)
        self.models['xgboost'] = self._calibrate_and_fit(xgb, X, y)
        
        # 2. LightGBM
        print("Training LightGBM...")
        lgbm = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.01, subsample=0.8, random_state=42, verbose=-1)
        self.models['lightgbm'] = self._calibrate_and_fit(lgbm, X, y)
        
        # 3. CatBoost
        print("Training CatBoost...")
        cat = CatBoostClassifier(n_estimators=500, depth=6, learning_rate=0.01, subsample=0.8, random_state=42, verbose=0)
        self.models['catboost'] = self._calibrate_and_fit(cat, X, y)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_dir, f'{name}_calibrated.joblib'))
        
        joblib.dump(feature_cols, os.path.join(self.model_dir, 'feature_cols.joblib'))
        print(f"Ensemble training complete. Models saved to {self.model_dir}")

    def _calibrate_and_fit(self, estimator, X, y):
        # Use Purged Walk-Forward for calibration and validation
        cal_cv = PurgedWalkForwardCV(n_splits=5, purge_horizon=10)
        calibrated = CalibratedClassifierCV(estimator=estimator, method='isotonic', cv=cal_cv)
        calibrated.fit(X, y)
        return calibrated

    def predict_weighted(self, X):
        """Simple average of probabilities."""
        probas = []
        for name, model in self.models.items():
            probas.append(model.predict_proba(X)[:, 1])
        
        return np.mean(probas, axis=0)

if __name__ == "__main__":
    data_path = 'QA/data/processed/master_labeled_dataset.parquet'
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        exclude = [
            'timestamp', 'symbol', 'target', 'future_price', 'future_return', 
            'future_return_t1', 'triple_barrier_label', 'is_synthetic',
            'ignore', 'close_time', 'open_time', 'future_max_high', 'future_return_3%',
            'actual_return', 'target_ret', 'regime', 'ot', 'ct', 'qv', 'nt', 'tbb', 'tbq', 'i', 'downloaded_at'
        ]
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        
        # --- NOISE CLEANING ---
        # 1. Remove constant features
        feature_cols = [c for c in feature_cols if df[c].std() > 0]
        
        # Define X and y
        # We need to drop NaNs in target
        df = df.dropna(subset=['target'])
        X = df[feature_cols]
        y = df['target']
        
        print(f"Dataset Size: {len(df)} | Features: {len(feature_cols)}")
        print(f"Target distribution: \n{y.value_counts(normalize=True)}")
        
        # --- OUT-OF-SAMPLE SPLIT ---
        # Only train on first 80% to allow honest 20% testing
        split_idx = int(len(X) * 0.8)
        X_train_oos = X.iloc[:split_idx]
        y_train_oos = y.iloc[:split_idx]
        
        print(f"Training on {len(X_train_oos)} samples (Out-of-sample split: 80/20)")
        
        ensemble = ModelEnsemble(model_dir='ml_models/saved_models/ensemble')
        ensemble.train_all(X_train_oos, y_train_oos, feature_cols)
    else:
        print(f"Data for training not found at {data_path}")
