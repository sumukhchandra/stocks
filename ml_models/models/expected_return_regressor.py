import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_models.validation.purged_cv import PurgedWalkForwardCV

class ExpectedReturnRegressor:
    """
    High-Capacity Ensemble Return Regressor:
    Combines LightGBM DART + CatBoost Huber + XGBoost Pseudo-Huber + ExtraTrees
    to accurately predict real financial return magnitude without artificial compression.
    """
    def __init__(self, model_dir='ml_models/saved_models'):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.models = {
            'lgbm': LGBMRegressor(
                boosting_type='dart',
                n_estimators=800,
                max_depth=7,
                learning_rate=0.04,
                num_leaves=63,
                objective='huber',
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1
            ),
            'catboost': CatBoostRegressor(
                n_estimators=800,
                depth=6,
                learning_rate=0.04,
                loss_function='Huber:delta=0.002',
                subsample=0.85,
                random_state=42,
                verbose=0
            ),
            'xgb': XGBRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.04,
                objective='reg:pseudohubererror',
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42
            ),
            'extratrees': ExtraTreesRegressor(
                n_estimators=400,
                max_depth=15,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1
            ),
        }

    def train(self, data_path='QA/data/processed/labeled_data.parquet', df=None):
        if df is None:
            if not os.path.exists(data_path):
                print("Training data not found.")
                return
            df = pd.read_parquet(data_path)

        # Load feature columns from registry if present
        feature_cols_path = os.path.join(self.model_dir, 'ensemble', 'feature_cols.joblib')
        if os.path.exists(feature_cols_path):
            feature_cols = joblib.load(feature_cols_path)
        else:
            exclude = [
                'timestamp', 'symbol', 'target', 'future_price', 'future_return',
                'future_return_t1', 'triple_barrier_label', 'is_synthetic',
                'future_max_high', 'future_return_3%', 'target_ret', 'actual_return',
                'max_future_return', 'regime', 'company_name', 'date_only',
                'close_time', 'open_time', 'downloaded_at'
            ]
            feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        # Clean target: remove NaNs and Infs
        df = df[np.isfinite(df['target_ret'])].copy()

        X = df.reindex(columns=feature_cols, fill_value=0)
        y = df['target_ret'].values

        print(f"Training High-Capacity Return Regressors (DART + CatBoost + XGB + ET) on {len(X)} samples...")
        print(f"  Target return range: min={y.min()*100:.3f}%, mean={y.mean()*100:.3f}%, max={y.max()*100:.3f}%")

        # Cross-validated evaluation with purged walk-forward
        cv = PurgedWalkForwardCV(n_splits=5, purge_horizon=10)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_preds = []
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                fold_preds.append(model.predict(X_test))

            # Weighted blend: DART (35%) + CatBoost (30%) + XGB (25%) + ExtraTrees (10%)
            ensemble_pred = (fold_preds[0] * 0.35) + (fold_preds[1] * 0.30) + (fold_preds[2] * 0.25) + (fold_preds[3] * 0.10)
            mae = mean_absolute_error(y_test, ensemble_pred)
            print(f"  Fold {fold+1} | MAE: {mae*100:.4f}%")

        # Final fit on entire dataset
        print("  Final fit on all training data...")
        for name, model in self.models.items():
            model.fit(X, y)
            model_path = os.path.join(self.model_dir, f'return_regressor_{name}.joblib')
            joblib.dump(model, model_path)
            print(f"    Saved {name} -> {model_path}")

        # Backward compatibility
        joblib.dump(self.models['xgb'], os.path.join(self.model_dir, 'expected_return_xgb.joblib'))
        print(f"Return Regressors saved to {self.model_dir}")

    def predict(self, X):
        """Weighted ensemble prediction across 4 models."""
        p_lgb = self.models['lgbm'].predict(X)
        p_cat = self.models['catboost'].predict(X)
        p_xgb = self.models['xgb'].predict(X)
        p_et = self.models['extratrees'].predict(X)
        return (p_lgb * 0.35) + (p_cat * 0.30) + (p_xgb * 0.25) + (p_et * 0.10)

    def load(self):
        """Load pre-trained models from disk."""
        for name in list(self.models.keys()):
            path = os.path.join(self.model_dir, f'return_regressor_{name}.joblib')
            if os.path.exists(path):
                self.models[name] = joblib.load(path)


if __name__ == "__main__":
    regressor = ExpectedReturnRegressor()
    regressor.train()
