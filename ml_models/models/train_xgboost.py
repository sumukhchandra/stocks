import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import matplotlib.pyplot as plt
import joblib
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_models.validation.purged_cv import PurgedWalkForwardCV
from ml_models.feature_registry import save_feature_cols

def train_xgboost(data_path='data/processed/cusum_sampled.parquet', model_dir='models/saved_models'):
    if not os.path.exists(data_path):
        # Fallback to labeled_data if cusum_sampled is not found
        fallback_path = 'data/processed/labeled_data.parquet'
        if os.path.exists(fallback_path):
            print(f"Sampled data not found, using {fallback_path}")
            data_path = fallback_path
        else:
            print(f"Data file not found: {data_path}")
            return
        
    df = pd.read_parquet(data_path)
    print(f"Training on {len(df)} samples from {data_path}")
    
    # Sort by time to prevent leakage
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Select features
    exclude_cols = [
        'timestamp', 'symbol', 'target', 'future_price', 'future_return', 
        'future_return_t1', 'triple_barrier_label', 'type', 
        'data_version', 'downloaded_at', 'source', 'ignore',
        'index', 'open_time', 'close_time' # Exclude non-numeric/index columns
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Features used for training: {feature_cols}")
    
    # Save to feature registry
    save_feature_cols(feature_cols, 'primary')
    
    # Drop rows with NaNs in features
    df = df.dropna(subset=feature_cols)
    
    if len(df) < 100:
        print("Not enough data to train.")
        return
        
    X = df[feature_cols]
    y = df['target']
    
    # Calculate scale_pos_weight for the ENTIRE dataset to handle imbalance
    scale_pos_weight_all = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1.0
    print(f"Dataset Imbalance Ratio (Neg/Pos): {scale_pos_weight_all:.2f}")

    # Walk-forward validation using Purged Walk-Forward
    # purge_horizon=5 matches the t1=5 in triple_barrier labeling
    tscv = PurgedWalkForwardCV(n_splits=5, pct_embargo=0.02, purge_horizon=5)
    
    print("Starting Walk-Forward Validation...")
    
    models = []
    scores = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # XGBoost configuration for high-frequency market data
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,  # Increased depth for complex microstructure patterns
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight_all, # Use global weight
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=50,
            random_state=42
        )
        
        eval_set = [(X_train, y_train), (X_test, y_test)]
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Evaluate
        preds = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, pred_proba)
        brier = brier_score_loss(y_test, pred_proba)
        
        print(f"Fold {fold+1} | Accuracy: {acc:.4f} | Log Loss: {loss:.4f} | Brier Score: {brier:.4f} | Best Iteration: {model.best_iteration}")
        
        models.append(model)
        scores.append(acc)
    
    print(f"Average Accuracy: {np.mean(scores):.4f}")
    
    # Calculate scale_pos_weight for final model
    scale_pos_weight_final = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1.0

    # Training final base model with global imbalance weighting
    print("Training final base model...")
    base_model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_all,
        objective='binary:logistic',
        random_state=42
    )
    
    # We'll use a standard CV for calibration if the custom one is causing issues 
    # with scikit-learn's internal validation, or ensure it's compatible.
    # For now, let's use 5-fold TimeSeriesSplit for calibration to ensure compatibility.
    print("Training and Fitting Isotonic Calibration (using TimeSeriesSplit)...")
    cal_cv = TimeSeriesSplit(n_splits=5)
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=cal_cv)
    calibrated_model.fit(X, y)
    
    # Generate Calibration Curve
    print("Generating Calibration Curve...")
    os.makedirs(model_dir, exist_ok=True)
    prob_pos = calibrated_model.predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Isotonic Calibrated XGBoost")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'calibration_curve.png'))
    plt.close()
    
    # Extract and save feature importances
    # Since calibrated_model contains an ensemble of models, we average their importances
    importances = np.mean([clf.estimator.feature_importances_ for clf in calibrated_model.calibrated_classifiers_], axis=0)
    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    fi_df.to_csv(os.path.join(model_dir, 'feature_importances.csv'), index=False)
    print("Feature importances saved.")
    
    # Save the calibrated model
    model_path = os.path.join(model_dir, 'calibrated_xgb_model.joblib')
    joblib.dump(calibrated_model, model_path)
    
    # Also save feature names for later verification
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.joblib'))
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_xgboost()
