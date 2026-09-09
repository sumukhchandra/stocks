import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score, brier_score_loss

def run_oos_test(data_path='data/processed/labeled_data.parquet', model_dir='models/saved_models/ensemble'):
    if not os.path.exists(data_path):
        print("Data not found.")
        return
        
    df = pd.read_parquet(data_path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Split by year
    train_df = df[df['timestamp'].dt.year == 2023].copy()
    test_df = df[df['timestamp'].dt.year == 2024].copy()
    
    if train_df.empty or test_df.empty:
        print(f"Insufficient data for OOS test. Train: {len(train_df)}, Test: {len(test_df)}")
        return

    print(f"OOS Test: Training on 2023 ({len(train_df)}), Testing on 2024 ({len(test_df)})")
    
    feature_cols_path = os.path.join(model_dir, 'feature_cols.joblib')
    if not os.path.exists(feature_cols_path):
        print("Feature columns not found.")
        return
    feature_cols = joblib.load(feature_cols_path)
    
    X_train, y_train = train_df[feature_cols], train_df['target']
    X_test, y_test = test_df[feature_cols], test_df['target']
    
    # Load ensemble models
    models = ['xgboost', 'lightgbm', 'catboost']
    results = {}
    
    for name in models:
        model_path = os.path.join(model_dir, f'{name}_calibrated.joblib')
        if not os.path.exists(model_path):
            continue
            
        model = joblib.load(model_path)
        # Re-fit on 2023 specifically for this test
        # Note: In practice, we'd use the pre-trained model if it was trained ONLY on 2023.
        # But here we'll re-fit to be sure.
        print(f"Re-fitting {name} on 2023 data...")
        # Since calibrated_model is already calibrated, we might just want to fit the underlying estimator
        # but let's just train a fresh one for the OOS test.
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        
        if name == 'xgboost':
            clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.01)
        elif name == 'lightgbm':
            clf = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.01, verbose=-1)
        else:
            clf = CatBoostClassifier(n_estimators=500, depth=6, learning_rate=0.01, verbose=0)
            
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        brier = brier_score_loss(y_test, probas)
        
        results[name] = {
            'accuracy': acc,
            'brier': brier,
            'report': classification_report(y_test, preds, output_dict=True)
        }
        
        print(f"Model: {name} | OOS Accuracy: {acc:.4f} | Brier: {brier:.4f}")

    # Weighted Ensemble
    ensemble_probas = np.mean([joblib.load(os.path.join(model_dir, f'{n}_calibrated.joblib')).predict_proba(X_test)[:, 1] for n in models], axis=0)
    ensemble_preds = (ensemble_probas > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ensemble_preds)
    print(f"Ensemble | OOS Accuracy: {ens_acc:.4f}")

if __name__ == "__main__":
    run_oos_test()
