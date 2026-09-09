import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib

class MicrostructureForecaster:
    """
    Research model for sub-minute price movement forecasting.
    """
    def __init__(self, horizon='5s'):
        self.horizon = horizon
        self.model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        self.feature_cols = [
            'imbalance', 'spread', 'microprice', 'book_slope', 
            'spread_velocity', 'slope_curvature', 'is_bid_wall', 
            'is_ask_wall', 'imbalance_momentum'
        ]

    def train(self, df):
        target_col = f'target_{self.horizon}'
        if target_col not in df.columns:
            print(f"Target {target_col} not found in dataframe.")
            return
            
        X = df[self.feature_cols].fillna(0)
        y = df[target_col]
        
        # Simple split (Train 80%, Test 20%)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training Forecaster for {self.horizon} horizon...")
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        
        print(f"Results for {self.horizon}: Accuracy {acc:.4f}, Precision {prec:.4f}")
        
    def save(self, path):
        joblib.dump(self.model, path)

if __name__ == "__main__":
    data_path = 'data/processed/orderbook_depth_dataset_labeled.parquet'
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        # Add book_slope etc if missing
        df['book_slope'] = df['spread'] / (np.abs(df['imbalance']) + 1e-6)
        df['spread_velocity'] = df['spread'].diff().fillna(0)
        df['slope_curvature'] = df['book_slope'].diff().fillna(0)
        df['is_bid_wall'] = 0 # placeholders
        df['is_ask_wall'] = 0
        df['imbalance_momentum'] = df['imbalance'].diff(5).fillna(0)
        df['microprice'] = (df['best_ask'] * df['best_bid_qty'] + df['best_bid'] * df['best_ask_qty']) / (df['best_bid_qty'] + df['best_ask_qty'] + 1e-9)

        forecaster = MicrostructureForecaster(horizon='5s')
        forecaster.train(df)
    else:
        print("Labeled research dataset not found. Run micro_horizon_labels.py first.")
