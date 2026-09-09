import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import os

class WalkForwardBacktester:
    def __init__(self, data_path='data/processed/labeled_data.parquet'):
        self.data_path = data_path
        self.results = []
        
    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
        
    def run_backtest(self, train_days=60, test_days=7, prob_threshold=0.7):
        df = self.load_data()
        
        # Prepare features
        exclude_cols = ['timestamp', 'symbol', 'target', 'future_price', 'future_return', 'type', 'data_version', 'downloaded_at', 'source']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        df = df.dropna(subset=feature_cols + ['target'])
        
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        current_train_start = start_date
        
        all_trades = []
        
        while True:
            current_train_end = current_train_start + pd.Timedelta(days=train_days)
            current_test_end = current_train_end + pd.Timedelta(days=test_days)
            
            if current_test_end > end_date:
                # If we can't complete a full test window, break or use remaining
                if current_train_end >= end_date:
                    break
                current_test_end = end_date
                
            print(f"Rolling Window: Train [{current_train_start.date()} to {current_train_end.date()}] | Test [{current_train_end.date()} to {current_test_end.date()}]")
            
            train_mask = (df['timestamp'] >= current_train_start) & (df['timestamp'] < current_train_end)
            test_mask = (df['timestamp'] >= current_train_end) & (df['timestamp'] < current_test_end)
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if len(test_df) == 0:
                break
                
            if len(train_df) < 1000:
                print("Not enough training data in this window. Skipping.")
                current_train_start += pd.Timedelta(days=test_days)
                continue
                
            X_train, y_train = train_df[feature_cols], train_df['target']
            X_test, y_test = test_df[feature_cols], test_df['target']
            
            scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
            
            base_model = xgb.XGBClassifier(
                n_estimators=100, # reduced for speed during walk-forward
                max_depth=6,
                learning_rate=0.03,
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                random_state=42,
                n_jobs=-1
            )
            
            # For speed in walk forward, we fit directly or do simple calibration
            base_model.fit(X_train, y_train)
            calibrated = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv='prefit')
            # Fit calibrator on last 20% of train
            calib_size = int(len(X_train) * 0.2)
            calibrated.fit(X_train.iloc[-calib_size:], y_train.iloc[-calib_size:])
            
            # Predict
            probs = calibrated.predict_proba(X_test)[:, 1]
            test_df = test_df.copy()
            test_df['predicted_prob'] = probs
            
            # Extract Trades
            trades = test_df[test_df['predicted_prob'] >= prob_threshold].copy()
            all_trades.append(trades)
            
            current_train_start += pd.Timedelta(days=test_days)
            
        if not all_trades:
            print("No trades generated during backtest.")
            return None
            
        final_trades = pd.concat(all_trades, ignore_index=True)
        return self.analyze_results(final_trades)
        
    def analyze_results(self, trades_df):
        print("\n--- Walk-Forward Regime Analysis ---")
        if 'regime' not in trades_df.columns:
            trades_df['regime'] = 0 # Default
            
        results = []
        for regime, group in trades_df.groupby('regime'):
            wins = (group['future_return'] > 0).sum()
            losses = (group['future_return'] <= 0).sum()
            total = wins + losses
            win_rate = wins / total if total > 0 else 0
            
            # Approximate Sharpe (Return / StdDev)
            # Assuming returns include slippage
            returns = group['future_return'] - 0.001 # 0.1% slippage/fee assumption
            mean_ret = returns.mean()
            std_ret = returns.std()
            sharpe = (mean_ret / std_ret) * np.sqrt(288 * 365) if std_ret > 0 else 0 # 288 5m candles/day
            
            cum_ret = (1 + returns).cumprod()
            drawdown = (cum_ret.cummax() - cum_ret) / cum_ret.cummax()
            max_dd = drawdown.max() if not drawdown.empty else 0
            
            results.append({
                'Regime': regime,
                'Total Trades': total,
                'Win Rate': win_rate,
                'Sharpe': sharpe,
                'Max Drawdown': max_dd
            })
            
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))
        
        os.makedirs('data/processed', exist_ok=True)
        trades_df.to_parquet('data/processed/backtest_trades.parquet', engine='pyarrow')
        print("Saved chronological trades for Monte Carlo simulation.")
        return res_df

if __name__ == "__main__":
    bt = WalkForwardBacktester(data_path='data/processed/labeled_data.parquet')
    bt.run_backtest(train_days=60, test_days=7, prob_threshold=0.7)
