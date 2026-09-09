import pandas as pd
import numpy as np
import xgboost as xgb
from metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_expectancy
import os
import joblib
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_models.feature_registry import load_feature_cols

def calculate_ulcer_index(cumulative_return):
    drawdown = (cumulative_return.cummax() - cumulative_return) / cumulative_return.cummax()
    ulcer_index = np.sqrt(np.mean(drawdown**2))
    return ulcer_index

def calculate_recovery_factor(cumulative_return):
    total_return = cumulative_return.iloc[-1] - 1
    max_dd = calculate_max_drawdown(cumulative_return)
    if max_dd == 0:
        return np.inf
    return total_return / max_dd

class Backtester:
    def __init__(self, model_path='../models/saved_models/calibrated_xgb_model.joblib', 
                 meta_model_path='../models/saved_models/meta_xgb_model.joblib',
                 data_path='../data/processed/labeled_data.parquet', fee_rate=0.001, slippage_base=0.0002, latency_ms=300):
        self.model_path = model_path
        self.meta_model_path = meta_model_path
        self.data_path = data_path
        self.fee_rate = fee_rate
        self.slippage_base = slippage_base
        self.latency_ms = latency_ms
        
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None
            print(f"Warning: Primary Model not found at {self.model_path}")
            
        if os.path.exists(self.meta_model_path):
            self.meta_model = joblib.load(self.meta_model_path)
        else:
            self.meta_model = None
            print(f"Warning: Meta Model not found at {self.meta_model_path}")
            
    def apply_execution_realism(self, trades):
        """
        Applies dynamic slippage and latency simulation.
        Slippage = spread * volatility_multiplier
        """
        # Dynamic Slippage based on spread and volatility (ATR)
        if 'spread' in trades.columns and trades['spread'].mean() > 0:
            spread_factor = trades['spread'] / trades['close']
            if 'atr_14' in trades.columns:
                vol_mult = (trades['atr_14'] / trades['close']) / (trades['atr_14'] / trades['close']).mean()
                vol_mult = vol_mult.fillna(1.0).clip(0.5, 3.0)
            else:
                vol_mult = 1.0
            dynamic_slippage = (spread_factor * 0.5) * vol_mult
        else:
            dynamic_slippage = self.slippage_base
            
        latency_penalty = (self.latency_ms / 100) * 0.0001
        trades['execution_cost'] = self.fee_rate + dynamic_slippage + latency_penalty
        trades['total_cost'] = trades['execution_cost'] * 2
        
        return_col = 'future_return_t1' if 'future_return_t1' in trades.columns else 'future_return'
        trades['simulated_return'] = trades[return_col] - trades['total_cost']
        return trades

    def run_backtest(self, prob_threshold=0.7, use_meta=True):
        if not os.path.exists(self.data_path):
            print("Data for backtesting not found.")
            return
            
        df = pd.read_parquet(self.data_path)
        
        # Use Registry for features
        primary_feature_cols = load_feature_cols('primary')
        if primary_feature_cols is None:
            exclude_cols = ['timestamp', 'symbol', 'target', 'future_price', 'future_return', 'type']
            primary_feature_cols = [c for c in df.columns if c not in exclude_cols]
            
        # Primary Prediction
        try:
            # Explicitly select registered features and convert to numpy for XGBoost safety
            X_primary = df[primary_feature_cols].values
            df['pred_prob'] = self.model.predict_proba(X_primary)[:, 1]
        except Exception as e:
            print(f"Primary prediction failed: {e}")
            return
            
        # Meta Prediction
        if use_meta and self.meta_model is not None:
            try:
                meta_feature_cols = load_feature_cols('meta')
                # Convert to numpy to avoid DataFrame dtype metadata issues
                X_meta_np = df[meta_feature_cols].values
                df['meta_pred'] = self.meta_model.predict(X_meta_np)
            except Exception as e:
                print(f"Meta prediction failed: {e}")
                df['meta_pred'] = 1
        else:
            df['meta_pred'] = 1
            
        # Filter trades
        trades = df[(df['pred_prob'] >= prob_threshold) & (df['meta_pred'] == 1)].copy()
        
        if len(trades) == 0:
            print(f"No trades triggered at threshold {prob_threshold} (Meta: {use_meta}).")
            return
            
        trades = self.apply_execution_realism(trades)
        
        # Metrics
        win_rate = (trades['simulated_return'] > 0).mean()
        avg_win = trades[trades['simulated_return'] > 0]['simulated_return'].mean()
        avg_loss = abs(trades[trades['simulated_return'] <= 0]['simulated_return'].mean())
        expectancy = calculate_expectancy(win_rate, avg_win, avg_loss)
        
        trades['cumulative_return'] = (1 + trades['simulated_return']).cumprod()
        max_drawdown = calculate_max_drawdown(trades['cumulative_return'])
        sharpe = calculate_sharpe_ratio(trades['simulated_return'])
        recovery = calculate_recovery_factor(trades['cumulative_return'])
        
        print(f"=== {'META' if use_meta else 'PRIMARY'} Backtest Results (Threshold: {prob_threshold}) ===")
        print(f"Total Trades: {len(trades)}")
        print(f"Avg Round-Trip Cost: {trades['total_cost'].mean()*100:.4f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Expectancy: {expectancy*100:.4f}% per trade")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Recovery Factor: {recovery:.2f}")
        
        trades.to_parquet('../data/processed/backtest_trades.parquet', engine='pyarrow')

if __name__ == "__main__":
    bt = Backtester()
    print("Running Backtest WITH Meta-Labeling...")
    bt.run_backtest(prob_threshold=0.5, use_meta=True)
    print("\nRunning Backtest WITHOUT Meta-Labeling...")
    bt.run_backtest(prob_threshold=0.5, use_meta=False)
