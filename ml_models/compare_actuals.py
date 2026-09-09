import os
import sys
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_models.final_ensemble_engine import FinalEnsembleEngine
from database.stocks_config import STOCK_SYMBOLS, STOCK_UNIVERSE

def compare_predictions():
    data_path = "QA/data/processed/master_labeled_dataset.parquet"
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    df = pd.read_parquet(data_path)
    df['date_only'] = df['timestamp'].dt.date
    
    # Get last full day
    dates = sorted(df['date_only'].unique())
    if len(dates) < 2:
        target_date = dates[-1]
    else:
        target_date = dates[-2]

    print(f"# Model Predictions vs Actual Returns ({target_date})")
    print("Comparing what the model predicted vs what actually happened.\n")
    print("| Time | Stock | Predicted P(Win) | Predicted Return | Actual Return | Error |")
    print("|---|---|---|---|---|---|")

    engine = FinalEnsembleEngine(model_dir="ml_models/saved_models")
    
    # We just want to sample some predictions (e.g., top 15 most confident ones)
    results = []

    for symbol in STOCK_SYMBOLS:
        symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
        day_data = symbol_data[symbol_data['date_only'] == target_date]
        
        for idx in range(len(day_data)):
            ts = day_data.iloc[idx]['timestamp']
            
            # Need 50 rows of history up to this point
            window = symbol_data[symbol_data['timestamp'] <= ts].tail(50)
            if len(window) < 50:
                continue
                
            actual_return = day_data.iloc[idx].get('target_ret', 0)
            if pd.isna(actual_return):
                continue
                
            try:
                window_clean = window.fillna(0)
                decision = engine.evaluate_state(window_clean)
                pred_prob = float(decision["final_probability"])
                pred_return = float(decision.get("expected_return", 0))
                
                results.append({
                    "time": ts.strftime('%H:%M'),
                    "symbol": symbol,
                    "company": STOCK_UNIVERSE.get(symbol, symbol),
                    "pred_prob": pred_prob,
                    "pred_return": pred_return,
                    "actual_return": actual_return,
                    "error": abs(pred_return - actual_return)
                })
            except Exception as e:
                continue

    # Sort by confidence and take top 15
    results.sort(key=lambda x: x["pred_prob"], reverse=True)
    
    for r in results[:15]:
        pred_ret_pct = r['pred_return'] * 100
        act_ret_pct = r['actual_return'] * 100
        error_pct = r['error'] * 100
        prob_str = f"{r['pred_prob']*100:.1f}%"
        print(f"| {r['time']} | {r['company']} | {prob_str} | {pred_ret_pct:+.3f}% | {act_ret_pct:+.3f}% | {error_pct:.3f}% |")
        
    print("\n*Note: 'Actual Return' is the actual highest return achievable over the next 6 intervals (30 mins) before hitting the stop loss barrier.*")

if __name__ == "__main__":
    compare_predictions()
