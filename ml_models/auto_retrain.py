"""
Smart auto-retrain system for the NSE stock trading models.

Three retrain triggers:
1. Daily at 9:00 AM IST (before market opens)
2. Performance-triggered: if rolling accuracy drops below threshold
3. Manual: user clicks "Force Retrain" in the dashboard
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.stocks_config import (
    DATA_FETCH_PERIOD,
    DATA_FETCH_INTERVAL,
    RETRAIN_HOUR,
    RETRAIN_MINUTE,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RETRAIN_LOG_FILE = os.path.join(SCRIPT_DIR, "logs", "retrain_log.jsonl")
os.makedirs(os.path.join(SCRIPT_DIR, "logs"), exist_ok=True)


def run_retrain(period=None, interval=None):
    """
    Full retrain pipeline:
    1. Fetch latest data for all stocks
    2. Retrain ensemble (XGBoost, LightGBM, CatBoost)
    3. Retrain expected return regressor
    4. Save models and log metadata

    Returns a dict with retrain results.
    """
    period = period or DATA_FETCH_PERIOD
    interval = interval or DATA_FETCH_INTERVAL

    result = {
        "status": "started",
        "start_time": datetime.now().isoformat(),
        "steps": [],
    }

    try:
        # -- Step 1: Fetch fresh data -------------------------------------
        print(f"[Retrain] Step 1/3: Fetching {period} of {interval} data...")
        from data.fetch_stock_data import build_stock_dataset

        df = build_stock_dataset(period=period, interval=interval)
        result["steps"].append({
            "step": "fetch_data",
            "status": "ok",
            "rows": len(df),
        })
        print(f"[Retrain] Fetched {len(df)} rows.")

        # -- Step 2: Retrain ensemble -------------------------------------
        print("[Retrain] Step 2/3: Training ensemble models...")
        import pandas as pd
        from ml_models.models.train_ensemble import ModelEnsemble

        exclude = [
            "timestamp", "symbol", "target", "future_price", "future_return",
            "future_return_t1", "triple_barrier_label", "is_synthetic",
            "ignore", "close_time", "open_time", "future_max_high",
            "future_return_3%", "actual_return", "target_ret", "regime",
            "ot", "ct", "qv", "nt", "tbb", "tbq", "i", "downloaded_at",
            "company_name",
        ]
        feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        feature_cols = [c for c in feature_cols if df[c].std() > 0]

        df = df.dropna(subset=["target"])
        X = df[feature_cols]
        y = df["target"]

        # 80/20 time-series split
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        ensemble = ModelEnsemble(model_dir="ml_models/models/saved_models/ensemble")
        ensemble.train_all(X_train, y_train, feature_cols)

        result["steps"].append({
            "step": "train_ensemble",
            "status": "ok",
            "features": len(feature_cols),
            "train_samples": len(X_train),
        })
        print(f"[Retrain] Ensemble trained on {len(X_train)} samples, {len(feature_cols)} features.")

        # -- Step 3: Retrain expected return ensemble regressor ------------
        print("[Retrain] Step 3/3: Training ensemble return regressor (3 models + quantile transform)...")
        from ml_models.models.expected_return_regressor import ExpectedReturnRegressor

        data_path = os.path.join(
            SCRIPT_DIR, "data", "processed", "master_labeled_dataset.parquet"
        )
        regressor = ExpectedReturnRegressor(model_dir="ml_models/models/saved_models")
        regressor.train(data_path)

        result["steps"].append({
            "step": "train_regressor",
            "status": "ok",
        })
        print("[Retrain] Ensemble regressor trained.")

        # -- Log retrain metadata -----------------------------------------
        result["status"] = "success"
        result["end_time"] = datetime.now().isoformat()
        result["dataset_size"] = len(df)
        result["feature_count"] = len(feature_cols)

        with open(RETRAIN_LOG_FILE, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

        # Update trader state with retrain timestamp
        trader_state_file = os.path.join(SCRIPT_DIR, "logs", "trader_state.json")
        if os.path.exists(trader_state_file):
            with open(trader_state_file, "r") as f:
                state = json.load(f)
            state["last_retrain"] = datetime.now().isoformat()
            with open(trader_state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

        print(f"[Retrain] COMPLETE! Dataset: {len(df)} rows, {len(feature_cols)} features.")
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()

        with open(RETRAIN_LOG_FILE, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

        print(f"[Retrain] ERROR: {e}")
        return result


def get_retrain_history():
    """Read all retrain log entries."""
    entries = []
    if os.path.exists(RETRAIN_LOG_FILE):
        with open(RETRAIN_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def should_auto_retrain_now():
    """Check if it's time for the daily auto-retrain (9:00 AM IST)."""
    now = datetime.now()
    if now.hour == RETRAIN_HOUR and now.minute < (RETRAIN_MINUTE + 5):
        # Check if already retrained today
        history = get_retrain_history()
        if history:
            last = history[-1]
            last_time = datetime.fromisoformat(last.get("start_time", "2000-01-01"))
            if last_time.date() == now.date():
                return False  # Already retrained today
        return True
    return False


if __name__ == "__main__":
    result = run_retrain()
    print(json.dumps(result, indent=2, default=str))
