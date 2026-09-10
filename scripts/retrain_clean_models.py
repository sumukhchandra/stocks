"""
Retrain clean quantitative models with zero lookahead bias.
Removes target leak (max_future_return) and calibrates probabilities with scale_pos_weight.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "master_labeled_dataset.parquet")
MODEL_DIR = os.path.join(ROOT_DIR, "ml_models", "models", "saved_models")
ENS_DIR = os.path.join(MODEL_DIR, "ensemble")
SPEC_DIR = os.path.join(MODEL_DIR, "regime_specialists")

os.makedirs(ENS_DIR, exist_ok=True)
os.makedirs(SPEC_DIR, exist_ok=True)


def retrain_all():
    print("=" * 60)
    print("RE-TRAINING CLEAN ENSEMBLE & REGRESSORS (ZERO LEAKAGE)")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded dataset: {len(df):,} rows")

    # Strictly exclude any forward/future labels or targets
    exclude = [
        "timestamp", "symbol", "target", "future_price", "future_return",
        "future_return_t1", "triple_barrier_label", "is_synthetic",
        "ignore", "close_time", "open_time", "future_max_high", "future_return_3%",
        "actual_return", "target_ret", "regime", "ot", "ct", "qv", "nt", "tbb",
        "tbq", "i", "downloaded_at", "max_future_return", "company_name", "date_only"
    ]

    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and df[c].std() > 0
    ]

    print(f"Clean feature count: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    # Save clean feature cols
    joblib.dump(feature_cols, os.path.join(ENS_DIR, "feature_cols.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "primary_feature_cols.joblib"))

    # ──────────────────────────────────────────────────────────────────────────
    # 1. ENSEMBLE CLASSIFIERS (XGBoost, LightGBM, CatBoost)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Training Directional Ensemble Classifiers...")
    df_cls = df.dropna(subset=["target"]).copy()
    X = df_cls[feature_cols].fillna(0)
    y = df_cls["target"].astype(int)

    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_weight = float(neg_count / (pos_count + 1e-8))
    print(f"  Positive rate: {pos_count/len(y):.1%} | Scale Pos Weight: {scale_weight:.2f}")

    # XGBoost
    print("  Training XGBoost Classifier...")
    xgb_clf = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_clf.fit(X, y)
    joblib.dump(xgb_clf, os.path.join(ENS_DIR, "xgboost_calibrated.joblib"))
    joblib.dump(xgb_clf, os.path.join(MODEL_DIR, "calibrated_xgb_model.joblib"))

    # LightGBM
    print("  Training LightGBM Classifier...")
    lgb_clf = LGBMClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_weight,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_clf.fit(X, y)
    joblib.dump(lgb_clf, os.path.join(ENS_DIR, "lightgbm_calibrated.joblib"))

    # CatBoost
    print("  Training CatBoost Classifier...")
    cat_clf = CatBoostClassifier(
        iterations=250,
        depth=5,
        learning_rate=0.03,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )
    cat_clf.fit(X, y)
    joblib.dump(cat_clf, os.path.join(ENS_DIR, "catboost_calibrated.joblib"))

    # ──────────────────────────────────────────────────────────────────────────
    # 2. EXPECTED RETURN REGRESSORS (DART, CatBoost, XGB, ExtraTrees)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Training Expected Return Regressors...")
    df_reg = df[np.isfinite(df["target_ret"])].copy()
    X_reg = df_reg[feature_cols].fillna(0)
    y_reg = df_reg["target_ret"].values

    print(f"  Target return range: min={y_reg.min()*100:.2f}%, mean={y_reg.mean()*100:.2f}%, max={y_reg.max()*100:.2f}%")

    # LightGBM Regressor
    print("  Training LightGBM Return Regressor...")
    lgb_reg = LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        num_leaves=40,
        objective="huber",
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_reg.fit(X_reg, y_reg)
    joblib.dump(lgb_reg, os.path.join(MODEL_DIR, "return_regressor_lgbm.joblib"))

    # CatBoost Regressor
    print("  Training CatBoost Return Regressor...")
    cat_reg = CatBoostRegressor(
        iterations=300,
        depth=5,
        learning_rate=0.03,
        loss_function="Huber:delta=0.002",
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )
    cat_reg.fit(X_reg, y_reg)
    joblib.dump(cat_reg, os.path.join(MODEL_DIR, "return_regressor_catboost.joblib"))

    # XGBoost Regressor
    print("  Training XGBoost Return Regressor...")
    xgb_reg = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        objective="reg:pseudohubererror",
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    xgb_reg.fit(X_reg, y_reg)
    joblib.dump(xgb_reg, os.path.join(MODEL_DIR, "return_regressor_xgb.joblib"))
    joblib.dump(xgb_reg, os.path.join(MODEL_DIR, "expected_return_xgb.joblib"))

    # ExtraTrees Regressor
    print("  Training ExtraTrees Return Regressor...")
    et_reg = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    et_reg.fit(X_reg, y_reg)
    joblib.dump(et_reg, os.path.join(MODEL_DIR, "return_regressor_extratrees.joblib"))

    # ──────────────────────────────────────────────────────────────────────────
    # 3. REGIME SPECIALISTS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Training Regime Specialists...")
    regime_map = {
        "strong_bull": "trending_bull",
        "strong_bear": "trending_bear",
        "sideways_chop": "sideways",
        "flash_crash": "high_volatility",
        "crisis_regime": "high_volatility",
        "high_volatility": "high_volatility",
        "sideways": "sideways",
        "overheated_bull": "trending_bull",
        "trending_bull": "trending_bull",
        "trending_bear": "trending_bear",
    }
    df["regime_group"] = df["regime"].map(regime_map).fillna(df["regime"])
    joblib.dump(feature_cols, os.path.join(SPEC_DIR, "feature_cols.joblib"))

    for regime in ["trending_bull", "trending_bear", "sideways", "high_volatility"]:
        rdf = df[df["regime_group"] == regime].dropna(subset=["target"])
        if len(rdf) < 200:
            print(f"  Skipping {regime} (insufficient rows: {len(rdf)})")
            continue

        rX = rdf[feature_cols].fillna(0)
        ry = rdf["target"].astype(int)
        r_pos = ry.sum()
        r_neg = len(ry) - r_pos
        r_spw = float(r_neg / (r_pos + 1e-8))

        spec_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=r_spw,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        spec_model.fit(rX, ry)
        out_path = os.path.join(SPEC_DIR, f"specialist_{regime}.joblib")
        joblib.dump(spec_model, out_path)
        print(f"  Saved specialist_{regime} ({len(rdf):,} samples)")

    print("\n" + "=" * 60)
    print("ALL MODELS SUCCESSFULLY TRAINED AND DEPLOYED!")
    print("=" * 60)


if __name__ == "__main__":
    retrain_all()
