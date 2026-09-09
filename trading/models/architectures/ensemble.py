"""
Ensemble Model Architecture: Combines CatBoost, LightGBM, XGBoost, Random Forest, and Expected Return Regressor.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class EnsemblePredictor:
    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            model_dir = os.path.join(root, "ml_models", "models", "saved_models")
        self.model_dir = model_dir
        self.models = {}
        self.primary_features = []
        self._load_models()

    def _load_models(self):
        feat_path = os.path.join(self.model_dir, "feature_names.json")
        if os.path.exists(feat_path):
            import json
            with open(feat_path, "r") as f:
                self.primary_features = json.load(f)

        for name in ["lightgbm", "xgboost", "catboost", "random_forest", "return_regressor"]:
            for ext in [".joblib", ".pkl", ".cbm"]:
                p = os.path.join(self.model_dir, f"{name}{ext}")
                if os.path.exists(p):
                    try:
                        if ext == ".cbm":
                            from catboost import CatBoostClassifier
                            cb = CatBoostClassifier()
                            cb.load_model(p)
                            self.models[name] = cb
                        else:
                            self.models[name] = joblib.load(p)
                    except Exception:
                        pass
                    break

    def is_loaded(self) -> bool:
        return len(self.models) > 0

    def predict_probability(self, X: pd.DataFrame) -> float:
        """Ensemble win probability [0.0 to 1.0]."""
        if not self.is_loaded():
            return 0.50

        # Align features
        if self.primary_features:
            X_eval = X.reindex(columns=self.primary_features, fill_value=0.0)
        else:
            X_eval = X.select_dtypes(include=[np.number]).fillna(0.0)

        probs = []
        for name, m in self.models.items():
            if name == "return_regressor":
                continue
            try:
                if hasattr(m, "predict_proba"):
                    p = float(m.predict_proba(X_eval.tail(1))[:, 1][0])
                    probs.append(p)
            except Exception:
                pass

        if not probs:
            return 0.50
        return float(np.mean(probs))

    def predict_expected_return(self, X: pd.DataFrame) -> float:
        """Expected 5-minute forward return decimal."""
        regressor = self.models.get("return_regressor")
        if regressor is None:
            return 0.001

        if self.primary_features:
            X_eval = X.reindex(columns=self.primary_features, fill_value=0.0)
        else:
            X_eval = X.select_dtypes(include=[np.number]).fillna(0.0)

        try:
            val = float(regressor.predict(X_eval.tail(1))[0])
            return val
        except Exception:
            return 0.001
