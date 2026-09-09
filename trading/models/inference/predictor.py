"""
5-Minute ML Predictor: Orchestrates model inference and attaches model_version.
"""

from typing import Dict, Any, Optional
import pandas as pd
from ..architectures.ensemble import EnsemblePredictor
from ..registry.model_registry import ModelRegistry


class FiveMinutePredictor:
    def __init__(self, model_dir: Optional[str] = None):
        self.ensemble = EnsemblePredictor(model_dir=model_dir)
        self.registry = ModelRegistry()
        active_info = self.registry.get_active_model_info()
        if active_info:
            self.model_version = active_info["version_id"]
        else:
            # Auto-register default ensemble if empty
            self.model_version = self.registry.register_model(
                algorithm_name="Ensemble_CatBoost_LGBM_XGB_RF",
                artifact_path=self.ensemble.model_dir,
                metrics={"status": "initial_production_baseline"},
                features=self.ensemble.primary_features,
                version_id="ens_prod_v1.0"
            )

    def predict(self, symbol: str, feature_window: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate 5-minute forward prediction for symbol.
        Returns dict containing expected_return, probability, confidence, and model_version.
        """
        current_price = float(feature_window["close"].iloc[-1]) if "close" in feature_window.columns else 0.0
        prob = self.ensemble.predict_probability(feature_window)
        exp_ret = self.ensemble.predict_expected_return(feature_window)
        predicted_price = current_price * (1.0 + exp_ret)

        return {
            "symbol": symbol,
            "horizon_minutes": 5,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "expected_return": round(exp_ret, 5),
            "expected_return_pct": round(exp_ret * 100, 3),
            "probability": round(prob, 4),
            "confidence": round(prob, 4),
            "model_version": self.model_version,
            "is_bullish": exp_ret > 0 and prob >= 0.50
        }
