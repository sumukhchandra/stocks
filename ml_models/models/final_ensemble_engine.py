import pandas as pd
import numpy as np
import joblib
import os
import torch
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ml_models.validation
import ml_models.validation.purged_cv
from ml_models.validation.purged_cv import PurgedWalkForwardCV
from backend.execution.regime_router import RegimeRouter
from backend.risk.dynamic_position_sizer import DynamicPositionSizer
from ml_models.models.lstm_orderflow_model import LSTMOrderFlowModel
from ml_models.models.expected_return_regressor import ExpectedReturnRegressor

class FinalEnsembleEngine:
    """
    The integrated decision engine: Ensemble + Specialists + LSTM + Meta Model + Regressor.
    """
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Default to the absolute path of the saved_models directory relative to this file
            self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models'))
        else:
            self.model_dir = model_dir
            
        self.ensemble_dir = os.path.join(self.model_dir, 'ensemble')
        self.specialist_dir = os.path.join(self.model_dir, 'regime_specialists')
        
        # 1. Load Universal Ensemble
        self.universal_models = {}
        for name in ['xgboost', 'lightgbm', 'catboost']:
            path = os.path.join(self.ensemble_dir, f'{name}_calibrated.joblib')
            if os.path.exists(path):
                self.universal_models[name] = joblib.load(path)
        self.primary_features = joblib.load(os.path.join(self.ensemble_dir, 'feature_cols.joblib'))

        # 2. Load Expected Return Regressor (Upgrade 1: Ensemble)
        self.return_models = {}
        self.target_transformer = None
        for name in ['lgbm', 'catboost', 'xgb', 'extratrees']:
            path = os.path.join(self.model_dir, f'return_regressor_{name}.joblib')
            if os.path.exists(path):
                self.return_models[name] = joblib.load(path)
        
        # Backward compat: fall back to single regressor if ensemble not available
        reg_path = os.path.join(self.model_dir, 'expected_return_xgb.joblib')
        if not self.return_models and os.path.exists(reg_path):
            self.return_models['xgb'] = joblib.load(reg_path)
        self.regressor = self.return_models.get('xgb')  # backward compat
                
        # 2. Load Regime Specialists
        self.specialists = {}
        if os.path.exists(self.specialist_dir):
            for f in os.listdir(self.specialist_dir):
                if f.startswith('specialist_') and f.endswith('.joblib'):
                    regime_name = f.replace('specialist_', '').replace('.joblib', '')
                    self.specialists[regime_name] = joblib.load(os.path.join(self.specialist_dir, f))
        
        # 3. Load Meta Model
        meta_path = os.path.join(self.model_dir, 'meta_xgb_model.joblib')
        self.meta_model = joblib.load(meta_path) if os.path.exists(meta_path) else None
        self.meta_features = joblib.load(os.path.join(self.model_dir, 'meta_feature_cols.joblib')) if os.path.exists(os.path.join(self.model_dir, 'meta_feature_cols.joblib')) else []

        # 4. Load LSTM
        lstm_path = os.path.join(self.model_dir, 'lstm', 'lstm_model.pth')
        self.lstm_features = ['vpin', 'imbalance', 'spread', 'volume_delta', 'ofi']
        if os.path.exists(lstm_path):
            self.lstm = LSTMOrderFlowModel(input_dim=len(self.lstm_features))
            self.lstm.load_state_dict(torch.load(lstm_path))
            self.lstm.eval()
            self.lstm_scaler = joblib.load(os.path.join(self.model_dir, 'lstm', 'scaler.joblib'))
        else:
            self.lstm = None
            
        # 5. Initialize Components
        self.router = RegimeRouter({
            'sideways': 'specialist_sideways',
            'trending_bull': 'specialist_trending_bull',
            'trending_bear': 'specialist_trending_bear',
            'high_volatility': 'specialist_high_volatility',
            'crisis_regime': 'specialist_crisis_regime'
        })
        self.sizer = DynamicPositionSizer()

    def predict_universal(self, X):
        if not self.universal_models: return 0.5
        probas = [m.predict_proba(X)[:, 1] for m in self.universal_models.values()]
        return np.mean(probas, axis=0)

    def predict_return_ensemble(self, X):
        """
        Predict expected return using the 4-model quant ensemble (DART, CatBoost, XGB, ExtraTrees).
        """
        if not self.return_models:
            return 0.0
        
        weights = {'lgbm': 0.35, 'catboost': 0.30, 'xgb': 0.25, 'extratrees': 0.10}
        total_weight = 0.0
        weighted_pred = 0.0
        
        for name, model in self.return_models.items():
            w = weights.get(name, 1.0)
            pred = model.predict(X)
            weighted_pred += pred * w
            total_weight += w
        
        if total_weight > 0:
            final_pred = weighted_pred / total_weight
        else:
            final_pred = np.zeros(len(X))
        
        return final_pred[0] if len(final_pred) == 1 else final_pred

    def _ensure_features(self, df):
        """Ensures all required features for LSTM and Ensemble are present."""
        working_df = df.copy()
        if 'ofi' not in working_df.columns and 'volume_delta' in working_df.columns:
            # Calculate OFI proxy if missing
            working_df['ofi'] = working_df['volume_delta'].rolling(5).mean().fillna(0)
        
        # List of required features that might be missing from some datasets
        required_features = self.lstm_features + ['last_macro_severity', 'macro_risk_factor', 'volatility', 'regime']
        
        for col in required_features:
            if col not in working_df.columns:
                if col == 'regime':
                    working_df[col] = 'sideways'
                else:
                    working_df[col] = 0.0
                    if col == 'volatility':
                        working_df[col] = 0.02 # Reasonable default
        return working_df

    def _compute_trajectory_features(self, df_window):
        """
        Upgrade 4: Compute trajectory-aware features from the full window
        and inject them into the current state row.
        """
        if 'returns' not in df_window.columns:
            df_window = df_window.copy()
            df_window['returns'] = df_window['close'].pct_change()
        
        returns = df_window['returns'].fillna(0).values
        
        # Recent momentum (sum of last 5 returns)
        recent_momentum = np.sum(returns[-5:]) if len(returns) >= 5 else 0.0
        # Older momentum (sum of returns from -10 to -5)
        older_momentum = np.sum(returns[-10:-5]) if len(returns) >= 10 else 0.0
        # Price acceleration (recent vs older momentum)
        price_acceleration = recent_momentum - older_momentum
        # Recent volatility (std of last 10 returns)
        recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else 0.02
        # Longer-term vol (std of last 50 returns)
        long_vol = np.std(returns[-50:]) if len(returns) >= 50 else recent_vol
        
        return {
            'trajectory_momentum': recent_momentum,
            'trajectory_acceleration': price_acceleration,
            'trajectory_recent_vol': recent_vol,
            'trajectory_vol_ratio': recent_vol / max(long_vol, 1e-8),
        }

    def evaluate_state(self, df_window):
        """
        Comprehensive evaluation of the current market state.
        """
        df_window = self._ensure_features(df_window)
        current_state = df_window.iloc[-1:]
        
        # 1. Primary Signal (Universal Ensemble)
        # Ensure we only use features the models were trained on
        X_univ = current_state.reindex(columns=self.primary_features, fill_value=0)
        universal_prob = self.predict_universal(X_univ)[0]
        
        # 2. Regime-Aware Signal
        regime = current_state['regime'].iloc[0]
        specialist_prob = universal_prob # Default
        specialist = self.specialists.get(regime)
        if specialist:
            # Handle potential feature mismatch for specialists
            try:
                # Correct way to get feature names for XGBClassifier
                spec_features = specialist.feature_names_in_
                X_spec = current_state.reindex(columns=spec_features, fill_value=0)
                specialist_prob = specialist.predict_proba(X_spec)[:, 1][0]
            except Exception as e:
                # Fallback to primary features if booster not available or other issue
                X_spec = current_state.reindex(columns=self.primary_features, fill_value=0)
                # Filter out is_synthetic if it was in primary but not specialist
                if 'is_synthetic' in X_spec.columns:
                    X_spec = X_spec.drop(columns=['is_synthetic'])
                specialist_prob = specialist.predict_proba(X_spec)[:, 1][0]
            
        # 3. LSTM Signal
        lstm_prob = 0.5
        if self.lstm and len(df_window) >= 16:
            features_data = df_window[self.lstm_features].tail(16).values
            scaled_data = self.lstm_scaler.transform(features_data)
            x_tensor = torch.FloatTensor(scaled_data).unsqueeze(0)
            with torch.no_grad():
                lstm_prob = self.lstm(x_tensor).item()
                
        # 4. Final Blending (Dynamic Active Model Weighting)
        active_weights = [1.0]
        active_probs = [universal_prob]
        
        if self.specialists and specialist:
            active_weights.append(0.5)
            active_probs.append(specialist_prob)
            
        if self.lstm and len(df_window) >= 16:
            active_weights.append(0.3)
            active_probs.append(lstm_prob)
            
        final_prob = sum(w * p for w, p in zip(active_weights, active_probs)) / sum(active_weights)
        
        # 5. Expected Return (Magnitude) -- Pure High-Precision ML Quant Ensemble
        expected_ret = 0.0
        if self.return_models:
            X_reg = current_state.reindex(columns=self.primary_features, fill_value=0)
            expected_ret = self.predict_return_ensemble(X_reg)
        elif self.regressor:
            # Backward compat with old single regressor
            X_reg = current_state.reindex(columns=self.primary_features, fill_value=0)
            expected_ret = self.regressor.predict(X_reg)[0]
        
        # 6. Confidence Score (Meta Model)
        confidence_score = 0.5
        if self.meta_model:
            X_meta = current_state.copy()
            X_meta['pred_prob'] = universal_prob # The meta model was trained on universal_prob
            X_meta = X_meta.reindex(columns=self.meta_features, fill_value=0)
            confidence_score = self.meta_model.predict_proba(X_meta)[:, 1][0]
            
        # 7. Position Sizing & Risk
        macro_sev = current_state['last_macro_severity'].iloc[0]
        macro_risk = current_state['macro_risk_factor'].iloc[0]
        volatility = current_state['volatility'].iloc[0] if 'volatility' in current_state.columns else 0.02
        
        price = current_state['close'].iloc[0]
        stop_loss = price * 0.98 if final_prob > 0.5 else price * 1.02 # Simple 2% SL
        
        base_size = self.sizer.calculate_size(price, stop_loss, final_prob, volatility)
        risk_mult = self.router.adjust_risk_multiplier(regime, macro_sev, macro_risk)
        
        # Final sizing adjustment based on confidence
        conf_mult = (confidence_score - 0.5) * 2.0 if confidence_score > 0.5 else 0.1
        recommended_size = base_size * risk_mult * max(0.1, conf_mult)
        
        # 8. Holding Time Estimation
        # Shifted toward swing: favor longer horizons to absorb tax/fees.
        base_holding_map = {
            'sideways': 15,          # 15 periods (e.g. 75 mins)
            'trending_bull': 120,    # 120 periods (e.g. 10 hours)
            'trending_bear': 60,     # 60 periods (e.g. 5 hours)
            'high_volatility': 10,   # 10 periods
            'crisis_regime': 2       # 2 periods
        }
        periods_to_hold = base_holding_map.get(regime, 30)
        expected_hold_mins = periods_to_hold * 5 # Assuming 5m base bars

        return {
            'timestamp': current_state['timestamp'].iloc[0],
            'price': price,
            'regime': regime,
            'final_probability': final_prob,
            'expected_return': expected_ret,
            'confidence_score': confidence_score,
            'macro_risk': macro_risk,
            'recommended_position_size': recommended_size,
            'expected_holding_mins': expected_hold_mins,
            'signals': {
                'universal': universal_prob,
                'specialist': specialist_prob,
                'lstm': lstm_prob if 'lstm_prob' in locals() else 0.5
            }
        }

if __name__ == "__main__":
    engine = FinalEnsembleEngine()
    data_path = 'data/processed/master_labeled_dataset.parquet'
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        if len(df) >= 16:
            # Test on a few samples from different regimes
            for regime in df['regime'].unique()[:3]:
                sample_idx = df[df['regime'] == regime].index[0]
                if sample_idx >= 16:
                    window = df.iloc[sample_idx-15 : sample_idx+1]
                    decision = engine.evaluate_state(window)
                    print(f"\n--- Decision for Regime: {regime} ---")
                    for k, v in decision.items():
                        if k != 'signals': print(f"{k}: {v}")
    else:
        print("Data not found.")
