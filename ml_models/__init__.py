"""
ml_models package: Machine learning models, feature engineering, and validation.
"""
import sys

# Import and expose primary models and features
from ml_models.models.final_ensemble_engine import FinalEnsembleEngine
from ml_models.models.train_ensemble import ModelEnsemble
from ml_models.models.expected_return_regressor import ExpectedReturnRegressor
from ml_models.models.feature_registry import load_feature_cols, save_feature_cols
from ml_models.models.lstm_orderflow_model import LSTMOrderFlowModel

# Backwards compatibility shims for direct submodule imports
import ml_models.models.final_ensemble_engine as _fee
import ml_models.models.train_ensemble as _te
import ml_models.models.expected_return_regressor as _err
import ml_models.models.feature_registry as _fr
import ml_models.models.lstm_orderflow_model as _lom
import ml_models.features as _feat
import ml_models.labels as _lab
import ml_models.validation as _val

sys.modules.setdefault('ml_models.final_ensemble_engine', _fee)
sys.modules.setdefault('ml_models.train_ensemble', _te)
sys.modules.setdefault('ml_models.expected_return_regressor', _err)
sys.modules.setdefault('ml_models.feature_registry', _fr)
sys.modules.setdefault('ml_models.lstm_orderflow_model', _lom)
sys.modules.setdefault('features', _feat)
sys.modules.setdefault('labels', _lab)
sys.modules.setdefault('validation', _val)

__all__ = [
    'FinalEnsembleEngine',
    'ModelEnsemble',
    'ExpectedReturnRegressor',
    'load_feature_cols',
    'save_feature_cols',
    'LSTMOrderFlowModel',
]
