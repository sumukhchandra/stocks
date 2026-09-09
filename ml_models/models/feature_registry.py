import joblib
import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEATURE_REGISTRY_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

def save_feature_cols(feature_cols, model_name='primary'):
    os.makedirs(FEATURE_REGISTRY_DIR, exist_ok=True)
    path = os.path.join(FEATURE_REGISTRY_DIR, f'{model_name}_feature_cols.joblib')
    joblib.dump(feature_cols, path)
    print(f"Feature registry updated for {model_name}: {path}")

def load_feature_cols(model_name='primary'):
    path = os.path.join(FEATURE_REGISTRY_DIR, f'{model_name}_feature_cols.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    # Fallback for older sessions
    old_path = os.path.join(FEATURE_REGISTRY_DIR, 'feature_cols.joblib')
    if os.path.exists(old_path):
        return joblib.load(old_path)
    return None
