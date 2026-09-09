import pandas as pd
import numpy as np
from hmmlearn import hmm
import joblib
import os

class RegimeDetector:
    def __init__(self, n_states=3, model_path='../models/saved_models/hmm_model.joblib'):
        self.n_states = n_states
        self.model_path = model_path
        self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100)
        
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True
        else:
            self.is_trained = False
            
    def prepare_features(self, df):
        """
        Extracts features for HMM training/prediction.
        Requires: close, atr_14, volume. (spread, imbalance if available).
        """
        df = df.copy()
        
        # Calculate returns
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
            df['rolling_vol'] = df['returns'].rolling(14).std()
        else:
            df['returns'] = 0
            df['rolling_vol'] = 0
            
        # Volume spikes (volume relative to 14-period MA)
        if 'volume' in df.columns:
            df['vol_ma'] = df['volume'].rolling(14).mean()
            df['volume_spike'] = df['volume'] / (df['vol_ma'] + 1e-8)
        else:
            df['volume_spike'] = 1.0
            
        hmm_features = ['returns', 'rolling_vol', 'volume_spike']
        if 'atr_14' in df.columns: hmm_features.append('atr_14')
        if 'spread' in df.columns: hmm_features.append('spread')
        if 'imbalance' in df.columns: hmm_features.append('imbalance')
        
        df_clean = df.dropna(subset=hmm_features)
        
        X = df_clean[hmm_features].values
        return X, df_clean.index
        
    def train(self, df):
        """Trains the HMM model."""
        X, _ = self.prepare_features(df)
        if X is None or len(X) < 100:
            print("Not enough data to train HMM.")
            return False
            
        print("Training HMM Regime Detector...")
        self.model.fit(X)
        self.is_trained = True
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print("HMM Model saved.")
        return True
        
    def predict_regime(self, df):
        """Predicts the current regime for the dataframe."""
        if not self.is_trained:
            print("HMM Model not trained. Returning default regime.")
            return pd.Series(0, index=df.index, dtype=float)
            
        X, valid_idx = self.prepare_features(df)
        if X is None or len(X) == 0:
            return pd.Series(0, index=df.index, dtype=float)
            
        states = self.model.predict(X)
        
        regime_series = pd.Series(np.nan, index=df.index, dtype=float)
        regime_series.loc[valid_idx] = states
        
        # Forward fill any missing regimes at the beginning
        regime_series = regime_series.ffill().fillna(0)
        
        return regime_series

if __name__ == "__main__":
    pass
