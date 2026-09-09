import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp

class FeatureDriftDetector:
    """
    Detects distribution shifts in features over time or between datasets.
    """
    def __init__(self, reference_df, current_df):
        self.reference_df = reference_df
        self.current_df = current_df

    def calculate_psi(self, feature_name, buckets=10):
        """
        Calculates Population Stability Index (PSI) for a feature.
        PSI = sum((Actual % - Expected %) * ln(Actual % / Expected %))
        """
        ref_values = self.reference_df[feature_name].dropna()
        curr_values = self.current_df[feature_name].dropna()

        if len(ref_values) == 0 or len(curr_values) == 0:
            return np.nan

        # Create buckets based on reference data
        breakpoints = np.percentile(ref_values, np.linspace(0, 100, buckets + 1))
        # Ensure breakpoints are unique
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        def get_counts(data, bins):
            counts = np.histogram(data, bins=bins)[0]
            return counts / len(data)

        ref_percents = get_counts(ref_values, breakpoints)
        curr_percents = get_counts(curr_values, breakpoints)

        # Replace 0s with small value to avoid log(0)
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)

        psi_val = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        return psi_val

    def calculate_ks_drift(self, feature_name):
        """Calculates KS test statistic and p-value."""
        res = ks_2samp(self.reference_df[feature_name], self.current_df[feature_name])
        return res.statistic, res.pvalue

    def run_drift_analysis(self, features):
        results = []
        for feature in features:
            if feature not in self.reference_df.columns or feature not in self.current_df.columns:
                continue
            
            psi = self.calculate_psi(feature)
            ks_stat, ks_p = self.calculate_ks_drift(feature)
            
            # Interpret PSI
            # < 0.1: No significant change
            # 0.1 - 0.25: Slight change
            # > 0.25: Significant change
            status = "STABLE"
            if psi > 0.25 or ks_p < 0.01:
                status = "DRIFTED"
            elif psi > 0.1:
                status = "WARNING"
                
            results.append({
                'feature': feature,
                'psi': psi,
                'ks_stat': ks_stat,
                'ks_pvalue': ks_p,
                'status': status
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage: compare real vs synthetic bootstrap
    real_path = 'data/processed/features_latest.parquet'
    synth_path = 'data/processed/synthetic_bootstrap.parquet'
    
    if os.path.exists(real_path) and os.path.exists(synth_path):
        ref = pd.read_parquet(real_path)
        curr = pd.read_parquet(synth_path)
        
        detector = FeatureDriftDetector(ref, curr)
        features = ['close', 'volume', 'vpin', 'mean_imbalance', 'mean_spread']
        drift_report = detector.run_drift_analysis(features)
        
        print("--- Feature Drift Report ---")
        print(drift_report)
    else:
        print("Data files not found for drift analysis.")
