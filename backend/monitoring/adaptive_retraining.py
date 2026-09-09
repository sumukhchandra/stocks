import pandas as pd
import numpy as np
import os
import joblib
import sys
import subprocess
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from monitoring.feature_drift import calculate_psi
except ImportError:
    calculate_psi = None

try:
    from monitoring.calibration_monitor import calculate_brier_score
except ImportError:
    calculate_brier_score = None

class AdaptiveRetrainingManager:
    """
    Monitors system health and triggers retraining based on drift or performance degradation.
    """
    def __init__(self, thresholds=None, project_root=None):
        self.thresholds = thresholds or {
            'psi': 0.2,            # Population Stability Index (High drift)
            'brier_score': 0.25,   # Calibration collapse
            'sharpe_drop': 0.5,    # 50% drop in rolling Sharpe
            'regime_shift': True   # Severe shift in regime distribution
        }
        self.history_path = 'logs/live_metrics/retraining_history.csv'
        self.project_root = project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.retraining_scripts = [
            'models/train_ensemble.py',
            'models/train_regime_specialists.py',
            'models/expected_return_regressor.py',
            'models/meta_labeling.py',
            'models/lstm_orderflow_model.py'
        ]

    def check_triggers(self, current_metrics):
        """
        Checks if any thresholds have been breached.
        """
        triggers = []
        
        if current_metrics.get('psi', 0) > self.thresholds['psi']:
            triggers.append('PSI_DRIFT_HIGH')
            
        if current_metrics.get('brier', 0) > self.thresholds['brier_score']:
            triggers.append('CALIBRATION_COLLAPSE')
            
        if current_metrics.get('sharpe_ratio', 1.0) < (current_metrics.get('baseline_sharpe', 0) * (1 - self.thresholds['sharpe_drop'])):
            triggers.append('PERFORMANCE_DETERIORATION')
            
        return triggers

    def trigger_retraining(self, reason):
        """
        Orchestrates the retraining pipeline.
        """
        print(f"[!] ADAPTIVE RETRAINING TRIGGERED: {reason}")
        self._append_history(reason, 'STARTED')

        try:
            for script in self.retraining_scripts:
                print(f"[RETRAIN] Running {script}...")
                subprocess.run(
                    [sys.executable, script],
                    cwd=self.project_root,
                    check=True
                )

            self._append_history(reason, 'COMPLETED')
            print("[RETRAIN] Completed successfully.")
            return True
        except subprocess.CalledProcessError as exc:
            self._append_history(reason, f'FAILED:{exc.returncode}')
            print(f"[RETRAIN] Failed while running {exc.cmd}: exit code {exc.returncode}")
            return False

    def _append_history(self, reason, status):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'status': status
        }
        history_path = os.path.join(self.project_root, self.history_path)
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        pd.DataFrame([log_entry]).to_csv(
            history_path,
            mode='a',
            header=not os.path.exists(history_path),
            index=False
        )

if __name__ == "__main__":
    print("Adaptive Retraining Manager Initialized. Monitoring for drift and performance triggers.")
