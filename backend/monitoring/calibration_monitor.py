import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import os

class CalibrationMonitor:
    """
    Monitors model probability calibration and confidence.
    """
    def __init__(self, y_true, y_prob):
        self.y_true = y_true
        self.y_prob = y_prob

    def calculate_ece(self, n_bins=10):
        """
        Calculates Expected Calibration Error (ECE).
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (self.y_prob > bin_lower) & (self.y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(self.y_true[in_bin])
                avg_confidence_in_bin = np.mean(self.y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def get_metrics(self):
        brier = brier_score_loss(self.y_true, self.y_prob)
        ece = self.calculate_ece()
        
        # Reliability curve data
        prob_true, prob_pred = calibration_curve(self.y_true, self.y_prob, n_bins=10)
        
        return {
            'brier_score': brier,
            'ece': ece,
            'prob_true': prob_true,
            'prob_pred': prob_pred
        }

    def plot_reliability_diagram(self, output_path='research/plots/reliability_diagram.png'):
        metrics = self.get_metrics()
        
        plt.figure(figsize=(8, 8))
        plt.plot(metrics['prob_pred'], metrics['prob_true'], "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives (Actual)")
        plt.xlabel("Mean predicted probability (Confidence)")
        plt.title(f"Reliability Diagram (Brier: {metrics['brier_score']:.4f}, ECE: {metrics['ece']:.4f})")
        plt.legend()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"Reliability diagram saved to {output_path}")

if __name__ == "__main__":
    # Example with dummy data
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.1, 0.6, 0.9, 0.8, 0.2, 0.7])
    
    monitor = CalibrationMonitor(y_true, y_prob)
    metrics = monitor.get_metrics()
    print("Calibration Metrics:")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    # monitor.plot_reliability_diagram()
