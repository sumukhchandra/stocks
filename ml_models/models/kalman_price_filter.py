import numpy as np
import pandas as pd

class KalmanPriceFilter:
    """
    Kalman Filter for real-time noise reduction in price data.
    Estimates the hidden 'Fair Price' state.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-3):
        self.q = process_noise  # Process variance
        self.r = measurement_noise  # Measurement variance
        self.x = None  # Estimated state
        self.p = 1.0   # Estimated error covariance

    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x

        # Prediction step
        p_prior = self.p + self.q
        
        # Update step
        k_gain = p_prior / (p_prior + self.r)
        self.x = self.x + k_gain * (measurement - self.x)
        self.p = (1 - k_gain) * p_prior
        
        return self.x

    def apply_to_series(self, prices):
        filtered = []
        self.x = None # reset
        for p in prices:
            filtered.append(self.update(p))
        return np.array(filtered)

if __name__ == "__main__":
    # Mock noisy price data
    base_price = 50000
    noise = np.random.normal(0, 5, 100)
    prices = base_price + np.cumsum(np.random.normal(0, 2, 100)) + noise
    
    kf = KalmanPriceFilter(process_noise=1.0, measurement_noise=25.0)
    filtered_prices = kf.apply_to_series(prices)
    
    print(f"Original Price Sample: {prices[:5]}")
    print(f"Filtered Price Sample: {filtered_prices[:5]}")
