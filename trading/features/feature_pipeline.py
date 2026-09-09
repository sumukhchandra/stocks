"""
Feature Pipeline: End-to-end feature extraction for 5-minute systematic prediction.
"""

import pandas as pd
import numpy as np
from .technical.moving_average import MovingAverageFeatures
from .technical.rsi import RSIFeatures
from .technical.macd import MACDFeatures
from .technical.volatility import VolatilityFeatures
from .market.regime import MarketRegimeFeatures


class FeaturePipeline:
    def __init__(self):
        pass

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # 1. Moving Averages
        df = MovingAverageFeatures.compute(df)

        # 2. RSI Indicators
        df = RSIFeatures.compute(df)

        # 3. MACD
        df = MACDFeatures.compute(df)

        # 4. Volatility & Bollinger Bands
        df = VolatilityFeatures.compute(df)

        # 5. Market Regime & Return lags
        df = MarketRegimeFeatures.compute(df)

        # 6. Volume features
        if "volume" in df.columns:
            df["vol_sma_20"] = df.groupby("symbol")["volume"].transform(
                lambda s: s.rolling(20, min_periods=1).mean()
            )
            df["vol_ratio"] = df["volume"] / (df["vol_sma_20"] + 1e-8)

        # 7. Time features (Cyclical Hour & Minute)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            minute_of_day = ts.dt.hour * 60 + ts.dt.minute
            df["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
            df["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

        return df.fillna(0.0)

    def add_labels(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.001) -> pd.DataFrame:
        """
        Add 5-minute forward return labels for supervised training.
        horizon=1 corresponds to next 5-minute candle.
        """
        df = df.copy()
        df["target_return"] = df.groupby("symbol")["close"].pct_change(horizon).shift(-horizon)
        # Binary target: 1 = Return > threshold, 0 = otherwise
        df["target_direction"] = (df["target_return"] > threshold).astype(int)
        return df
