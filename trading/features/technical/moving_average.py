"""
Moving Average Indicators: SMA, EMA, and cross signals.
"""

import pandas as pd
import numpy as np


class MovingAverageFeatures:
    @staticmethod
    def compute(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        df = df.copy()
        for window in [9, 21, 50, 200]:
            df[f"sma_{window}"] = df.groupby("symbol")[price_col].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            df[f"ema_{window}"] = df.groupby("symbol")[price_col].transform(
                lambda s: s.ewm(span=window, adjust=False).mean()
            )
            df[f"price_to_sma_{window}"] = (df[price_col] - df[f"sma_{window}"]) / (df[f"sma_{window}"] + 1e-8)

        # Fast/Slow EMA Ratio
        df["ema_9_21_ratio"] = df["ema_9"] / (df["ema_21"] + 1e-8)
        return df
