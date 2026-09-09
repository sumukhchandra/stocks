"""
Volatility and Bollinger Band indicators.
"""

import pandas as pd
import numpy as np


class VolatilityFeatures:
    @staticmethod
    def compute(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        df = df.copy()

        # Bollinger Bands (20, 2)
        window = 20
        df["bb_middle"] = df.groupby("symbol")[price_col].transform(lambda s: s.rolling(window, min_periods=1).mean())
        df["bb_std"] = df.groupby("symbol")[price_col].transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
        df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
        df["bb_pct_b"] = (df[price_col] - df["bb_lower"]) / ((df["bb_upper"] - df["bb_lower"]) + 1e-8)
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-8)

        # Average True Range (ATR 14)
        if all(c in df.columns for c in ["high", "low", "close"]):
            prev_close = df.groupby("symbol")["close"].shift(1)
            tr1 = df["high"] - df["low"]
            tr2 = (df["high"] - prev_close).abs()
            tr3 = (df["low"] - prev_close).abs()
            df["_tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr_14"] = df.groupby("symbol")["_tr"].transform(lambda s: s.rolling(14, min_periods=1).mean())
            df["atr_pct"] = df["atr_14"] / (df[price_col] + 1e-8)
            df.drop(columns=["_tr"], inplace=True)

        return df
