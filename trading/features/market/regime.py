"""
Market Regime and Trend Classification.
"""

import pandas as pd
import numpy as np


class MarketRegimeFeatures:
    @staticmethod
    def compute(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        df = df.copy()

        # Returns over short and medium horizons
        df["ret_1"] = df.groupby("symbol")[price_col].pct_change(1).fillna(0)
        df["ret_3"] = df.groupby("symbol")[price_col].pct_change(3).fillna(0)
        df["ret_6"] = df.groupby("symbol")[price_col].pct_change(6).fillna(0)
        df["ret_12"] = df.groupby("symbol")[price_col].pct_change(12).fillna(0)

        # Intraday regime classification
        sma_20 = df.groupby("symbol")[price_col].transform(lambda s: s.rolling(20, min_periods=1).mean())
        sma_50 = df.groupby("symbol")[price_col].transform(lambda s: s.rolling(50, min_periods=1).mean())

        # Regime indicator: 1 = Trending Up, -1 = Trending Down, 0 = Sideways / Volatile
        df["regime_trend"] = np.where(sma_20 > sma_50, 1, np.where(sma_20 < sma_50, -1, 0))
        return df
