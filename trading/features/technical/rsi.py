"""
RSI (Relative Strength Index) technical indicator.
"""

import pandas as pd
import numpy as np


class RSIFeatures:
    @staticmethod
    def _compute_single_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @classmethod
    def compute(cls, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        df = df.copy()
        for p in [7, 14, 21]:
            df[f"rsi_{p}"] = df.groupby("symbol")[price_col].transform(
                lambda s: cls._compute_single_rsi(s, period=p)
            )
        return df
