"""
Data Cleaner: Outlier detection, zero-volume handling, and candle deduplication.
"""

import pandas as pd
import numpy as np


class DataCleaner:
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        # Drop duplicates by timestamp and symbol
        if "symbol" in df.columns:
            df = df.drop_duplicates(subset=["timestamp", "symbol"]).reset_index(drop=True)
        else:
            df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # Ensure positive non-zero prices
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df = df[df[col] > 0]

        # Ensure logical candle high >= low, high >= open, high >= close
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            valid_mask = (df["high"] >= df["low"]) & (df["high"] >= df["open"]) & (df["high"] >= df["close"])
            df = df[valid_mask]

        # Fill missing volume with 0
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).clip(lower=0)

        return df.sort_values("timestamp").reset_index(drop=True)
