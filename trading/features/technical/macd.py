"""
MACD (Moving Average Convergence Divergence) Indicator.
"""

import pandas as pd


class MACDFeatures:
    @staticmethod
    def compute(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        df = df.copy()

        def _calc_macd(series: pd.Series):
            ema12 = series.ewm(span=12, adjust=False).mean()
            ema26 = series.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_hist": hist})

        res = df.groupby("symbol")[price_col].apply(_calc_macd).reset_index(level=0, drop=True)
        df["macd"] = res["macd"]
        df["macd_signal"] = res["macd_signal"]
        df["macd_hist"] = res["macd_hist"]
        return df
