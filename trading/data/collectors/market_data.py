"""
Market Data Collector: Ingests 5-minute OHLCV candles from NSE and index benchmarks.
"""

import os
import pandas as pd
import yfinance as yf
from typing import List, Optional
from datetime import datetime

DEFAULT_SYMBOLS = [
    "RELIANCE.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "TATACAP.NS", "INFY.NS", "TCS.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS"
]

INDEX_SYMBOL = "^NSEI"


class MarketDataCollector:
    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or DEFAULT_SYMBOLS

    def fetch_ohlcv(self, symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        """Fetch raw candles for a single ticker."""
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False
            )
            if df.empty:
                return pd.DataFrame()

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]

            df = df.reset_index()
            time_col = next((c for c in ["datetime", "date", "index", "timestamp"] if c in df.columns), None)
            if not time_col:
                return pd.DataFrame()

            df = df.rename(columns={time_col: "timestamp", "adj close": "adj_close"})
            required = ["timestamp", "open", "high", "low", "close", "volume"]
            if any(c not in df.columns for c in required):
                return pd.DataFrame()

            df = df[required].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df["symbol"] = symbol
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def fetch_all(self, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        """Fetch candles for all configured stocks."""
        frames = []
        for sym in self.symbols:
            sub = self.fetch_ohlcv(sym, period=period, interval=interval)
            if not sub.empty:
                frames.append(sub)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fetch_benchmark(self, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        """Fetch Nifty 50 benchmark index data."""
        return self.fetch_ohlcv(INDEX_SYMBOL, period=period, interval=interval)
