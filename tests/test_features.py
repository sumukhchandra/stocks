"""
Unit Tests for Feature Extraction Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading.features.feature_pipeline import FeaturePipeline


def test_feature_pipeline_extraction():
    # Build sample 5-minute candle dataframe
    base_time = datetime(2026, 9, 8, 9, 15)
    rows = []
    for i in range(60):
        t = base_time + timedelta(minutes=5 * i)
        price = 1000.0 + np.sin(i / 5.0) * 10.0 + i * 0.5
        rows.append({
            "timestamp": t,
            "symbol": "RELIANCE.NS",
            "open": price - 1.0,
            "high": price + 2.0,
            "low": price - 2.0,
            "close": price,
            "volume": 10000 + i * 100
        })

    df = pd.DataFrame(rows)
    pipeline = FeaturePipeline()
    featured = pipeline.extract_features(df)

    assert not featured.empty
    assert "sma_9" in featured.columns
    assert "ema_21" in featured.columns
    assert "rsi_14" in featured.columns
    assert "macd" in featured.columns
    assert "bb_pct_b" in featured.columns
    assert "atr_14" in featured.columns
    assert "regime_trend" in featured.columns
    assert "time_sin" in featured.columns
    assert "time_cos" in featured.columns
