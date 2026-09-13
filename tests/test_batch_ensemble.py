"""
Unit Tests for FinalEnsembleEngine Vectorized Batch Inference.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ml_models.models.final_ensemble_engine import FinalEnsembleEngine
from database.stocks_config import STOCK_SYMBOLS


def test_final_ensemble_batch_evaluation():
    engine = FinalEnsembleEngine()
    assert engine.universal_models
    assert engine.return_models
    
    # Generate mock stock windows
    base_time = datetime.now() - timedelta(hours=4)
    stock_windows = {}
    
    for s in STOCK_SYMBOLS[:4]:
        rows = []
        p = 1000.0
        for i in range(25):
            t = base_time + timedelta(minutes=5 * i)
            p += np.sin(i / 2.0) * 4.0
            rows.append({
                "timestamp": t,
                "symbol": s,
                "open": p - 1.0,
                "high": p + 3.0,
                "low": p - 2.0,
                "close": p,
                "volume": 50000,
                "returns": 0.001,
                "volatility": 0.015,
                "regime": "sideways",
                "last_macro_severity": 0.0,
                "macro_risk_factor": 0.0,
            })
        stock_windows[s] = pd.DataFrame(rows)
        
    results = engine.evaluate_batch(stock_windows)
    assert len(results) == 4
    for s in STOCK_SYMBOLS[:4]:
        assert s in results
        res = results[s]
        assert "final_probability" in res
        assert 0.0 <= res["final_probability"] <= 1.0
        assert "expected_return" in res
        assert "confidence_score" in res
        assert "signals" in res
        assert "breakdown" in res["signals"]
