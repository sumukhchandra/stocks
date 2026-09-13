"""
Unit Tests for FastMarketDataFeed and MarketAnalyzer.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.market_data_feed import FastMarketDataFeed, market_feed
from backend.market_analyzer import MarketAnalyzer, market_analyzer
from database.stocks_config import STOCK_SYMBOLS


def test_market_feed_singleton():
    f1 = FastMarketDataFeed()
    f2 = FastMarketDataFeed()
    assert f1 is f2
    assert f1.symbols == list(STOCK_SYMBOLS)


def test_market_feed_quotes_fallback():
    feed = FastMarketDataFeed()
    quotes = feed.get_realtime_quotes(["RELIANCE.NS", "TCS.NS"])
    assert not quotes.empty
    assert "symbol" in quotes.columns
    assert "price" in quotes.columns
    assert "change_pct" in quotes.columns
    assert any(quotes["symbol"] == "RELIANCE.NS")


def test_market_analyzer_overview():
    analyzer = MarketAnalyzer()
    
    mock_quotes = pd.DataFrame([
        {"symbol": "RELIANCE.NS", "change_pct": 1.25, "price": 1250.0, "volume": 100000},
        {"symbol": "TCS.NS", "change_pct": 0.85, "price": 3200.0, "volume": 50000},
        {"symbol": "HDFCBANK.NS", "change_pct": -0.45, "price": 700.0, "volume": 80000},
        {"symbol": "INFY.NS", "change_pct": -1.10, "price": 1050.0, "volume": 60000},
    ])
    
    overview = analyzer.analyze_market_overview(mock_quotes)
    assert overview["advance_count"] == 2
    assert overview["decline_count"] == 2
    assert overview["adv_dec_ratio"] == 1.0
    assert "sector_performance" in overview
    assert overview["top_gainer"]["symbol"] == "RELIANCE.NS"
    assert overview["top_loser"]["symbol"] == "INFY.NS"


def test_technical_screener_computation():
    analyzer = MarketAnalyzer()
    
    # Generate mock 5m candles
    base_time = datetime.now() - timedelta(hours=5)
    rows = []
    for s in ["RELIANCE.NS", "TCS.NS"]:
        p = 1000.0 if s == "RELIANCE.NS" else 3000.0
        for i in range(35):
            t = base_time + timedelta(minutes=5 * i)
            p += np.sin(i / 3.0) * 5.0 + 0.5
            rows.append({
                "timestamp": t,
                "symbol": s,
                "open": p - 2.0,
                "high": p + 4.0,
                "low": p - 3.0,
                "close": p,
                "volume": 10000 + i * 500,
                "returns": 0.001,
                "rsi_14": 55.0 + i * 0.2,
                "atr_14": 10.0,
                "vwap": p - 1.0,
                "bb_pct_b": 0.65,
                "regime": "trending_bull",
            })
            
    df_feat = pd.DataFrame(rows)
    screener_df = analyzer.compute_technical_screener(df_feat)
    
    assert not screener_df.empty
    assert len(screener_df) == 2
    assert "rsi_14" in screener_df.columns
    assert "supertrend" in screener_df.columns
    assert "action" in screener_df.columns
    assert "action_badge" in screener_df.columns
