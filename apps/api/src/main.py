"""
FastAPI Backend Gateway: REST API for NSE Systematic 5-Minute Trading Platform.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trading.engine import SystematicTradingEngine
from trading.models.registry.model_registry import ModelRegistry
from database.stocks_config import STOCK_SYMBOLS, STOCK_UNIVERSE

app = FastAPI(
    title="NSE Systematic 5-Minute Trading API",
    description="High-performance algorithmic trading and 5-minute prediction engine with Indian taxation and sovereign risk gating.",
    version="2.0.0"
)

# Enable CORS for frontend web application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Trading Engine instance
trading_engine = SystematicTradingEngine()
model_registry = ModelRegistry()


@app.get("/health", tags=["System"])
def health_check():
    active_info = model_registry.get_active_model_info()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_model_version": active_info["version_id"] if active_info else "none",
        "is_kill_switch_active": trading_engine.risk_manager.kill_switch.is_active
    }


@app.get("/api/v1/stocks", tags=["Market Data"])
def get_tracked_stocks():
    return [
        {"symbol": s, "company": STOCK_UNIVERSE.get(s, s)}
        for s in STOCK_SYMBOLS
    ]


@app.get("/api/v1/predictions", tags=["Predictions"])
def get_latest_predictions():
    raw_df = trading_engine.collector.fetch_all(period="5d", interval="5m")
    if raw_df.empty:
        return {"predictions": [], "message": "No market data available"}

    feat_df = trading_engine.feature_pipeline.extract_features(raw_df)
    predictions = []
    for s in STOCK_SYMBOLS:
        sub = feat_df[feat_df["symbol"] == s].sort_values("timestamp")
        if len(sub) >= 50:
            pred = trading_engine.predictor.predict(s, sub.tail(50))
            pred["company"] = STOCK_UNIVERSE.get(s, s)
            predictions.append(pred)

    return {
        "count": len(predictions),
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions
    }


@app.get("/api/v1/portfolio", tags=["Portfolio"])
def get_portfolio_summary():
    return trading_engine.portfolio.get_state()


@app.get("/api/v1/orders", tags=["Orders"])
def get_order_history():
    return trading_engine.order_manager.get_order_history()


@app.post("/api/v1/engine/cycle", tags=["Trading Engine"])
def execute_trading_cycle():
    """Trigger one systematic 5-minute scan-and-trade cycle."""
    return trading_engine.execute_cycle()


@app.post("/api/v1/risk/kill-switch", tags=["Risk Management"])
def toggle_kill_switch(activate: bool, reason: str = "Operator Manual Intervention"):
    if activate:
        trading_engine.risk_manager.kill_switch.trigger(reason)
    else:
        trading_engine.risk_manager.kill_switch.reset()

    return {
        "kill_switch_active": trading_engine.risk_manager.kill_switch.is_active,
        "reason": trading_engine.risk_manager.kill_switch.trigger_reason,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/models", tags=["Model Registry"])
def list_model_versions():
    return model_registry.list_models()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
