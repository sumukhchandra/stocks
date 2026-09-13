"""
FastAPI Backend Gateway: REST API for NSE Systematic Real-Time Trading Platform.
Hosts heavy ML models (CatBoost, XGBoost, LightGBM, Regressors), Market Analyzer,
and Automated Trade Execution with API Key authentication.
"""

from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from datetime import datetime
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.stocks_config import STOCK_SYMBOLS, STOCK_UNIVERSE

# API Key Authentication Setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
DEFAULT_API_KEY = os.getenv("NSE_BACKEND_API_KEY", "nse_secret_alpha_2026")


def verify_api_key(api_key: str = Security(api_key_header)):
    """Verifies that incoming request provides a valid X-API-Key header."""
    if not api_key or api_key != DEFAULT_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-API-Key header. Provide valid key in X-API-Key header."
        )
    return api_key


app = FastAPI(
    title="NSE Systematic 5-Minute Trading API",
    description="High-performance algorithmic trading and 5-minute prediction engine with Indian taxation and sovereign risk gating.",
    version="2.1.0"
)

from fastapi.staticfiles import StaticFiles

# Enable CORS for frontend web and mobile client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Mobile Web App
mobile_dir = os.path.join(PROJECT_ROOT, "frontend", "mobile")
if os.path.exists(mobile_dir):
    app.mount("/mobile", StaticFiles(directory=mobile_dir, html=True), name="mobile")


# Lazy-loaded singletons to ensure fast server boot
_trading_engine = None
_model_registry = None


def get_engine():
    global _trading_engine
    if _trading_engine is None:
        from trading.engine import SystematicTradingEngine
        _trading_engine = SystematicTradingEngine()
    return _trading_engine


def get_registry():
    global _model_registry
    if _model_registry is None:
        from trading.models.registry.model_registry import ModelRegistry
        _model_registry = ModelRegistry()
    return _model_registry


# ─── Public Health Check ───────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Public health status check (no authentication required)."""
    reg = get_registry()
    eng = get_engine()
    active_info = reg.get_active_model_info()
    return {
        "status": "healthy",
        "service": "NSE Backend Trading Engine",
        "timestamp": datetime.now().isoformat(),
        "active_model_version": active_info["version_id"] if active_info else "none",
        "is_kill_switch_active": eng.risk_manager.kill_switch.is_active
    }


# ─── Auth Verification ────────────────────────────────────────────────────────
@app.post("/api/v1/auth/verify", tags=["Authentication"])
def verify_auth(api_key: str = Depends(verify_api_key)):
    """Verify that client credentials are valid."""
    return {
        "authenticated": True,
        "timestamp": datetime.now().isoformat(),
        "message": "API key valid. Connected to NSE Backend Engine."
    }


# ─── Market Analysis & Screener ───────────────────────────────────────────────
@app.get("/api/v1/market/overview", tags=["Market Analysis"])
def get_market_overview(api_key: str = Depends(verify_api_key)):
    """Fetch real-time market sentiment radar, advance/decline ratio, and sector heatmap."""
    from backend.market_data_feed import market_feed
    from backend.market_analyzer import market_analyzer
    quotes_df = market_feed.get_realtime_quotes()
    feature_df = market_feed.get_live_feature_dataset()
    return market_analyzer.analyze_market_overview(quotes_df, feature_df)


@app.get("/api/v1/market/screener", tags=["Market Analysis"])
def get_market_screener(api_key: str = Depends(verify_api_key)):
    """Fetch real-time 8-indicator technical screener matrix across all tracked stocks."""
    from backend.market_data_feed import market_feed
    from backend.market_analyzer import market_analyzer
    feature_df = market_feed.get_live_feature_dataset()
    quotes_df = market_feed.get_realtime_quotes()
    df = market_analyzer.compute_technical_screener(feature_df, quotes_df)
    return df.to_dict(orient="records")


@app.get("/api/v1/market/opportunities", tags=["Market Analysis"])
def get_market_opportunities(api_key: str = Depends(verify_api_key)):
    """Fetch algorithmic intraday volume breakout and oversold rebound setups."""
    from backend.market_data_feed import market_feed
    from backend.market_analyzer import market_analyzer
    feature_df = market_feed.get_live_feature_dataset()
    return market_analyzer.scan_breakout_opportunities(feature_df)


@app.get("/api/v1/market/quotes", tags=["Market Data"])
def get_market_quotes(symbols: str = "", api_key: str = Depends(verify_api_key)):
    """Fetch cached real-time quotes (<5s TTL) for active universe stocks."""
    from backend.market_data_feed import market_feed
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else STOCK_SYMBOLS
    df = market_feed.get_realtime_quotes(sym_list)
    return df.to_dict(orient="records") if hasattr(df, "to_dict") else df


@app.get("/api/v1/stocks", tags=["Market Data"])
def get_tracked_stocks(api_key: str = Depends(verify_api_key)):
    """List of all tracked NSE stocks and corresponding company names."""
    return [
        {"symbol": s, "company": STOCK_UNIVERSE.get(s, s)}
        for s in STOCK_SYMBOLS
    ]


# ─── Machine Learning Predictions ─────────────────────────────────────────────
@app.post("/api/v1/predictions/batch", tags=["Predictions"])
def get_batch_predictions(api_key: str = Depends(verify_api_key)):
    """Vectorized multi-stock ML evaluation across CatBoost, XGBoost, LightGBM, and Regressors."""
    from backend.market_data_feed import market_feed
    from ml_models.models.final_ensemble_engine import final_ensemble

    df_live = market_feed.get_live_feature_dataset()
    stock_windows_map = {}
    for s in STOCK_SYMBOLS:
        sub = df_live[df_live["symbol"] == s]
        if not sub.empty:
            stock_windows_map[s] = sub

    return final_ensemble.evaluate_batch(stock_windows_map)


@app.get("/api/v1/predictions", tags=["Predictions"])
def get_latest_predictions(api_key: str = Depends(verify_api_key)):
    """Legacy individual predictions endpoint."""
    eng = get_engine()
    raw_df = eng.collector.fetch_all(period="5d", interval="5m")
    if raw_df.empty:
        return {"predictions": [], "message": "No market data available"}

    feat_df = eng.feature_pipeline.extract_features(raw_df)
    predictions = []
    for s in STOCK_SYMBOLS:
        sub = feat_df[feat_df["symbol"] == s].sort_values("timestamp")
        if len(sub) >= 50:
            pred = eng.predictor.predict(s, sub.tail(50))
            pred["company"] = STOCK_UNIVERSE.get(s, s)
            predictions.append(pred)

    return {
        "count": len(predictions),
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions
    }


# ─── Portfolio & Execution ───────────────────────────────────────────────────
@app.get("/api/v1/portfolio", tags=["Portfolio"])
def get_portfolio_summary(api_key: str = Depends(verify_api_key)):
    """Get portfolio state, balances, and active positions."""
    eng = get_engine()
    return eng.portfolio.get_state()


@app.get("/api/v1/orders", tags=["Orders"])
def get_order_history(api_key: str = Depends(verify_api_key)):
    """Get completed and pending order histories."""
    eng = get_engine()
    return eng.order_manager.get_order_history()


@app.post("/api/v1/engine/cycle", tags=["Trading Engine"])
def execute_trading_cycle(api_key: str = Depends(verify_api_key)):
    """Trigger one systematic 5-minute scan-and-trade cycle."""
    eng = get_engine()
    return eng.execute_cycle()


@app.post("/api/v1/risk/kill-switch", tags=["Risk Management"])
def toggle_kill_switch(activate: bool, reason: str = "Operator Manual Intervention", api_key: str = Depends(verify_api_key)):
    """Emergency trading halt or resumption."""
    eng = get_engine()
    if activate:
        eng.risk_manager.kill_switch.trigger(reason)
    else:
        eng.risk_manager.kill_switch.reset()

    return {
        "kill_switch_active": eng.risk_manager.kill_switch.is_active,
        "reason": eng.risk_manager.kill_switch.trigger_reason,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/models", tags=["Model Registry"])
def list_model_versions(api_key: str = Depends(verify_api_key)):
    """List available and active machine learning model checkpoints."""
    reg = get_registry()
    return reg.list_models()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
