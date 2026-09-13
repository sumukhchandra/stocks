"""
Integration Tests for FastAPI Endpoints with API Key Security.
"""

from fastapi.testclient import TestClient
from apps.api.src.main import app

client = TestClient(app)
VALID_API_KEY = "nse_secret_alpha_2026"
AUTH_HEADERS = {"X-API-Key": VALID_API_KEY}


def test_health_endpoint():
    """Health check must be publicly accessible without API key."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_model_version" in data
    assert data["is_kill_switch_active"] is False


def test_unauthorized_access():
    """Protected endpoints must reject requests without valid X-API-Key."""
    # No header
    res1 = client.get("/api/v1/stocks")
    assert res1.status_code == 401

    # Wrong key
    res2 = client.get("/api/v1/stocks", headers={"X-API-Key": "wrong_key_123"})
    assert res2.status_code == 401


def test_auth_verify():
    """Verify endpoint must validate correct key."""
    res = client.post("/api/v1/auth/verify", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["authenticated"] is True


def test_stocks_endpoint():
    response = client.get("/api/v1/stocks", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert any(s["symbol"] == "RELIANCE.NS" for s in data)


def test_portfolio_endpoint():
    response = client.get("/api/v1/portfolio", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "total_capital" in data
    assert "available_capital" in data
    assert "positions" in data


def test_market_overview_endpoint():
    response = client.get("/api/v1/market/overview", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment_score" in data
    assert "adv_dec_ratio" in data
    assert "sector_performance" in data


def test_market_screener_endpoint():
    response = client.get("/api/v1/market/screener", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "rsi_14" in data[0]
    assert "supertrend" in data[0]
    assert "action" in data[0]


def test_market_opportunities_endpoint():
    response = client.get("/api/v1/market/opportunities", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_kill_switch_toggle():
    # Activate
    res = client.post("/api/v1/risk/kill-switch?activate=true&reason=Test%20Halt", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["kill_switch_active"] is True

    # Check health reflects it
    h_res = client.get("/health")
    assert h_res.json()["is_kill_switch_active"] is True

    # Deactivate
    res2 = client.post("/api/v1/risk/kill-switch?activate=false", headers=AUTH_HEADERS)
    assert res2.status_code == 200
    assert res2.json()["kill_switch_active"] is False
