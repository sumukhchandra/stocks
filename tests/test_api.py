"""
Integration Tests for FastAPI Endpoints.
"""

from fastapi.testclient import TestClient
from apps.api.src.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_model_version" in data
    assert data["is_kill_switch_active"] is False


def test_stocks_endpoint():
    response = client.get("/api/v1/stocks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert any(s["symbol"] == "RELIANCE.NS" for s in data)


def test_portfolio_endpoint():
    response = client.get("/api/v1/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert "total_capital" in data
    assert "available_capital" in data
    assert "positions" in data


def test_kill_switch_toggle():
    # Activate
    res = client.post("/api/v1/risk/kill-switch?activate=true&reason=Test%20Halt")
    assert res.status_code == 200
    assert res.json()["kill_switch_active"] is True

    # Check health reflects it
    h_res = client.get("/health")
    assert h_res.json()["is_kill_switch_active"] is True

    # Deactivate
    res2 = client.post("/api/v1/risk/kill-switch?activate=false")
    assert res2.status_code == 200
    assert res2.json()["kill_switch_active"] is False
