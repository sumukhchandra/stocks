"""
Frontend API Client Bridge.
Communicates with the FastAPI Backend Engine over HTTP using X-API-Key header.
Provides seamless fallback to direct local Python modules if the backend URL is unreachable.
"""
import os
import requests
import pandas as pd
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("BackendAPIClient")

DEFAULT_API_URL = os.getenv("NSE_BACKEND_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("NSE_BACKEND_API_KEY", "nse_secret_alpha_2026")


class BackendAPIClient:
    """Client for querying the remote/local NSE Backend API."""

    def __init__(self, base_url: str = DEFAULT_API_URL, api_key: str = DEFAULT_API_KEY, timeout: int = 8):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._connected = None

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def verify_connection(self) -> Dict[str, Any]:
        """Check connection and authentication status with the backend."""
        try:
            url = f"{self.base_url}/api/v1/auth/verify"
            resp = requests.post(url, headers=self.headers, timeout=3)
            if resp.status_code == 200:
                self._connected = True
                return {"connected": True, "authenticated": True, "message": "Connected to Backend API"}
            elif resp.status_code == 401:
                self._connected = False
                return {"connected": True, "authenticated": False, "message": "Invalid API Key"}
            else:
                self._connected = False
                return {"connected": False, "authenticated": False, "message": f"HTTP {resp.status_code}"}
        except Exception as e:
            self._connected = False
            return {"connected": False, "authenticated": False, "message": f"Offline/Unreachable: {e}"}

    def is_online(self) -> bool:
        """Quick boolean check if the remote API is reachable and valid."""
        res = self.verify_connection()
        return res["connected"] and res["authenticated"]

    def get_market_overview(self) -> Dict[str, Any]:
        """Fetch market overview from API with fallback to local analyzer."""
        if self._connected is not False:
            try:
                url = f"{self.base_url}/api/v1/market/overview"
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API get_market_overview error: {e}. Using local fallback.")

        # Local Fallback
        from backend.market_analyzer import market_analyzer
        return market_analyzer.get_market_overview()

    def get_technical_screener(self) -> pd.DataFrame:
        """Fetch technical screener matrix from API with fallback to local analyzer."""
        if self._connected is not False:
            try:
                url = f"{self.base_url}/api/v1/market/screener"
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    return pd.DataFrame(data)
            except Exception as e:
                logger.warning(f"API get_technical_screener error: {e}. Using local fallback.")

        # Local Fallback
        from backend.market_analyzer import market_analyzer
        return market_analyzer.get_technical_screener_matrix()

    def get_opportunities(self) -> List[Dict[str, Any]]:
        """Fetch breakout and oversold opportunities from API with fallback."""
        if self._connected is not False:
            try:
                url = f"{self.base_url}/api/v1/market/opportunities"
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API get_opportunities error: {e}. Using local fallback.")

        # Local Fallback
        from backend.market_analyzer import market_analyzer
        return market_analyzer.get_intraday_opportunities()

    def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch cached realtime quotes from API with fallback."""
        if self._connected is not False:
            try:
                sym_str = ",".join(symbols)
                url = f"{self.base_url}/api/v1/market/quotes?symbols={sym_str}"
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API get_realtime_quotes error: {e}. Using local fallback.")

        # Local Fallback
        from backend.market_data_feed import market_feed
        return market_feed.get_realtime_quotes(symbols)

    def get_batch_predictions(self) -> Dict[str, Any]:
        """Fetch batch ML model predictions from API with fallback."""
        if self._connected is not False:
            try:
                url = f"{self.base_url}/api/v1/predictions/batch"
                resp = requests.post(url, headers=self.headers, timeout=12)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API get_batch_predictions error: {e}. Using local fallback.")

        # Local Fallback
        from backend.market_data_feed import market_feed
        from ml_models.models.final_ensemble_engine import final_ensemble
        from database.stocks_config import STOCK_SYMBOLS
        df_live = market_feed.get_live_feature_dataset()
        stock_windows_map = {s: df_live[df_live["symbol"] == s] for s in STOCK_SYMBOLS if not df_live[df_live["symbol"] == s].empty}
        return final_ensemble.evaluate_batch(stock_windows_map)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Fetch portfolio summary from API with fallback."""
        if self._connected is not False:
            try:
                url = f"{self.base_url}/api/v1/portfolio"
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API get_portfolio_summary error: {e}. Using local fallback.")

        # Local Fallback
        from backend.nse_auto_trader import nse_trader
        return nse_trader.get_portfolio_summary()

    def execute_trading_cycle(self) -> Dict[str, Any]:
        """Trigger one systematic 5-minute trade cycle from API with fallback."""
        if self._connected is not False:
            try:
                url = f"{self.base_url}/api/v1/engine/cycle"
                resp = requests.post(url, headers=self.headers, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API execute_trading_cycle error: {e}. Using local fallback.")

        # Local Fallback
        from backend.nse_auto_trader import nse_trader
        return nse_trader.run_single_cycle()


# Singleton default client
api_client = BackendAPIClient()
