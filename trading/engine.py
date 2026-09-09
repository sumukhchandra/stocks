"""
Core Systematic 5-Minute Trading Engine.
Orchestrates: Market Data -> Processing -> Feature Pipeline -> ML Predictor -> Signal Generator -> Risk Engine -> Order Manager -> Broker.
"""

from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

from .data.collectors.market_data import MarketDataCollector
from .data.processors.validator import DataValidator
from .features.feature_pipeline import FeaturePipeline
from .models.inference.predictor import FiveMinutePredictor
from .strategy.signal_generator import SignalGenerator
from .strategy.position_sizing import PositionSizer
from .risk.risk_manager import RiskManager
from .execution.order_manager import OrderManager
from .execution.brokers.paper import PaperBroker
from .portfolio.portfolio_manager import PortfolioManager


class SystematicTradingEngine:
    def __init__(self, initial_capital: float = 100000.0, min_confidence: float = 0.60):
        self.collector = MarketDataCollector()
        self.feature_pipeline = FeaturePipeline()
        self.predictor = FiveMinutePredictor()
        self.signal_gen = SignalGenerator(min_confidence=min_confidence)
        self.risk_manager = RiskManager()
        self.broker = PaperBroker(initial_capital=initial_capital)
        self.order_manager = OrderManager(broker=self.broker)
        self.portfolio = PortfolioManager(starting_capital=initial_capital)

    def execute_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete systematic cycle across all configured stocks.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "predictions": [],
            "signals": [],
            "executed_orders": [],
            "rejected_signals": []
        }

        # 1. Ingest Market Data
        raw_df = self.collector.fetch_all(period="5d", interval="5m")
        if raw_df.empty:
            results["status"] = "NO_DATA"
            return results

        # 2. Extract Features
        featured_df = self.feature_pipeline.extract_features(raw_df)
        portfolio_state = self.portfolio.get_state()

        # 3. Predict & Trade for Each Stock
        for symbol in self.collector.symbols:
            sub = featured_df[featured_df["symbol"] == symbol].sort_values("timestamp")
            if len(sub) < 50:
                continue

            window = sub.tail(50)
            latest_ts = window["timestamp"].iloc[-1]

            # Model Inference (Tagged with model_version)
            pred = self.predictor.predict(symbol, window)
            results["predictions"].append(pred)

            # Signal Generation
            signal = self.signal_gen.generate_signal(pred)
            results["signals"].append(signal)

            if signal["action"] == "BUY":
                # Sovereign Risk Audit
                approved, reason, risk_meta = self.risk_manager.audit_signal(
                    signal=signal,
                    portfolio_state=portfolio_state,
                    latest_timestamp=latest_ts
                )

                if approved:
                    alloc = PositionSizer.calculate_allocation(
                        portfolio_state["available_capital"],
                        signal["confidence"]
                    )
                    if alloc > 0:
                        qty = alloc / signal["current_price"]
                        order = self.order_manager.submit_order(
                            symbol=symbol,
                            side="BUY",
                            quantity=qty,
                            price=signal["current_price"],
                            model_version=pred["model_version"]
                        )
                        self.portfolio.record_entry(
                            symbol=symbol,
                            qty=qty,
                            price=order["executed_price"],
                            invested=order["turnover"],
                            model_version=pred["model_version"]
                        )
                        results["executed_orders"].append(order)
                else:
                    results["rejected_signals"].append({
                        "symbol": symbol,
                        "reason": reason
                    })

        results["status"] = "OK"
        results["portfolio_summary"] = self.portfolio.get_state()
        return results
