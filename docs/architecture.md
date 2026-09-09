# 🏛️ System Architecture: NSE 5-Minute Algorithmic Trading Platform

## 1. System Overview

This system is a **short-horizon systematic trading platform** operating on 5-minute candles on the National Stock Exchange of India (NSE). It combines statistical learning, machine learning ensemble models, sovereign risk management, and multi-broker execution abstractions.

```
+-----------------------------------------------------------------------------------+
|                                  OPERATIONS LAYER                                 |
|   +---------------------------------------+   +-------------------------------+   |
|   |         Web/Mobile Dashboard          |   |       FastAPI Gateway         |   |
|   |   (Observes, monitors, manual audit)  |   |  (REST + WebSockets feeds)    |   |
|   +---------------------------------------+   +---------------+---------------+   |
+---------------------------------------------------------------|-------------------+
                                                                |
+---------------------------------------------------------------v-------------------+
|                                  TRADING CORE ENGINE                              |
|                                                                                   |
|  [Market Feed] ---> [Data Cleaner] ---> [Feature Pipeline] ---> [ML Predictor]    |
|                                                                        |          |
|                                                                        v          |
|  [Execution Broker] <--- [Order Manager] <--- [Risk Engine] <--- [Signal Gen]     |
|   (Paper / Live)        (Frictions, Sizing)  (Kill Switch, Veto)  (5m Horizon)    |
|                               |                                                   |
|                               v                                                   |
|                      [Portfolio & Ledger]                                         |
+-----------------------------------------------------------------------------------+
                                |
+-------------------------------v---------------------------------------------------+
|                                  PERSISTENCE LAYER                                |
|   +-------------------+   +--------------------+   +--------------------------+   |
|   |  Database (SQLite |   |   Model Registry   |   | Protected Parquet Caches |   |
|   |   / PostgreSQL)   |   |  (model_version)   |   |     (60d 5m Candles)     |   |
|   +-------------------+   +--------------------+   +--------------------------+   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Invariants

1. **Frontend-Execution Decoupling**: The user interface never executes trades directly. It issues instructions to the backend API or queries current state.
2. **Sovereign Risk Authority**: The Risk Engine has absolute veto authority over the ML model. No order is dispatched without risk approval.
3. **Model Versioning Traceability**: Every single prediction, signal, order, and trade execution MUST be tagged with an explicit `model_version_id`.
4. **Friction-Aware Gating**: Intraday trading at a 5-minute horizon is heavily influenced by costs. Orders must be gated on *net* profit viability after Zerodha brokerage, STT, GST, exchange charges, stamp duty, and 25% STCG tax.
5. **Fail-Safe Operation**: If market data is stale (>90 seconds old during trading hours), the core loop automatically enters a HOLD state and refuses to trade.
