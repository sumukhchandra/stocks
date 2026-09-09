# 🗄️ Database Architecture & Schema Specification

## 1. Overview

The persistence layer stores operational state, audit trails, historical market candles, predictions, and trades. To guarantee institutional compliance and historical explainability, all signals and executions link directly to an explicit `model_version`.

---

## 2. Core Relational Tables

### 1. `model_versions`
Tracks every trained model bundle deployed to production:
- `id`: Unique version string (e.g., `ens_v2.5_20260908`)
- `algorithm_name`: String (e.g., `Ensemble_XGB_LGB_CatBoost_RF`)
- `feature_set_version`: String
- `trained_at`: Timestamp
- `metrics_json`: JSON string of training/validation metrics
- `artifact_path`: File path to serialized model weights
- `is_active`: Boolean flag indicating current live inference model

### 2. `predictions`
Stores ex-ante predictions and ex-post reality comparisons:
- `id`: Integer primary key
- `timestamp`: Bar timestamp
- `symbol`: NSE ticker symbol
- `horizon_minutes`: 5
- `current_price`: Close price at prediction time
- `predicted_price`: Expected price 5 minutes forward
- `predicted_return`: Float decimal return
- `confidence`: Probability estimate $[0.0, 1.0]$
- `model_version_id`: Foreign key $\rightarrow$ `model_versions.id`
- `actual_price_after`: Price observed ex-post (populated 5m later)
- `error_pct`: Realized prediction error percentage

### 3. `signals`
Stores actionable recommendations emitted by the strategy:
- `id`: Integer primary key
- `prediction_id`: Foreign key $\rightarrow$ `predictions.id`
- `symbol`: NSE ticker
- `action`: `BUY`, `SELL`, or `HOLD`
- `confidence`: Confidence score
- `expected_return`: Predicted return
- `status`: `PENDING`, `APPROVED`, or `REJECTED`

### 4. `orders`
Immutable record of all broker order dispatches:
- `id`: Broker order ID or UUID
- `symbol`: NSE ticker
- `side`: `BUY` or `SELL`
- `order_type`: `MARKET` or `LIMIT`
- `quantity`: Executed quantity
- `requested_price`: Price at order submission
- `executed_price`: Actual fill price
- `status`: `PENDING`, `FILLED`, `CANCELLED`, `REJECTED`
- `broker_name`: `PAPER` or `ZERODHA`
- `brokerage`, `stt`, `gst`, `exchange_charges`, `stamp_duty`, `total_friction`
- `model_version_id`: Foreign key $\rightarrow$ `model_versions.id`
