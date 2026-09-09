# 🛡️ Risk Management & Sovereign Guardrails

## 1. Principles of Sovereign Risk Management

In production trading systems, the **Risk Engine is sovereign**: it sits downstream of the strategy and machine learning models and holds unappealable veto power over every proposed order.

```
Model Prediction ---> Strategy Signal Generator ---> [ SOVEREIGN RISK ENGINE ] ---> Order Dispatch
                                                            |
                                        +-------------------+-------------------+
                                        |                   |                   |
                                   [Approved]          [Rejected]        [Kill Switch]
                                        |                   |                   |
                                        v                   v                   v
                                   Order Router        Logged in DB       Halt Engine
```

---

## 2. Hard Risk Constraints

| Risk Gate | Threshold / Rule | Action on Breach |
| :--- | :--- | :--- |
| **Market Hours** | 9:15 AM to 3:30 PM IST | Block entries outside market hours |
| **EOD Square-Off** | 3:15 PM IST | Force market exit of all intraday positions |
| **Max Portfolio Drawdown** | 3.0% in single trading session | **Kill Switch Triggered**: Halt all trading for day |
| **Max Single Position Loss** | 0.8% gross loss | **Stop Loss**: Immediate market order exit |
| **Max Concurrent Positions** | 5 simultaneous symbols | Block new entries until an existing position closes |
| **Sector Concentration** | Max 40% capital in one sector | Reject new entries exceeding sector threshold |
| **Trade Net Viability** | Predicted net profit > 0.05% after all Zerodha fees & 25% tax | Reject trade as non-viable |
| **Data Staleness** | Latest candle > 90 seconds old | Reject trade; log stale market data warning |

---

## 3. Emergency Kill Switch Protocols

The Kill Switch can be triggered through two channels:
1. **Automated Trigger**: Activated when cumulative daily drawdown exceeds the circuit-breaker threshold (3%).
2. **Manual Operator Trigger**: Activated via the Web Dashboard or API endpoint `/api/v1/risk/kill-switch`.

When triggered:
- All pending limit/stop orders are immediately cancelled.
- All open positions are liquidated with market orders.
- The engine enters `STATE_HALTED` until an operator manually performs a reset.
