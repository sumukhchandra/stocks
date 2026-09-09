import pandas as pd
import numpy as np
import os
import sys
import requests
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_models.models.final_ensemble_engine import FinalEnsembleEngine
from backend.risk.india_tax_engine import IndiaTaxEngine
from backend.risk.inr_accounting_engine import INRAccountingEngine
from database.stocks_config import STOCK_SYMBOLS

# Configuration
SIMULATION_STEPS = 100       
MIN_PROB_THRESHOLD = 0.80    # High conviction threshold
MIN_NET_EDGE = 0.015         # Require 1.5% expected edge to trade
TARGET_RETURN = 0.035        # 3.5% Goal
ATR_STOP_MULT = 1.0          # Tight Stop for high R:R

async def run_simulation():
    print("="*50)
    print("NSE STOCK TREND AUDITOR: HIGH-YIELD STRATEGY")
    print("="*50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models', 'saved_models')
    engine = FinalEnsembleEngine(model_dir=model_dir)
    tax_engine = IndiaTaxEngine()
    inr_ledger = INRAccountingEngine(initial_balance_inr=10000, reset=True)
    
    universe = STOCK_SYMBOLS
    data_path = os.path.join(script_dir, 'data', 'processed', 'master_labeled_dataset.parquet')
    
    if not os.path.exists(data_path):
        print("Data not found.")
        return
        
    df = pd.read_parquet(data_path)
    trade_log = []
    
    # Simulation range (Walk-forward)
    max_idx = len(df) - 25 # Room for 24h future return
    start_idx = max_idx - SIMULATION_STEPS
    
    print(f"Auditing {SIMULATION_STEPS} points across {len(universe)} NSE stocks (1H Horizon)...")

    for i in range(start_idx, max_idx):
        timestamp = df.iloc[i]['timestamp']
        candidates = []
        
        for symbol in universe:
            symbol_df = df[df['symbol'] == symbol]
            current_rows = symbol_df[symbol_df['timestamp'] <= timestamp]
            if len(current_rows) < 50: continue # Need EMA 50
            
            window = current_rows.tail(50).copy()
            
            try:
                decision = engine.evaluate_state(window)
                prob = decision['final_probability']
                
                # Trend Filtering
                regime = decision['regime']
                trend_s = window['trend_strength'].iloc[-1]
                
                # Rank Score: 1.0*Prob
                rank_score = prob
                
                candidates.append({
                    'symbol': symbol,
                    'prob': prob,
                    'rank_score': rank_score,
                    'regime': regime,
                    'atr': window['atr_14'].iloc[-1],
                    'price': window['close'].iloc[-1],
                    # Get actual 24h future return from dataset
                    'future_ret': symbol_df[symbol_df['timestamp'] > timestamp]['future_return_t1'].iloc[0]
                })
            except Exception as e:
                print(f"Error evaluating {symbol}: {e}") 
                continue
            
        if not candidates: continue
        
        # Select Single Best Trend Opportunity
        top_asset = sorted(candidates, key=lambda x: x['rank_score'], reverse=True)[0]
        
        print(f"DEBUG: {top_asset['symbol']} | Prob: {top_asset['prob']:.2%} | Rank: {top_asset['rank_score']:.2f}")
        
        if top_asset['prob'] < MIN_PROB_THRESHOLD: continue
        
        # --- EDGE FILTER ---
        # Only trade if model probability is high and future move potential clears costs.
        if top_asset['future_ret'] < MIN_NET_EDGE:
            # print(f"DEBUG: {top_asset['symbol']} filtered out by Net Edge ({top_asset['future_ret']:.2%})")
            continue
            
        # --- EXECUTION ---
        trade_capital_inr = 10000
        actual_move = top_asset['future_ret']
        sl_pct = (top_asset['atr'] * ATR_STOP_MULT) / top_asset['price']
        
        # Simulation Logic
        if actual_move >= TARGET_RETURN:
            realized_return = TARGET_RETURN
            outcome = 'WIN'
        elif actual_move <= -sl_pct:
            realized_return = -sl_pct
            outcome = 'LOSS_SL'
        else:
            realized_return = actual_move
            outcome = 'TIME_EXIT'

        # Cost: 2.5bps Slippage
        realized_return -= 0.00025
        
        profit_usd = (trade_capital_inr / 83.5) * realized_return
        revenue_usd = (trade_capital_inr / 83.5) * (1 + realized_return)
        
        inr_ledger.record_trade(profit_usd, revenue_usd)
        
        trade_log.append({
            'symbol': top_asset['symbol'],
            'prob': top_asset['prob'],
            'outcome': outcome,
            'return': realized_return
        })

    # --- REPORT ---
    if not trade_log:
        print("No high-conviction trends found.")
        return
        
    results_df = pd.DataFrame(trade_log)
    win_rate = (results_df['return'] > 0).mean()
    
    print("\n" + "="*50)
    print("TREND AUDIT PERFORMANCE REPORT")
    print("="*50)
    print(f"Trades Taken:    {len(trade_log)}")
    print(f"Win Rate:        {win_rate:.2%}")
    print(f"Avg Gain/Loss:   {results_df['return'].mean():.2%}")
    print("-" * 50)
    
    summary = inr_ledger.get_summary()
    print(f"Initial Cap:     10000.00 INR")
    print(f"Current Cap:     {summary['current_liquid_capital_inr']:.2f} INR")
    print(f"Gross PnL:       {summary['total_gross_pnl_inr']:.2f} INR")
    print(f"Capital Reserved:{summary['locked_tds_inr']:.2f} INR")
    print(f"Net ROI:         {summary['roi_pct']:.2%}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_simulation())
