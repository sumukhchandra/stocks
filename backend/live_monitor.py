import pandas as pd
import numpy as np
import os
import sys
import requests
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.final_ensemble_engine import FinalEnsembleEngine
from features.technical import calculate_rsi, calculate_vwap, calculate_atr

def fetch_recent_klines(symbol, interval='1h', limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']:
            df[col] = df[col].astype(float)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['symbol'] = symbol
        return df
    except:
        return None

def build_live_features(df):
    group = df.copy().sort_values('timestamp')
    # Indicators
    group['rsi_14'] = calculate_rsi(group, period=14, price_col='close')
    group['vwap'] = calculate_vwap(group, price_col='close', volume_col='volume')
    group['vwap_dist'] = (group['close'] - group['vwap']) / group['vwap']
    group['atr_14'] = calculate_atr(group, period=14)
    group['ema_20'] = group['close'].ewm(span=20).mean()
    group['ema_50'] = group['close'].ewm(span=50).mean()
    group['trend_strength'] = (group['ema_20'] - group['ema_50']) / group['ema_50']
    group['volatility'] = group['close'].pct_change().rolling(24).std()
    
    v_taker = group['taker_buy_base_asset_volume']
    v_total = group['volume']
    group['volume_delta'] = v_taker - (v_total - v_taker)
    group['vpin'] = group['volume_delta'].abs().rolling(24).sum() / (v_total.rolling(24).sum() + 1e-8)

    # Compatibility Mocks
    group['imbalance'] = 0.0
    group['spread'] = 0.0
    group['ofi'] = 0.0
    group['crisis_flag'] = 0
    group['last_funding_rate'] = 0.0
    group['last_macro_severity'] = 0
    group['macro_risk_factor'] = 1.0
    group['days_until_macro'] = 10
    group['days_since_macro'] = 10
    
    # Regime Detection
    group['regime'] = 'sideways'
    group.loc[group['trend_strength'] > 0.015, 'regime'] = 'trending_bull'
    group.loc[group['trend_strength'] < -0.015, 'regime'] = 'trending_bear'
    
    return group.dropna().tail(50)

def main():
    print("="*50)
    print("GEMINI LIVE MONITOR: AGGRESSIVE GROWTH (2x-3x Goal)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models', 'saved_models')
    engine = FinalEnsembleEngine(model_dir=model_dir)
    
    # Universe updated to strictly Institutional Liquidity (Mandate)
    universe = ['BTCUSDT', 'ETHUSDT']
    threshold = 0.70 # High-Conviction Mandate (Phase 1)
    
    opportunities = []

    for symbol in universe:
        print(f"Scanning {symbol}...")
        raw_data = fetch_recent_klines(symbol)
        if raw_data is None: continue
        
        features = build_live_features(raw_data)
        if len(features) < 1: continue
        
        try:
            decision = engine.evaluate_state(features)
            prob = decision['final_probability']
            regime = decision['regime']
            exp_ret = decision['expected_return']
            
            # Ranking Score for Institutional Swing: Prob + (Exp_Ret * 10)
            # Prioritize both probability and magnitude
            rank_score = prob + (exp_ret * 10)
            
            opportunities.append({
                'symbol': symbol,
                'prob': prob,
                'exp_ret': exp_ret,
                'regime': regime,
                'rank': rank_score,
                'price': features['close'].iloc[-1],
                'atr': features['atr_14'].iloc[-1]
            })
        except: continue

    # Select the #1 "High-Conviction" opportunity
    if opportunities:
        top = sorted(opportunities, key=lambda x: x['rank'], reverse=True)[0]
        
        print("\n" + "-"*30)
        print(f"TOP OPPORTUNITY: {top['symbol']}")
        print(f"Confidence:     {top['prob']:.2%}")
        print(f"Expected Move:  {top['exp_ret']:.2%}")
        print(f"Regime:         {top['regime']}")
        print(f"Current Price:  ${top['price']:,.4f}")
        print("-" * 30)

        mag_threshold = 0.015 # Mandate: Reject if < 1.5%
        if top['prob'] >= threshold and top['exp_ret'] >= mag_threshold:
            print(f"\n[ACTION] GO FOR IT! High Conviction Entry Recommended.")
            print(f"Recommended Position: FULL Rs.10,000")
            print(f"Targeting: {top['exp_ret']*100:.1f}%+ with Trailing Stop")
            
            # Calculate dynamic stop loss
            sl_price = top['price'] - (top['atr'] * 1.5)
            print(f"Initial Stop Loss: ${sl_price:,.4f} (~{((top['price']-sl_price)/top['price']):.2%})")
        else:
            reason = "low confidence" if top['prob'] < threshold else "low magnitude"
            print(f"\n[WAIT] Opportunity rejected due to {reason}.")
            print(f"Confidence: {top['prob']:.1%} (Req: {threshold:.0%})")
            print(f"Expected Move: {top['exp_ret']:.2%} (Req: {mag_threshold:.1%})")
    else:
        print("\n[ERROR] No data available from Binance.")

if __name__ == "__main__":
    main()
