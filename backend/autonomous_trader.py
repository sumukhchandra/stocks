import pandas as pd
import numpy as np
import os
import sys
import requests
import time
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.final_ensemble_engine import FinalEnsembleEngine
from features.technical import calculate_rsi, calculate_vwap, calculate_atr
from risk.inr_accounting_engine import INRAccountingEngine
from data.news_sentiment_engine import CryptoNewsfetcher
from data.cross_asset_engine import CrossAssetEngine
from data.whale_flow_monitor import WhaleFlowMonitor

# --- 24-HOUR HIGH-CONVICTION PIVOT (User Plan: 100% ETH) ---
UNIVERSE = ['ETHUSDT']

# Entry Thresholds optimized for ETH Momentum
TREND_THRESHOLD = 0.72  # Slightly lower for faster entry in a 24h window
TREND_TARGET = 0.05     # 5% Target for 24h
TREND_INTERVAL = '1h'

SCALP_THRESHOLD = 0.85
SCALP_TARGET = 0.02
SCALP_INTERVAL = '5m'

STATE_FILE = 'logs/live_metrics/dual_autonomous_state.json'

class DualAutonomousEngine:
    def __init__(self, initial_capital=10000):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.script_dir, 'models', 'saved_models')
        self.engine = FinalEnsembleEngine(model_dir=self.model_dir)
        self.ledger = INRAccountingEngine(initial_balance_inr=initial_capital, ledger_path=os.path.join(self.script_dir, 'logs', 'live_metrics', 'dual_ledger.json'))
        self.news = CryptoNewsfetcher()
        self.macro = CrossAssetEngine()
        self.whale = WhaleFlowMonitor()
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                'current_capital': 10000.0,
                'in_position': False,
                'current_trade': None,
                'total_trades': 0,
                'last_retrain': datetime.now().isoformat()
            }

    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def fetch_data(self, symbol, interval):
        # 1. Fetch Klines
        kline_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100"
        # 2. Fetch Orderbook Depth
        depth_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100"
        
        try:
            k_res = requests.get(kline_url).json()
            d_res = requests.get(depth_url).json()
            
            df = pd.DataFrame(k_res, columns=['ot','o','h','l','c','v','ct','qv','nt','tbb','tbq','i'])
            for col in ['o','h','l','c','v','tbb']: df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['ot'], unit='ms')
            df['symbol'] = symbol
            df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume','tbb':'taker_buy_base_asset_volume', 'nt':'number_of_trades'})
            
            # Extract Depth Metrics
            bids = np.array(d_res['bids'], dtype=float)
            asks = np.array(d_res['asks'], dtype=float)
            
            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            bid_vol = bids[:, 1].sum()
            ask_vol = asks[:, 1].sum()
            
            depth_metrics = {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': best_ask - best_bid,
                'imbalance': (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
            }
            
            return df, depth_metrics
        except Exception as e:
            print(f"      Data Fetch Error: {e}")
            return None, None

    def build_features(self, df, depth_metrics):
        group = df.copy().sort_values('timestamp')
        group['rsi_14'] = calculate_rsi(group, period=14, price_col='close')
        group['vwap'] = calculate_vwap(group, price_col='close', volume_col='volume')
        group['vwap_dist'] = (group['close'] - group['vwap']) / group['vwap']
        group['atr_14'] = calculate_atr(group, period=14)
        group['ema_20'] = group['close'].ewm(span=20).mean()
        group['ema_50'] = group['close'].ewm(span=50).mean()
        group['trend_strength'] = (group['ema_20'] - group['ema_50']) / group['ema_50']
        group['volatility'] = group['close'].pct_change().rolling(24).std()
        
        v_taker = group['taker_buy_base_asset_volume'].astype(float)
        v_total = group['volume'].astype(float)
        group['volume_delta'] = v_taker - (v_total - v_taker)
        
        group['vol_sum_24'] = group['volume'].rolling(2).sum() 
        group['vdelta_abs_sum_24'] = group['volume_delta'].abs().rolling(2).sum()
        group['vpin'] = group['vdelta_abs_sum_24'] / (group['vol_sum_24'] + 1e-8)
        group['downloaded_at'] = 0.0

        # Real Depth Integration
        group['spread'] = depth_metrics['spread']
        group['imbalance'] = depth_metrics['imbalance']

        for col in ['ofi','crisis_flag','last_funding_rate','last_macro_severity','macro_risk_factor','days_until_macro','days_since_macro']:
            group[col] = 0.0
        group['regime'] = 'sideways'
        group.loc[group['trend_strength'] > 0.015, 'regime'] = 'trending_bull'
        group.loc[group['trend_strength'] < -0.015, 'regime'] = 'trending_bear'
        return group.dropna().tail(50)

    def scan_side(self, interval, threshold, strategy_name):
        # 1. Global Confirmation 
        macro_signals = self.macro.fetch_macro_signals()
        whale_sentiment = float(np.mean(self.whale.get_whale_sentiment()))
        macro_score = float(np.mean(macro_signals['macro_risk_score']))
        
        # Hard Filter: No Global Risk-Off
        if macro_score < -0.012:
            return None

        best_opp = None
        best_score = -1

        for symbol in UNIVERSE:
            data, depth = self.fetch_data(symbol, interval)
            if data is None: continue
            features = self.build_features(data, depth)
            try:
                # Noise Cancellation Layer 1: Volatility Floor
                # Ignore "Dead Markets" where nothing is happening
                current_vol = float(features['volatility'].iloc[-1])
                if current_vol < 0.0005: 
                    continue

                # Noise Cancellation Layer 2: Trend Confirmation
                # Only look for buys in Bullish or Neutral-recovering regimes
                regime = features['regime'].iloc[-1]
                if regime == 'trending_bear':
                    continue

                coin_base = symbol.replace('USDT', '')
                news_sentiment = float(self.news.fetch_sentiment(coin_base))
                
                decision = self.engine.evaluate_state(features)
                prob = float(decision['final_probability'])
                
                # Multi-Factor Conviction Synthesis
                conviction = (prob * 0.4) + (news_sentiment * 0.2) + ((macro_score + 0.05)*2 * 0.2) + (whale_sentiment * 0.2)
                
                trend = float(features['trend_strength'].iloc[-1])
                score = conviction + (abs(trend) * 5)
                
                # High-Signal Logging Only
                if conviction >= (threshold - 0.15): # Only log if it's "close" to a signal
                    print(f"  [SIGNAL] {symbol}: Conviction {conviction:.1%} (Tech: {prob:.1%}, News: {news_sentiment:.2f}) | Vol: {current_vol:.4f}")
                
                if conviction >= threshold and news_sentiment >= 0.45 and score > best_score:
                    best_score = score
                    best_opp = {
                        'symbol': symbol,
                        'price': features['close'].iloc[-1],
                        'prob': conviction,
                        'atr': features['atr_14'].iloc[-1],
                        'strategy': strategy_name,
                        'sentiment': news_sentiment,
                        'spread': depth['spread'],
                        'imbalance': depth['imbalance']
                    }
                    
                    # Reality-Grade Trade Ticket
                    print("\n--- [!] ELITE TRADE TICKET [!] ---")
                    print(f"  Symbol:    {symbol}")
                    print(f"  Conviction: {conviction:.2%}")
                    print(f"  Sentiment:  {news_sentiment:.2f}")
                    print(f"  L2 Spread:  {depth['spread']:.6f}")
                    print(f"  L2 Imbal:   {depth['imbalance']:.4f}")
                    print("---------------------------------")
            except Exception:
                continue
        return best_opp

    def run_cycle(self):
        # Silent Mode: Only print cycle header if we find something or once an hour
        current_time = datetime.now().strftime('%H:%M')
        
        if self.state['in_position']:
            print(f"\n--- Monitoring Position: {current_time} ---")
            self.manage_position()
            return

        # Scanning for Elite setups
        big_opp = self.scan_side(TREND_INTERVAL, TREND_THRESHOLD, "TREND")
        
        # Real-Time Heartbeat Ticker
        best_sym = big_opp['symbol'] if isinstance(big_opp, dict) else 'None'
        best_prob = big_opp['prob'] if isinstance(big_opp, dict) else 0.0
        print(f"  [TICKER] {current_time} | Best: {best_sym} | Conviction: {best_prob:.1%}")

        if big_opp:
            self.execute_buy(big_opp)
            return

        # Background Pulse (every 10 cycles) to show bot is alive
        if int(time.time() / 60) % 10 == 0:
            print(f"  [REAL-TIME PULSE] {current_time} | Balance: Rs.{self.state['current_capital']:.2f} | Status: Scanning Orderbooks...")

    def execute_buy(self, opp):
        print(f"\n [ACTION] Entering {opp['strategy']} BUY: {opp['symbol']} @ ${opp['price']:.4f}")
        self.state['in_position'] = True
        self.state['current_trade'] = {
            'symbol': opp['symbol'],
            'entry_price': opp['price'],
            'highest_price': opp['price'],
            'atr': opp['atr'],
            'capital_inr': self.state['current_capital'],
            'strategy': opp['strategy']
        }
        self.save_state()

    def manage_position(self):
        trade = self.state['current_trade']
        interval = TREND_INTERVAL if trade['strategy'] == 'TREND' else SCALP_INTERVAL
        data, depth = self.fetch_data(trade['symbol'], interval)
        if data is None: return
        
        current_price = data['close'].iloc[-1]
        if current_price > trade['highest_price']: trade['highest_price'] = current_price
        
        # User requested Stop Loss for even 1-10 Rupees loss.
        # On a 10,000 INR trade, 1 INR is 0.01% (0.0001).
        # We trigger if PnL is negative (even slightly).
        is_losing = pnl_pct < -0.0005 # Approx 5 Rupees loss
        
        target = TREND_TARGET if trade['strategy'] == 'TREND' else SCALP_TARGET
        
        print(f"  Holding {trade['strategy']} on {trade['symbol']} | Price: ${current_price:.4f} | PnL: {pnl_pct:.2%} (Loss Exit: {is_losing})")

        if is_losing:
            self.execute_sell(current_price, "Ultra-Tight Stop Loss (Rs.5-Rs.10 threshold)")
        elif pnl_pct >= target:
            self.execute_sell(current_price, f"Target Hit ({target:.1%})")

    def execute_sell(self, price, reason):
        trade = self.state['current_trade']
        pnl_pct = (price - trade['entry_price']) / trade['entry_price']
        
        revenue_usd = (trade['capital_inr'] / 83.5) * (1 + pnl_pct)
        profit_usd = (trade['capital_inr'] / 83.5) * pnl_pct
        summary = self.ledger.record_trade(profit_usd, revenue_usd)
        
        self.state['current_capital'] = summary['current_liquid_capital_inr']
        self.state['in_position'] = False
        self.state['current_trade'] = None
        self.state['total_trades'] += 1
        self.save_state()
        
        print(f"\n [EXIT] {trade['strategy']} SELL {trade['symbol']} | Result: {pnl_pct:.2%} | New Balance: Rs.{self.state['current_capital']:.2f}")

if __name__ == "__main__":
    bot = DualAutonomousEngine()
    print("Institutional Trading Terminal ACTIVE.")
    while True:
        try:
            bot.run_cycle()
            time.sleep(60) # High-frequency scan
        except KeyboardInterrupt: break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
