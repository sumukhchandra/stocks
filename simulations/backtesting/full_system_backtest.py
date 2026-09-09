import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_models.final_ensemble_engine import FinalEnsembleEngine
from execution.latency_simulator import LatencySimulator, ExecutionFilter

class FullSystemBacktest:
    def __init__(self, data_path='data/processed/master_labeled_dataset.parquet'):
        self.data_path = data_path
        self.engine = FinalEnsembleEngine()
        self.latency_sim = LatencySimulator(base_latency_ms=50, jitter_ms=20)
        self.exec_filter = ExecutionFilter(min_prob=0.6, max_spread=0.001, max_vpin=0.8)
        self.trades = []
        self.capital = 10000.0

    def run(self, start_idx=0, end_idx=None, window_size=16):
        if not os.path.exists(self.data_path):
            print("Master dataset not found.")
            return
            
        df = pd.read_parquet(self.data_path)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if end_idx is None:
            end_idx = len(df)
            
        print(f"Running Full System Backtest on {end_idx - start_idx} bars...")
        
        # We need continuous windows, so we'll just slice the dataframe
        for i in tqdm(range(start_idx + window_size, end_idx), desc="Backtesting"):
            window = df.iloc[i - window_size : i]
            current_state = window.iloc[-1]
            
            decision = self.engine.evaluate_state(window)
            prob = decision['final_probability']
            
            # Simple spread proxy if missing
            spread = current_state.get('spread', 0.0001) 
            vpin = current_state.get('vpin', 0.0)
            regime = decision['regime']
            
            # Execution Filter
            should_trade, reason = self.exec_filter.should_trade(prob, spread, vpin, regime)
            
            if should_trade:
                price = current_state['close']
                volatility = current_state.get('volatility', 0.01)
                
                # Determine direction
                side = 'buy' if prob > 0.5 else 'sell'
                
                # Simulate slippage
                exec_price, slippage_pct = self.latency_sim.calculate_slippage(price, volatility, side=side)
                
                # Calculate trade outcome simply (e.g. holding for future_return_t1)
                # Note: this is a simplification. Real backtest would track the open position.
                actual_return = current_state['future_return_t1']
                
                # If sell, return is inverted
                if side == 'sell':
                    actual_return = -actual_return
                    
                # Subtract slippage
                net_return = actual_return - slippage_pct
                
                # Funding cost (simplified: apply current rate to the position)
                funding_rate = current_state.get('last_funding_rate', 0.0)
                if side == 'buy':
                    net_return -= funding_rate
                else:
                    net_return += funding_rate
                
                # Transaction cost (exchange fee)
                fee_pct = 0.0004 # 4 bps maker/taker avg
                net_return -= fee_pct
                
                trade_pnl = decision['recommended_position_size'] * net_return
                self.capital += trade_pnl
                
                self.trades.append({
                    'timestamp': decision['timestamp'],
                    'side': side,
                    'regime': regime,
                    'prob': prob,
                    'size': decision['recommended_position_size'],
                    'slippage': slippage_pct,
                    'net_return': net_return,
                    'pnl': trade_pnl,
                    'capital': self.capital
                })

    def print_results(self):
        if not self.trades:
            print("No trades executed.")
            return
            
        trades_df = pd.DataFrame(self.trades)
        
        win_rate = (trades_df['net_return'] > 0).mean()
        avg_win = trades_df[trades_df['net_return'] > 0]['net_return'].mean() if win_rate > 0 else 0
        avg_loss = trades_df[trades_df['net_return'] < 0]['net_return'].mean() if win_rate < 1.0 else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Drawdown calculation
        trades_df['equity_peak'] = trades_df['capital'].cummax()
        trades_df['drawdown'] = (trades_df['capital'] - trades_df['equity_peak']) / trades_df['equity_peak']
        max_dd = trades_df['drawdown'].min()
        
        # Approximate Sharpe (assuming risk free = 0)
        returns = trades_df['net_return']
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 24 * 12) 
        
        print("\n--- Backtest Results ---")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Max Drawdown: {max_dd*100:.2f}%")
        print(f"Expectancy (per trade): {expectancy*100:.4f}%")
        print(f"Estimated Sharpe: {sharpe:.2f}")
        print(f"Avg Slippage: {trades_df['slippage'].mean()*100:.4f}%")
        
if __name__ == "__main__":
    bt = FullSystemBacktest()
    # Run on a small subset for quick validation
    bt.run(end_idx=5000)
    bt.print_results()
