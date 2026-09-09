import asyncio
import pandas as pd
import xgboost as xgb
import os
import sys
import joblib
from datetime import datetime

# Add project root and current dir to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paper_wallet import PaperWallet
from risk.volatility_filter import VolatilityFilter
from risk.global_risk_controller import GlobalRiskController
from risk.post_tax_expectancy_filter import PostTaxExpectancyFilter
from risk.trade_ranking_engine import TradeRankingEngine
from risk.inr_accounting_engine import INRAccountingEngine
from execution.execution_quality_monitor import ExecutionQualityMonitor
from execution.market_impact_model import MarketImpactModel

from ml_models.final_ensemble_engine import FinalEnsembleEngine

class PaperExecutor:
    def __init__(self, metrics_dir='logs/live_metrics'):
        self.wallet = PaperWallet(initial_balance=10000)
        self.engine = FinalEnsembleEngine()
        self.risk_controller = GlobalRiskController()
        self.tax_filter = PostTaxExpectancyFilter(min_net_bps=15)
        self.ranker = TradeRankingEngine(min_gross_move=0.015)
        self.inr_accounting = INRAccountingEngine(initial_balance_inr=800000)
        self.impact_model = MarketImpactModel()
        self.exec_monitor = ExecutionQualityMonitor()
        self.metrics_dir = metrics_dir
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.metrics_dir, 'daily_metrics.csv')
        self.trades_path = os.path.join(self.metrics_dir, 'trades.csv')
        
    async def log_decision(self, decision):
        """Logs engine decisions to a persistent CSV."""
        df = pd.DataFrame([decision])
        # Flatten signals
        if 'signals' in df.columns:
            for k, v in decision['signals'].items():
                df[f'signal_{k}'] = v
            df = df.drop(columns=['signals'])
            
        header = not os.path.exists(self.metrics_path)
        df.to_csv(self.metrics_path, mode='a', header=header, index=False)

    async def log_trade(self, action, symbol, price, qty, profit=0):
        """Logs trade execution to a persistent CSV."""
        log_entry = {
            'timestamp': datetime.now(),
            'action': action,
            'symbol': symbol,
            'price': price,
            'qty': qty,
            'profit': profit,
            'balance': self.wallet.balance
        }
        df = pd.DataFrame([log_entry])
        header = not os.path.exists(self.trades_path)
        df.to_csv(self.trades_path, mode='a', header=header, index=False)
        
    async def evaluate_trade(self, features_df):
        """
        Evaluates the current features and executes a trade using the integrated engine.
        """
        if features_df is None or len(features_df) < 16:
            return
            
        latest = features_df.iloc[-1]
        symbol = latest.get('symbol', 'UNKNOWN')
        
        # 1. Asset Restriction (BTC/ETH only for liquidity)
        if symbol not in ['BTCUSDT', 'ETHUSDT', 'SYNTH_BTC']: # SYNTH_BTC for validation
            return

        market_vol = latest.get('volatility', 0.01)
        
        # 2. Global Risk Check
        is_safe, reason = self.risk_controller.check_safety(self.wallet.balance, self.wallet.positions, market_vol)
        if not is_safe:
            print(f"[!] RISK SHUTDOWN: {reason}")
            return

        # 3. Decision & Ranking
        decision = self.engine.evaluate_state(features_df)
        
        # Get trade count for frequency penalty
        daily_count = self.inr_accounting.state['trade_count'] # Simplified to total count for now
        
        rank_res = self.ranker.calculate_score(
            decision['final_probability'],
            decision['expected_return'],
            decision['confidence_score'],
            market_vol,
            decision['regime'],
            daily_trade_count=daily_count
        )
        
        if not rank_res['is_high_quality']:
            if rank_res.get('reason') != 'MOVE_TOO_SMALL':
                print(f"Trade Rejected: {rank_res.get('reason', 'LOW_QUALITY')} | Quality: {rank_res.get('quality_score', 0.0):.4f} | Net Exp: {rank_res.get('net_expectancy_bps', 0.0):.2f} bps")
            return

        await self.log_decision(decision)
        
        current_price = decision['price']
        print(f"[{decision['timestamp']}] Engine Decision for {symbol} | Prob: {decision['final_probability']:.4f} | Exp Ret: {decision['expected_return']*100:.2f}% | Regime: {decision['regime']}")
        
        # 4. Execution logic
        recommended_size = decision['recommended_position_size']
        
        if recommended_size > 0:
            if symbol not in self.wallet.positions:
                # Calculate dynamic slippage
                spread_bps = (latest.get('spread', 0.5) / current_price) * 10000
                slippage_bps = self.impact_model.calculate_slippage(recommended_size, spread_bps, market_vol)
                exec_price = current_price * (1 + slippage_bps/10000)
                
                # Sizing check against risk limits
                max_size = self.wallet.balance * self.risk_controller.limits['max_position_size_pct']
                actual_size = min(recommended_size, max_size)

                if self.wallet.buy(symbol, exec_price, actual_size):
                    qty = actual_size / exec_price
                    await self.log_trade('BUY', symbol, exec_price, qty)
                    self.exec_monitor.log_execution(symbol, 'BUY', current_price, exec_price, 50)
        
        # 5. Exit Logic & INR Accounting
        if symbol in self.wallet.positions:
            entry = self.wallet.positions[symbol]['entry_price']
            change = (current_price - entry) / entry
            
            # Use engine-recommended hold time or 2% profit target / 1% stop loss
            if change >= 0.025 or change <= -0.015:
                pos = self.wallet.positions[symbol]
                qty = pos['qty']
                invested = pos['invested']
                
                # Exit slippage
                slippage_bps = self.impact_model.calculate_slippage(qty * current_price, 0.5, market_vol)
                exit_price = current_price * (1 - slippage_bps/10000)
                
                if self.wallet.sell(symbol, exit_price):
                    revenue = qty * exit_price
                    profit_usd = revenue - invested
                    await self.log_trade('SELL', symbol, exit_price, qty, profit_usd)
                    self.exec_monitor.log_execution(symbol, 'SELL', current_price, exit_price, 10)
                    
                    # Update INR Ledger
                    self.inr_accounting.record_trade(profit_usd, revenue)

if __name__ == "__main__":
    executor = PaperExecutor()
    print("Paper executor initialized with HMM Integration and Trade Quality Filters.")
