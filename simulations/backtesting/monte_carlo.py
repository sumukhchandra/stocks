import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class MonteCarloSimulator:
    def __init__(self, trades_path='data/processed/backtest_trades.parquet', n_simulations=1000):
        self.trades_path = trades_path
        self.n_simulations = n_simulations
        
    def run(self):
        if not os.path.exists(self.trades_path):
            print(f"Trades file not found: {self.trades_path}")
            return
            
        trades = pd.read_parquet(self.trades_path)
        if len(trades) < 10:
            print("Not enough trades for Monte Carlo.")
            return
            
        base_returns = trades['future_return'].values
        
        # Slippage sensitivity analysis
        slippage_levels = [0.0, 0.0005, 0.001, 0.0015, 0.002] # 0 bps to 20 bps round trip
        survival_probs = []
        median_returns = []
        
        os.makedirs('../experiments/monte_carlo', exist_ok=True)
        
        print("\n--- Slippage Sensitivity Analysis ---")
        for sl in slippage_levels:
            final_equities = []
            for _ in range(self.n_simulations // 2): # Run fewer simulations per slippage level to save time
                shuffled = np.random.choice(base_returns, size=len(base_returns), replace=True)
                # Apply deterministic slippage + fee (0.05% taker * 2)
                net = shuffled - sl - 0.001
                cum = np.cumprod(1 + net)
                final_equities.append(cum[-1])
            
            surv = np.mean(np.array(final_equities) > 0.5)
            med_ret = np.median(final_equities)
            survival_probs.append(surv)
            median_returns.append(med_ret)
            print(f"Slippage: {sl*10000:.0f} bps | Survival (>50%): {surv*100:.1f}% | Median Equity: {med_ret:.2f}x")
            
        # Plot Slippage Decay
        plt.figure(figsize=(8, 5))
        plt.plot([sl * 10000 for sl in slippage_levels], survival_probs, marker='o', color='b')
        plt.title('Strategy Survival Probability vs. Slippage')
        plt.xlabel('Round Trip Slippage (bps)')
        plt.ylabel('Survival Probability')
        plt.grid(True, alpha=0.3)
        plt.savefig('../experiments/monte_carlo/slippage_decay.png')
        plt.close()
        
        # Standard Monte Carlo (using a realistic baseline slippage 5 bps round trip)
        print(f"\nRunning {self.n_simulations} Monte Carlo simulations on {len(trades)} trades with baseline slippage...")
        max_drawdowns = []
        final_equities = []
        
        for i in range(self.n_simulations):
            shuffled_returns = np.random.choice(base_returns, size=len(base_returns), replace=True)
            # Stochastic slippage: mean 5 bps, std 2 bps
            slippage = np.random.normal(loc=0.0005, scale=0.0002, size=len(shuffled_returns))
            net_returns = shuffled_returns - slippage - 0.001 # 5 bps fee * 2
            
            cum_ret = np.cumprod(1 + net_returns)
            drawdown = (np.maximum.accumulate(cum_ret) - cum_ret) / np.maximum.accumulate(cum_ret)
            max_dd = np.max(drawdown)
            
            max_drawdowns.append(max_dd)
            final_equities.append(cum_ret[-1])
            
        # Analytics
        survival_prob = np.mean(np.array(final_equities) > 0.5)
        worst_dd_99 = np.percentile(max_drawdowns, 99)
        median_dd = np.median(max_drawdowns)
        median_eq = np.median(final_equities)
        
        print("\n--- Standard Monte Carlo Results ---")
        print(f"Survival Probability (>50% equity): {survival_prob * 100:.2f}%")
        print(f"Worst-Case Drawdown (99th %ile): {worst_dd_99 * 100:.2f}%")
        print(f"Median Max Drawdown: {median_dd * 100:.2f}%")
        print(f"Median Final Equity Multiplier: {median_eq:.2f}x")
        
        # Plot distribution of Drawdowns
        plt.figure(figsize=(10, 5))
        plt.hist(max_drawdowns, bins=50, alpha=0.75, color='red')
        plt.title('Monte Carlo Max Drawdown Distribution')
        plt.xlabel('Max Drawdown')
        plt.ylabel('Frequency')
        plt.axvline(worst_dd_99, color='k', linestyle='dashed', linewidth=1)
        plt.text(worst_dd_99, plt.ylim()[1]*0.9, f' 99th %ile: {worst_dd_99*100:.1f}%')
        
        plt.savefig('../experiments/monte_carlo/drawdown_dist.png')
        print("\nSaved plots to ../experiments/monte_carlo/")
        plt.close()
        
if __name__ == "__main__":
    mc = MonteCarloSimulator()
    mc.run()
