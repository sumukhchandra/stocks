import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) == 0:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    # Assuming returns are 5m returns, annualized factor = 288 * 365
    annualized_factor = np.sqrt(288 * 365)
    return (mean_return - risk_free_rate) / std_return * annualized_factor

def calculate_max_drawdown(cumulative_returns):
    if len(cumulative_returns) == 0:
        return 0.0
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    return np.min(drawdowns)

def calculate_expectancy(win_rate, avg_win, avg_loss):
    if np.isnan(avg_win): avg_win = 0
    if np.isnan(avg_loss): avg_loss = 0
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
