import numpy as np

def sharpe(portfolio_returns, risk_free_rate=0.05):

    daily_rf = risk_free_rate / 252
    excess_returns = portfolio_returns.mean() - daily_rf
    sharpe_ratio = excess_returns / portfolio_returns.std() * np.sqrt(252)

    return float(sharpe_ratio)

def sortino(portfolio_returns, risk_free_rate=0.05):

    daily_rf = risk_free_rate / 252
    excess_returns = portfolio_returns.mean() - daily_rf
    excess_returns_vector = portfolio_returns.values - daily_rf
    only_losses = np.minimum(excess_returns_vector, 0)
    downside_dev = np.sqrt(np.mean(only_losses ** 2))
    sortino_ratio = excess_returns / downside_dev * np.sqrt(252)

    return float(sortino_ratio)

def max_drawdown(portfolio_returns):

    equity_curve = np.exp(np.cumsum(portfolio_returns))
    previous_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - previous_peak) / previous_peak
    mdd = drawdowns.min()

    return float(mdd)

def calmar(portfolio_returns, risk_free_rate=0.05):

    mdd = max_drawdown(portfolio_returns)
    calmar = (portfolio_returns.mean() * 252 - risk_free_rate) / abs(mdd)

    return float(calmar)