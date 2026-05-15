from data_loader import get_price_data, compute_log_returns, compute_portfolio_returns
from stats import annualized_mean_returns, calculate_portfolio_variance, calculate_skew_kurt, test_normality, fit_t_distribution, annual_cov_matrix
from VaR import calc_hist_var, calc_parametric_var, calc_mc_var
from CVaR import calc_hist_cvar, calc_parametric_cvar
from metrics import sharpe, sortino, max_drawdown, calmar
from plots import plot_var_comparison, plot_cumulative_return, plot_heatmap, plot_drawdown

import os
os.makedirs("output", exist_ok=True)

import numpy as np

tickers = ["AAPL", "MSFT", "JPM", "GS", "XOM", "JNJ", "TSLA", "SPY", "GLD", "TLT"]
weights = np.array([0.1] * 10)

def main():

    # DATA
    price_data = get_price_data(tickers, years=5)
    log_returns = compute_log_returns(price_data)
    portfolio_returns = compute_portfolio_returns(log_returns, weights)
    plot_cumulative_return(portfolio_returns)

    print("\n--- Portfolio Performance Data Saved---")

    # STATS
    annual_portfolio_returns, annual_asset_returns = annualized_mean_returns(portfolio_returns, log_returns)
    variance, volatility = calculate_portfolio_variance(log_returns, weights)
    skewness, kurtosis = calculate_skew_kurt(portfolio_returns)
    jarque_bera = test_normality(portfolio_returns)
    t_distribution = fit_t_distribution(portfolio_returns)

    print("\n--- Portfolio Stats ---")
    print(f"Annualized Portfolio Returns: {annual_portfolio_returns:.4f}")
    print(f"Volatility: {volatility:.4f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    print(f"Jarque-Bera: {jarque_bera}")
    print(f"T Distribution: {t_distribution}")

    # VaR
    hist_var = calc_hist_var(portfolio_returns, confidence=0.95)
    parametric_var = calc_parametric_var(portfolio_returns, confidence=0.95)
    mc_var = calc_mc_var(log_returns, weights)

    print("\n--- Var (95%, 1 day) ---")
    print(f"VaR (hist): {hist_var:.4f}")
    print(f"VaR (par): {parametric_var:.4f}")
    print(f"MC VaR: {mc_var:.4f}")

    # CVaR
    hist_cvar = calc_hist_cvar(portfolio_returns, confidence=0.95)
    parametric_cvar = calc_parametric_cvar(portfolio_returns, confidence=0.95)

    print("\n--- CVaR (95%, 1 day) ---")
    print(f"CVaR (hist): {hist_cvar:.4f}")
    print(f"CVaR (par): {parametric_cvar:.4f}")

    # Metrics
    sharpe_ratio = sharpe(portfolio_returns, risk_free_rate=0.05)
    sortino_ratio = sortino(portfolio_returns, risk_free_rate=0.05)
    max_dd = max_drawdown(portfolio_returns)
    calmar_ratio = calmar(portfolio_returns, risk_free_rate=0.05)

    print("\n--- Metrics ---")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {sortino_ratio:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")
    print(f"Calmar Ratio: {calmar_ratio:.4f}")

    # PLOTS
    print("\n--- Plots ---")
    fig1 = plot_var_comparison(portfolio_returns, hist_var, parametric_var, mc_var)
    fig1.savefig("output/var_comparison.png")
    fig2 = plot_heatmap(annual_cov_matrix(log_returns))
    fig2.savefig("output/covariance_matrix.png")
    fig3 = plot_drawdown(portfolio_returns)
    fig3.savefig("output/drawdown.png")
    fig4 = plot_cumulative_return(portfolio_returns)
    fig4.savefig("output/cumulative_return.png")
    print("Plots saved to the current directory.")

if __name__ == "__main__":
    main()
    
