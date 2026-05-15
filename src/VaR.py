import numpy as np
from numpy.linalg import cholesky

from scipy.stats import norm


# HISTORICAL VaR

def calc_hist_var(portfolio_returns, confidence=0.95):
    hist_var = portfolio_returns.quantile(1-confidence)
    return float(-hist_var)


# PARAMETRIC VaR

def calc_parametric_var(portfolio_returns, confidence=0.95):
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    par_var = norm.ppf(1-confidence, loc=mu, scale=sigma)
    return float(-par_var)


# MONTE CARLO VaR (Brownian Motion)

def calc_mc_var(log_returns, weights, confidence=0.95, n_simulations=10000, steps=1):
    weights = np.array(weights)
    n_assets = len(weights)
    mu = log_returns.mean().values
    cov = log_returns.cov().values
    L = cholesky(cov)
    dt = 1/steps    # time step

    simulated_portfolio_returns = np.zeros(n_simulations)

    for i in range(n_simulations):        # TODO: vectorize using numpy matrix ops for  better performance, loop is too slow
        z = norm.rvs(size=(n_assets, steps))
        correlated_shocks = L @ z
        asset_returns = mu.reshape(-1,1) * dt + correlated_shocks * np.sqrt(dt)
        asset_returns = asset_returns.sum(axis=1)
        simulated_portfolio_returns[i] = weights @ asset_returns

    mc_var = np.quantile(simulated_portfolio_returns, 1-confidence)
    return float(-mc_var)

