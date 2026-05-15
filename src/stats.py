import numpy as np
from scipy import stats

def annual_cov_matrix(log_returns):
    cov_matrix = log_returns.cov() * 252
    return cov_matrix

def annualized_mean_returns(portfolio_returns, log_returns):
    annualized_portfolio_return = portfolio_returns.mean() * 252
    annualized_asset_return = log_returns.mean() * 252

    return annualized_portfolio_return, annualized_asset_return


def calculate_portfolio_variance(log_returns, weights):
    daily_cov_matrix = log_returns.cov().values
    annualized_cov_matrix = daily_cov_matrix * 252
    w = weights.reshape(-1, 1)
    portfolio_variance = (w.T @ annualized_cov_matrix @ w).item()
    portfolio_volatility = np.sqrt(portfolio_variance)

    return float(portfolio_variance), float(portfolio_volatility)

def calculate_skew_kurt(portfolio_returns):
    """
    Calculate skewness and excess kurtosis of portfolio returns.

    Skewness: measures asymmetry of the distribution
        - negative: left tail is heavier (more extreme losses)
        - positive: right tail is heavier
    Kurtosis: measures tail thickness (excess kurtosis, normal = 0)
        - positive: fat tails, extreme returns more frequent than normal distribution predicts
    """
    skew = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurtosis()
    return float(skew), float(kurtosis)

def test_normality(portfolio_returns):
    """
    Jarque-Bera test for normality of portfolio returns.

    H0: returns follow a normal distribution
    H1: returns do not follow a normal distribution

    p-value < 0.05: reject H0 - distribution is not normal (fat tails / skewness)
    p-value > 0.05: fail to reject H0 - distribution may be normal
    """
    jb_stat, jb_p = stats.jarque_bera(portfolio_returns)
    result = {
        'jb_stat' : float(jb_stat),
        'jb_p' : float(jb_p),
        'is normal' : bool(jb_p > 0.05)
    }

    return result

def fit_t_distribution(portfolio_returns):
    """
    Fit Student's t-distribution to portfolio returns using MLE.

    df: degrees of freedom, controls tail thickness
        - df < 5: very fat tails
        - df < 10: fat tails, t-distribution significantly better than normal
        - df > 30: nearly identical to normal distribution
    loc: mean of the distribution
    scale: spread parameter (not equal to std dev for t-distribution)
    """
    df, loc, scale = stats.t.fit(portfolio_returns)
    return {
        'df': float(df),
        'loc': float(loc),
        'scale': float(scale)
    }