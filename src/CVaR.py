from scipy.stats import norm

from src.VaR import calc_hist_var


def calc_hist_cvar(portfolio_returns, confidence=0.95):
    hist_var = calc_hist_var(portfolio_returns, confidence)
    tail_losses = portfolio_returns[portfolio_returns <= -hist_var]

    return float(-tail_losses.mean())


def calc_parametric_cvar(portfolio_returns, confidence=0.95):
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    alpha = 1-confidence

    z_score = norm.ppf(alpha)
    par_cvar = -(mu - sigma * (norm.pdf(z_score) / alpha))

    return float(par_cvar)

