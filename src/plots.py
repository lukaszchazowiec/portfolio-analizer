import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

from src.metrics import max_drawdown


def plot_var_comparison(portfolio_returns, hist_var, par_var, mc_var):
    fig, ax = plt.subplots(figsize=(12, 7))

    # 1. histogram with KDE
    sns.histplot(portfolio_returns, bins=100, kde=True, ax=ax)

    # 2. overlay normal distribution curve
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 300)

    # scale to match histogram height
    bin_width = (portfolio_returns.max() - portfolio_returns.min()) / 100
    scale_factor = len(portfolio_returns) * bin_width

    ax.plot(x, stats.norm.pdf(x, mu, sigma) * scale_factor,
            color='blue', linewidth=2, linestyle='-',
            label='rozkład normalny', alpha=0.7)

    # 3. vertical VaR lines
    ax.axvline(-hist_var, color='red', linestyle='--', linewidth=2,
               label=f'hist VaR: {hist_var:.2%}')
    ax.axvline(-par_var, color='orange', linestyle='-', linewidth=2,
               label=f'par VaR: {par_var:.2%}')
    ax.axvline(-mc_var, color='green', linestyle=':', linewidth=2,
               label=f'MC VaR: {mc_var:.2%}')

    ax.set_title('Comparison of VaR methods', fontsize=15)
    ax.set_xlabel('Daily portfolio log returns', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    return fig

def plot_cumulative_return(portfolio_returns):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(portfolio_returns.cumsum(), color='blue', linewidth=2, linestyle='-',
            label='cumulative returns')
    ax.set_title('Cumulative returns', fontsize=15)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Cumulative return', fontsize=12)

    return fig

def plot_heatmap(cov_matrix):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title('Covariance matrix', fontsize=15)
    ax.set_xlabel('Stock', fontsize=12)
    ax.set_ylabel('Stock', fontsize=12)

    return fig


def plot_drawdown(portfolio_returns):

    wealth_index = np.exp(np.cumsum(portfolio_returns))
    previous_peaks = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    mdd_value = max_drawdown(portfolio_returns)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
    ax.plot(drawdowns.index, drawdowns, color='red', linewidth=1)

    ax.set_title(f'Underwater Plot | Maximum Drawdown: {mdd_value:.2%}')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    return fig
