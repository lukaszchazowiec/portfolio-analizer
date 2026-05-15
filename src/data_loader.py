import numpy as np
import yfinance as yf

from dateutil.relativedelta import relativedelta
from datetime import date

def get_price_data(tickers, years=5):
    end = date.today()
    start = end - relativedelta(years=years)
    prices = yf.download(tickers, start=start, end=end)['Close']
    return prices


def compute_log_returns(prices):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def compute_portfolio_returns(log_returns, weights):
    portfolio_returns = log_returns.dot(weights)
    return portfolio_returns



