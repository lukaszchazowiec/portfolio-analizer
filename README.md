# Portfolio Analyzer

A Python-based portfolio risk analytics tool implementing multiple VaR/CVaR 
estimation methods and key performance metrics.

## Portfolio

This project uses an equally-weighted portfolio of 10 S&P 500 stocks (10% each).
The portfolio is hypothetical and constructed for educational purposes only.

## Methods Implemented

**Risk Metrics**
- Historical VaR & CVaR
- Parametric VaR & CVaR (normal distribution)
- Monte Carlo VaR (Geometric Brownian Motion with Cholesky decomposition)

**Performance Metrics**
- Annualized return & volatility
- Sharpe & Sortino Ratios
- Maximum Drawdown
- Calmar Ratio

## Project Structure

portfolio_analizer/
├── src/
│   ├── main.py         # entry point
│   ├── data_loader.py  # price data via yfinance
│   ├── VaR.py          
│   ├── CVaR.py         
│   ├── stats.py        # descriptive statistics, normality tests
│   ├── metrics.py      # performance metrics
│   └── plots.py        
├── notebooks/
│   └── analysis.ipynb  # interactive analysis
└── requirements.txt

## Getting Started

pip install -r requirements.txt
python src/main.py

## Key Findings

- Jarque-Bera test rejects normality of portfolio returns (p-value < 0.05)
- Fitted Student's t-distribution confirms fat tails (typical ν ≈ 3-6 for equity returns)
- At the time of the analysis, kurtosis > 0, which further 
  confirms the suspicion of non-normality
- Historical CVaR consistently exceeds Parametric CVaR - normal distribution 
  underestimates tail risk
- Results are computed on rolling 5-year window and will vary with market conditions

## Limitations & Future Work

- Monte Carlo simulation uses a loop instead of vectorized operations
- Normal distribution assumption in parametric methods underestimates tail risk
- Potential improvements: replace normal distribution with Student's t in MC and parametric VaR:
- In Monte Carlo, bootstrap from historical returns instead of parametric distribution:
  `z = np.random.choice(log_returns.values.flatten(), size=(n_assets, steps))`