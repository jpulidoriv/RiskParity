import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download(["SPY", "TLT", "GLD"], start = "2015-01-01")

prices = data["Close"].dropna()
logReturns = np.log(prices / prices.shift(1))
logReturns = logReturns.dropna()

avgDailyReturnSPY = logReturns["SPY"].mean()
avgAnnualReturnSPY = (avgDailyReturnSPY * 252)
volDailySPY = logReturns["SPY"].std()
volAnnualSPY = (volDailySPY * np.sqrt(252))

avgDailyReturnTLT = logReturns["TLT"].mean()
avgAnnualReturnTLT = (avgDailyReturnTLT * 252)
volDailyTLT = logReturns["TLT"].std()
volAnnualTLT = (volDailyTLT * np.sqrt(252))

avgDailyReturnGLD = logReturns["GLD"].mean()
avgAnnualReturnGLD = (avgDailyReturnGLD * 252)
volDailyGLD = logReturns["GLD"].std()
volAnnualGLD = (volDailyGLD * np.sqrt(252))

covMatrix = logReturns.cov()
corrMatrix = logReturns.corr()

#Portfolio 100% SPY
w_spy = np.array([0.0, 1.0, 0.0])

#Portfolio Naive
w_naive = np.array([1/3, 1/3, 1/3])

#Portfolio Volatility Parity
volDaily = logReturns.std()

inv_vol = 1 / volDaily
#Normalizamos cuando dividimos todos por su suma
w_volpar = inv_vol / inv_vol.sum()
w_volpar = w_volpar.values

port_spy = logReturns.dot(w_spy)
port_naive = logReturns.dot(w_naive)
port_volpar = logReturns.dot(w_volpar)

portfolios_log = pd.DataFrame ({
    "SP500": port_spy,
    "Naive": port_naive,
    "VolParity": port_volpar,
})

annualReturns = (portfolios_log.mean() * 252)
annualVolatility = (portfolios_log.std() * np.sqrt(252))
sharpe = annualReturns / annualVolatility
portfolio_equity = np.exp(portfolios_log.cumsum())

def max_drawdown(equity_df):
    roll_max = equity_df.cummax()
    dd = equity_df / roll_max - 1.0
    return dd.min()  

mdd = max_drawdown(portfolio_equity)

metrics = pd.DataFrame({
    "Annual Return (%)": annualReturns * 100,
    "Annual Vol (%)":    annualVolatility * 100,
    "Sharpe":            sharpe,
    "Max DD (%)":        mdd * 100
})

plt.figure(figsize=(10, 6))

for col in portfolio_equity.columns:
    plt.plot(portfolio_equity.index, portfolio_equity[col], label=col)

plt.title("Equity Curves – SP500 vs Naive vs Volatility Parity")
plt.xlabel("Date")
plt.ylabel("Equity)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

profitability = ((portfolio_equity - 1) * 100)
profitability = profitability.tail(1)
print (profitability)
print (metrics.round(2))
print(logReturns.columns)
