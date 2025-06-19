import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_stock_prices(initial_value, periods, drift=0.001, volatility=0.01):
    returns = np.random.normal(drift, volatility, periods)
    price_series = initial_value * np.exp(np.cumsum(returns))
    return price_series

def generate_spread_signals(stock1, stock2, window=30, threshold=2):
    spread = stock1 - stock2
    mean_spread = spread.rolling(window=window).mean()
    std_spread = spread.rolling(window=window).std()
    
    z_score = (spread - mean_spread) / std_spread
    long_signal = z_score < -threshold
    short_signal = z_score > threshold
    
    return spread, mean_spread, long_signal, short_signal

def simulate_trading(stock1, stock2, long_signal, short_signal):
    positions1 = np.zeros(len(stock1))
    positions2 = np.zeros(len(stock2))
    
    positions1[long_signal] = 1
    positions1[short_signal] = -1
    
    positions2[long_signal] = -1
    positions2[short_signal] = 1
    
    returns1 = np.diff(stock1) / stock1[:-1]
    returns2 = np.diff(stock2) / stock2[:-1]
    
    portfolio_returns = positions1[:-1] * returns1 + positions2[:-1] * returns2
    portfolio_value = np.cumprod(1 + portfolio_returns)
    
    return portfolio_value

# Initial conditions
initial_value = 100
periods = 252  # Simulating one year of daily data
drift = 0.001
volatility = 0.02

# Simulate stock prices
np.random.seed(42)
stock1 = simulate_stock_prices(initial_value, periods, drift, volatility)
stock2 = simulate_stock_prices(initial_value, periods, drift, volatility) + np.random.normal(0, 2, periods)

# Generate spread signals
spread, mean_spread, long_signal, short_signal = generate_spread_signals(pd.Series(stock1), pd.Series(stock2))

# Simulate trading
portfolio_value = simulate_trading(stock1, stock2, long_signal, short_signal)

# Plot the results
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(stock1, label='Stock 1')
plt.plot(stock2, label='Stock 2')
plt.plot(spread, label='Spread', linestyle='--')
plt.plot(mean_spread, label='Mean Spread', linestyle=':')
plt.fill_between(range(periods), mean_spread - 2 * spread.rolling(30).std(), mean_spread + 2 * spread.rolling(30).std(), color='gray', alpha=0.2)
plt.legend()
plt.title('Stock Prices and Spread')

plt.subplot(2, 1, 2)
plt.plot(portfolio_value, label='Portfolio Value')
plt.legend()
plt.title('Portfolio Value Over Time')

plt.tight_layout()
plt.show()
