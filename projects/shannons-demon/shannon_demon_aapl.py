import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def shannons_demon_with_cash(stock_returns):
    """
    Simulate Shannon's Demon with cash and a stock based on actual stock returns.
    """
    # Initialize wealth, cash and stock position
    wealth = [1]
    cash = 0.5
    stock_position = 0.5
    
    for ret in stock_returns:
        # Update stock position due to returns
        stock_position *= ret
        
        # Update total wealth (cash + stock position)
        wealth.append(cash + stock_position)
        
        # Rebalance: 50% cash, 50% stock position
        cash = wealth[-1] * 0.5
        stock_position = wealth[-1] * 0.5

    return wealth

# Fetch symbol data for the last 1000 trading days
symbol = "JNJ"
data = yf.download(symbol, period="5y")['Close'].tail(1000)

# Calculate daily returns
returns = data.pct_change().dropna() + 1

# Simulate and plot
wealth = shannons_demon_with_cash(returns)
plt.plot(wealth)
plt.title(f"Shannon's Demon with 50% Cash Rebalancing for {symbol}")
plt.xlabel("Trading Days")
plt.ylabel("Wealth")
plt.show()
