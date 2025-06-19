import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def shannons_demon_with_conditional_rebalance(stock_returns):
    """
    Simulate Shannon's Demon with conditional rebalancing based on a 20% appreciation or decrease.
    """
    # Initialize wealth, cash and stock position
    wealth = [1]
    cash = 0.5
    stock_position = 0.5
    
    initial_stock_position = stock_position
    
    for ret in stock_returns:
        # Update stock position due to returns
        stock_position *= ret
        
        # Update total wealth (cash + stock position)
        wealth.append(cash + stock_position)
        
        # Check for a 20% appreciation or decrease
        if stock_position <= initial_stock_position * 0.50:
            # Rebalance: 50% cash, 50% stock position
            cash = wealth[-1] * 0.5
            stock_position = wealth[-1] * 0.5
            initial_stock_position = stock_position

    return wealth

# Fetch AAPL data for the last 1000 trading days
data = yf.download("ARKK", period="5y")['Close'].tail(1000)

# Calculate daily returns
returns = data.pct_change().dropna() + 1

# Simulate and plot
wealth = shannons_demon_with_conditional_rebalance(returns)
plt.plot(wealth)
plt.title("Shannon's Demon with Conditional Rebalancing for AAPL")
plt.xlabel("Trading Days")
plt.ylabel("Wealth")
plt.show()
