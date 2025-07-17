import numpy as np
import matplotlib.pyplot as plt

def shannons_demon_with_cash(steps=1000, drift=0, volatility=0.02, seed=None):
    """
    Simulate Shannon's Demon with cash and a stock.
    """
    np.random.seed(seed)
    
    # Generate random returns for the stock
    stock_returns = np.exp(drift + np.random.randn(steps) * volatility)
    
    # Initialize wealth, cash and stock position
    wealth = [1]
    cash = 0.5
    stock_position = 0.5
    
    for i in range(1, steps):
        # Update stock position due to returns
        stock_position *= stock_returns[i]
        
        # Update total wealth (cash + stock position)
        wealth.append(cash + stock_position)
        
        # Rebalance: 50% cash, 50% stock position
        cash = wealth[-1] * 0.5
        stock_position = wealth[-1] * 0.5

    return wealth

# Simulate and plot
wealth = shannons_demon_with_cash(volatility=0.1, seed=42)
plt.plot(wealth)
plt.title("Shannon's Demon with 50% Cash Rebalancing")
plt.xlabel("Time Steps")
plt.ylabel("Wealth")
plt.show()
