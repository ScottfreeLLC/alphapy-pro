import numpy as np
import matplotlib.pyplot as plt

def shannons_demon(steps=1000, drift=0, volatility=0.02, seed=None):
    """
    Simulate Shannon's Demon with two uncorrelated assets.
    """
    np.random.seed(seed)
    
    # Generate random returns for two uncorrelated assets
    returns_1 = np.exp(drift + np.random.randn(steps) * volatility)
    returns_2 = np.exp(drift - np.random.randn(steps) * volatility)
    
    # Initialize wealth and allocation
    wealth = [1]
    fraction_1 = 0.5
    fraction_2 = 0.5
    
    for i in range(1, steps):
        # Update wealth due to returns
        wealth.append(wealth[-1] * (fraction_1 * returns_1[i] + fraction_2 * returns_2[i]))
        
        # Rebalance
        fraction_1 = 0.5
        fraction_2 = 0.5

    return wealth

# Simulate and plot
wealth = shannons_demon(seed=42)
plt.plot(wealth)
plt.title("Shannon's Demon Simulation")
plt.xlabel("Time Steps")
plt.ylabel("Wealth")
plt.show()
