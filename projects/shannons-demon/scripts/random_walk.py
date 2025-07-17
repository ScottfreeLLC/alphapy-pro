import numpy as np
import matplotlib.pyplot as plt

def geometric_random_walk(initial_value, periods, prob=0.5):
    stock_prices = [initial_value]
    
    for _ in range(periods):
        # Simulate stock price change (double or halve)
        if np.random.rand() < prob:
            new_price = stock_prices[-1] * 2  # Stock price doubles
        else:
            new_price = stock_prices[-1] / 2  # Stock price halves
        stock_prices.append(new_price)
    
    return stock_prices

# Initial conditions
initial_value = 1  # Start the stock price at 1
periods = 100  # Number of periods to simulate

# Simulate geometric random walk
stock_prices = geometric_random_walk(initial_value, periods)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(stock_prices, marker='o', markersize=3, label='Stock Price')
plt.yscale('log')
plt.title("Geometric Random Walk of Stock Price")
plt.xlabel("Time")
plt.ylabel("Stock Price ($)")

# Manually set y-axis ticks
ticks = [0.01, 0.10, 1.00, 10.00, 100.00, 1000.00, 10000.00, 100000.00, 1000000.00, 10000000.00]
plt.gca().yaxis.set_major_locator(plt.FixedLocator(ticks))
plt.gca().yaxis.set_major_formatter(plt.FixedFormatter(['$0.01', '$0.10', '$1.00', '$10.00', '$100.00', '$1,000.00', '$10,000.00', '$100,000.00', '$1,000,000.00', '$10,000,000.00']))

plt.legend()
plt.grid(True)
plt.show()
