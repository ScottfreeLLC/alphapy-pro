import yfinance as yf

# Define the ticker symbol
ticker = "GME"

# Get the stock data
stock = yf.Ticker(ticker)

# Get historical market data for the past month
historical_data = stock.history(period="1mo")

# Calculate the average volume
average_volume = historical_data['Volume'].mean()

# Fetch and display fundamental data
shares_outstanding = stock.info.get('sharesOutstanding')
market_cap = stock.info.get('marketCap')
pe_ratio = stock.info.get('trailingPE')
free_float = stock.info.get('floatShares')

print(f"\nFundamental Data for {ticker}:")
print(f"Shares Outstanding: {shares_outstanding}")
print(f"Market Cap: {market_cap}")
print(f"Price/Earnings Ratio (P/E): {pe_ratio}")
print(f"Free Float: {free_float}")
print(f"Average Daily Volume (past month): {average_volume}")
