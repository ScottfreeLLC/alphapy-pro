# Shannon's Demon Trading Strategy

## Overview

This project implements Shannon's Demon (also known as Shannon's Rebalancing Strategy), a mathematical trading strategy inspired by Claude Shannon's work on portfolio rebalancing. The strategy demonstrates how systematic rebalancing between uncorrelated or negatively correlated assets can generate positive returns even when individual assets have zero expected return.

## Objective

The project explores:
- The mathematical principles behind Shannon's Demon
- Implementation of rebalancing strategies with real market data
- Performance analysis across various asset pairs including stocks, ETFs, and cryptocurrencies
- Comparison of different rebalancing thresholds and frequencies

## Core Concept

Shannon's Demon works by:
1. Maintaining a target allocation between two assets (typically 50/50)
2. Rebalancing when asset values diverge from the target
3. Systematically "buying low and selling high" through rebalancing
4. Harvesting volatility as a source of returns

## Project Components

### Python Scripts

- **shannon_demon.py**: Core implementation of Shannon's Demon with simulated assets
- **demon.py**: Extended implementation with real market data analysis
- **random_walk.py**: Geometric random walk simulations for theoretical analysis
- **shannon_demon_50.py**: 50/50 rebalancing strategy implementation
- **shannon_demon_aapl.py**: Apple stock specific implementation
- **shannon_demon_threshold.py**: Threshold-based rebalancing strategy
- **spread.py**: Spread analysis between correlated assets
- **match_dates.py**: Utility for aligning time series data

### Jupyter Notebook

- **demon.ipynb**: Interactive analysis and visualization of Shannon's Demon strategies

### Data Files

The `data/` directory contains historical price data for various assets:

**Stocks**: AAPL, JNJ, TSLA, WMT, AFRM, MGRX, SPRT, and others

**ETFs**: 
- SMH (Semiconductors)
- SOXL/SOXS (3x Semiconductor Bull/Bear)
- TQQQ/SQQQ (3x NASDAQ Bull/Bear)
- TNA/TZA (3x Small Cap Bull/Bear)
- LABU/LABD (3x Biotech Bull/Bear)
- UVXY (Volatility)

**Cryptocurrencies**: BTC-USD, ETH-USD, ADA-USD, DOGE-USD, and many others

**Market Indicators**: DX-Y.NYB (Dollar Index), ^VIX (Volatility Index)

## How to Run

### Basic Simulation

```bash
# Run basic Shannon's Demon simulation
python shannon_demon.py
```

### Real Market Data Analysis

```bash
# Run the main demon analysis with market data
python demon.py
```

### Interactive Analysis

```bash
# Launch Jupyter notebook for interactive exploration
jupyter notebook demon.ipynb
```

### Specific Strategies

```bash
# Run threshold-based rebalancing
python shannon_demon_threshold.py

# Analyze specific asset (e.g., Apple)
python shannon_demon_aapl.py

# Generate random walk simulations
python random_walk.py
```

## Strategy Variations

### 1. Fixed Time Rebalancing
Rebalance at regular intervals (daily, weekly, monthly) regardless of price movements.

### 2. Threshold Rebalancing
Only rebalance when allocation deviates beyond a certain threshold (e.g., 60/40 or 70/30).

### 3. Volatility-Adjusted
Adjust rebalancing frequency based on asset volatility.

## Expected Results

The strategy typically performs best with:
- **High volatility** assets
- **Mean-reverting** price behavior
- **Low or negative correlation** between assets
- **Low transaction costs**

Performance characteristics:
- Reduces portfolio volatility compared to individual assets
- Captures gains from volatility ("volatility harvesting")
- May underperform in strong trending markets
- Outperforms in choppy, sideways markets

## Visualization Outputs

The scripts generate various plots including:
- **GRW1.png, GRW2.png, GRW3.png**: Geometric random walk simulations
- **demon.png**: Shannon's Demon performance visualization
- Asset-specific charts (btc-usd.png, tsla.png, etc.)

## Mathematical Foundation

The strategy exploits the mathematical property that:
- Geometric mean ≤ Arithmetic mean
- Rebalancing forces selling winners and buying losers
- This creates a "volatility pump" that extracts value from price fluctuations

## Risk Considerations

1. **Transaction Costs**: Frequent rebalancing can erode returns
2. **Tax Implications**: Rebalancing triggers taxable events
3. **Correlation Changes**: Strategy assumes stable correlation patterns
4. **Black Swan Events**: Extreme moves can cause significant losses

## Tips for Implementation

1. **Asset Selection**: Choose assets with different risk/return profiles
2. **Rebalancing Frequency**: Balance between capturing volatility and minimizing costs
3. **Position Sizing**: Consider using a small portion of portfolio for testing
4. **Backtesting**: Always test strategies on historical data before live trading
5. **Risk Management**: Set stop-losses and maximum position sizes

## Further Research

- Extend to multi-asset portfolios (N > 2 assets)
- Incorporate momentum indicators for timing
- Optimize rebalancing thresholds using machine learning
- Compare with other portfolio strategies (buy-and-hold, momentum, etc.)