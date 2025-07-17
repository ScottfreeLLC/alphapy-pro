# Shannon's Demon Tutorial: A Complete Guide to Rebalancing Strategies

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Getting Started](#getting-started)
4. [Basic Implementation](#basic-implementation)
5. [Advanced Strategies](#advanced-strategies)
6. [AlphaPy Integration](#alphapy-integration)
7. [Performance Analysis](#performance-analysis)
8. [Machine Learning Enhancements](#machine-learning-enhancements)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Introduction

Shannon's Demon is a rebalancing strategy discovered by Claude Shannon, the father of information theory. This tutorial will guide you through understanding, implementing, and optimizing this powerful investment strategy using AlphaPy Pro.

### What is Shannon's Demon?

Shannon's Demon is a portfolio rebalancing strategy that can generate profits even from assets with zero expected return, as long as they exhibit volatility. The strategy works by maintaining a fixed allocation (typically 50/50) between two assets and periodically rebalancing to maintain this ratio.

### Why Use Shannon's Demon?

1. **Volatility Harvesting**: Profits from volatility rather than directional moves
2. **Risk Management**: Automatic selling high and buying low
3. **Simplicity**: Easy to understand and implement
4. **Adaptability**: Works with various asset classes

## Mathematical Foundation

### The Core Principle

Consider a portfolio with two assets:
- Asset A: Risky asset (e.g., stock, cryptocurrency)
- Asset B: Stable asset (e.g., cash, bonds)

Starting with 50% in each asset, when Asset A rises, its proportion increases. By rebalancing back to 50/50, you:
1. Sell some of Asset A (at a high price)
2. Buy more of Asset B

When Asset A falls, you:
1. Buy more of Asset A (at a low price)
2. Sell some of Asset B

### Mathematical Proof

For a geometric random walk where an asset has equal probability of going up by factor `u` or down by factor `d` where `u = 1/d`:

- Buy and Hold return: `E[R] = 0.5 * u + 0.5 * d = 0.5 * (u + 1/u)`
- Shannon's Demon return: `E[R] = sqrt(u * d) = 1`

When `u > 1`, Shannon's Demon outperforms buy-and-hold even with zero expected return!

### Key Formula

Portfolio value after rebalancing:
```
V_new = V_old * sqrt(1 + r^2)
```
Where `r` is the return of the risky asset.

## Getting Started

### Prerequisites

1. Install AlphaPy Pro:
```bash
pip install -e .
```

2. Navigate to the Shannon's Demon project:
```bash
cd projects/shannons-demon
```

3. Review available data files:
```bash
ls data/
```

### Project Structure

```
shannons-demon/
├── config/
│   └── model.yml          # AlphaPy configuration
├── data/                  # Market data (CSV files)
│   ├── BTC-USD.csv       # Bitcoin data
│   ├── ETH-USD.csv       # Ethereum data
│   ├── AAPL.csv          # Apple stock data
│   └── ...               # 30+ other assets
├── scripts/               # Original implementations
│   ├── shannon_demon.py   # Core implementation
│   ├── demon.py          # Real market data analysis
│   ├── random_walk.py    # Geometric random walk
│   ├── shannon_demon_threshold.py  # Threshold-based strategy
│   ├── shannon_demon_50.py        # 50/50 rebalancing
│   ├── shannon_demon_aapl.py      # Apple-specific
│   ├── spread.py         # Spread analysis
│   └── match_dates.py    # Date alignment utility
├── notebooks/             # Tutorial notebooks
│   ├── 01_basic_demo.ipynb
│   ├── 02_real_data.ipynb
│   └── 03_ml_enhanced.ipynb
├── demon.ipynb           # Original comprehensive analysis
├── prepare_data.py       # Data preparation for AlphaPy
├── TUTORIAL.md           # This file
├── README.md            # Comprehensive documentation
└── README_ALPHAPY.md    # Quick start guide
```

### Multiple Learning Paths

This project offers several ways to learn Shannon's Demon:

1. **📚 Theory First**: Start with this tutorial for mathematical foundations
2. **🔬 Interactive Analysis**: Open `demon.ipynb` for comprehensive market analysis
3. **🎯 Hands-On**: Run `scripts/shannon_demon.py` for immediate results
4. **🤖 ML Enhanced**: Use `prepare_data.py` + `alphapy` for advanced strategies

## Basic Implementation

### Step 1: Simple Simulation

Start with the basic Shannon's Demon simulation using synthetic data:

```python
# Run the basic simulation
python scripts/shannon_demon.py
```

This creates a random walk simulation and compares:
- Buy and hold strategy
- Shannon's Demon rebalancing strategy

**Alternative**: For a more detailed geometric random walk simulation:
```python
python scripts/random_walk.py
```

### Step 2: Real Market Data

Test with actual market data using the comprehensive analysis script:

```python
# Run comprehensive market analysis
python scripts/demon.py
```

This script analyzes multiple assets and generates:
- Performance comparisons
- Rebalancing signals
- Correlation analysis
- Risk metrics

**Interactive Analysis**: For the most comprehensive analysis, open the Jupyter notebook:
```bash
jupyter notebook demon.ipynb
```

This notebook includes:
- Dynamic asset selection
- Real-time rebalancing simulation
- Performance visualization
- Trading pair analysis (TQQQ/SQQQ, LABU/LABD, etc.)

### Step 3: Specialized Strategies

Explore different rebalancing approaches:

```python
# Threshold-based rebalancing
python scripts/shannon_demon_threshold.py

# 50/50 rebalancing strategy
python scripts/shannon_demon_50.py

# Apple-specific implementation
python scripts/shannon_demon_aapl.py

# Spread analysis between assets
python scripts/spread.py
```

### Step 4: Visualization

The scripts automatically generate performance charts showing:
- Portfolio value over time
- Asset allocation changes
- Rebalancing points
- Cumulative returns comparison

## Advanced Strategies

### Threshold-Based Rebalancing

Instead of rebalancing at fixed intervals, rebalance when allocation deviates by a threshold:

```python
# Run threshold-based strategy
python shannon_demon_threshold.py --threshold 0.2
```

This rebalances when allocation moves beyond 60/40 or 40/60.

### Multi-Asset Rebalancing

Extend to multiple assets:

```python
# Example: 3-asset portfolio
weights = [0.33, 0.33, 0.34]  # Equal weight
assets = ['AAPL', 'MSFT', 'CASH']
```

### Transaction Cost Modeling

Include realistic transaction costs:

```python
# Add transaction costs
transaction_cost = 0.001  # 0.1% per trade
slippage = 0.0005        # 0.05% slippage
```

## AlphaPy Integration

Shannon's Demon is now fully integrated with AlphaPy Pro's machine learning pipeline. This integration provides:

1. **Automated Feature Engineering**: AlphaPy automatically generates technical indicators, lag features, and portfolio-specific variables
2. **Multi-Algorithm Training**: Train multiple ML models simultaneously with automated hyperparameter tuning
3. **Comprehensive Evaluation**: Built-in performance metrics, plots, and backtesting capabilities
4. **Configuration-Driven**: All strategy parameters controlled through YAML files

### Data Preparation

First, prepare your data for AlphaPy training:

```bash
# Prepare Bitcoin data for ML training
python prepare_data.py --symbol BTC-USD --start 2018-01-01 --end 2023-12-31

# This creates train.csv and test.csv files with:
# - Market data (OHLCV)
# - Portfolio features (weight deviation, profit potential)
# - Technical indicators (RSI, volatility, returns)
# - Target variable (rebalance_signal)
```

### Configuration Files

The AlphaPy integration uses several configuration files:

**`config/model.yml`** - Project-specific ML configuration:
- Classification model to predict rebalancing signals
- Time series options for proper temporal handling
- Multiple algorithms (XGBoost, LightGBM, Random Forest)
- Feature engineering and selection options

**Global configurations** (in main config/ directory):
- `variables.yml` - Shannon's Demon specific variables and signals
- `systems.yml` - Trading system signal definitions
- `algos.yml` - Algorithm parameters and grid search ranges

### Running AlphaPy

```bash
# Navigate to the Shannon's Demon project
cd projects/shannons-demon

# Run AlphaPy with Shannon's Demon configuration
alphapy --config config/model.yml

# Or use the market flow pipeline
mflow --config config/model.yml
```

### AlphaPy Output

AlphaPy generates comprehensive results in the `runs/` directory:

```
runs/run_YYYYMMDD_HHMMSS/
├── config/         # Configuration snapshot
├── input/          # Copy of input data
├── model/          # Trained models (.pkl files)
├── output/         # Predictions and metrics
└── plots/          # Performance visualizations
```

### Key Features

1. **Automated Feature Generation**: 
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Lag features for time series
   - Portfolio-specific variables (weight deviation, profit potential)

2. **Multi-Algorithm Training**:
   - XGBoost, LightGBM, Random Forest, AdaBoost, Gradient Boosting
   - Automated hyperparameter tuning via grid search
   - Model ensembling and blending

3. **Comprehensive Evaluation**:
   - Cross-validation with proper time series handling
   - Feature importance analysis (including LOFO)
   - Performance metrics (F1, ROC-AUC, Precision, Recall)
   - Calibration plots and learning curves

4. **Configuration-Driven Strategy**:
   - All parameters in YAML files
   - Easy to modify without code changes
   - Experiment tracking and reproducibility

## Performance Analysis

### Key Metrics

1. **Sharpe Ratio**: Risk-adjusted returns
2. **Maximum Drawdown**: Largest peak-to-trough decline
3. **Calmar Ratio**: Annual return / Max drawdown
4. **Win Rate**: Percentage of profitable rebalances

### Backtesting Framework

```python
from alphapy.portfolio import Portfolio
from alphapy.analysis import calculate_metrics

# Run backtest
results = backtest_shannon_demon(
    data=price_data,
    rebalance_freq='daily',
    threshold=0.2,
    transaction_cost=0.001
)

# Calculate metrics
metrics = calculate_metrics(results)
print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

## Machine Learning Enhancements

### Feature Engineering

Create features to predict optimal rebalancing times:

```python
features = {
    'volatility': rolling_std(returns, window=20),
    'momentum': returns.rolling(10).mean(),
    'rsi': calculate_rsi(prices, period=14),
    'volume_ratio': volume / volume.rolling(20).mean()
}
```

### ML-Driven Rebalancing

Train models to predict:
1. **Optimal rebalancing times**
2. **Dynamic threshold adjustments**
3. **Asset allocation weights**

```python
# Train rebalancing classifier
from alphapy.model import train_model

model = train_model(
    features=features,
    target=optimal_rebalance_signal,
    algorithm='xgboost'
)
```

### Ensemble Strategies

Combine multiple approaches:
1. Fixed-interval rebalancing
2. Threshold-based rebalancing  
3. ML-predicted rebalancing

## Best Practices

### 1. Asset Selection
- Choose assets with low correlation
- Ensure adequate liquidity
- Consider tax implications

### 2. Rebalancing Frequency
- Daily: High transaction costs, captures more volatility
- Weekly/Monthly: Lower costs, may miss opportunities
- Threshold-based: Adaptive to market conditions

### 3. Risk Management
- Set maximum position sizes
- Implement stop-loss rules
- Monitor correlation changes

### 4. Performance Monitoring
- Track all rebalancing events
- Calculate rolling performance metrics
- Compare against benchmarks

## Troubleshooting

### Common Issues

1. **No Profit Despite Volatility**
   - Check transaction costs
   - Verify rebalancing is occurring
   - Ensure assets are truly volatile

2. **Poor Performance vs Buy-and-Hold**
   - May occur in strong trending markets
   - Consider adaptive strategies
   - Check rebalancing frequency

3. **Data Issues**
   - Ensure date alignment between assets
   - Handle missing data appropriately
   - Verify data quality

### Debugging Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Track rebalancing events
rebalance_log = []
for date, action in rebalancing_events:
    rebalance_log.append({
        'date': date,
        'action': action,
        'portfolio_value': portfolio.value
    })
```

## Next Steps

1. **Experiment with Different Assets**
   - Try cryptocurrency pairs
   - Test with sector ETFs
   - Explore international markets

2. **Optimize Parameters**
   - Use grid search for thresholds
   - Test various rebalancing frequencies
   - Find optimal asset allocations

3. **Extend the Strategy**
   - Add more assets (3+ asset portfolios)
   - Implement dynamic weighting
   - Create adaptive thresholds

4. **Production Deployment**
   - Set up automated execution
   - Implement real-time monitoring
   - Create performance dashboards

## Resources

- [Original Shannon's Demon Paper](https://www.jstor.org/stable/2333242)
- [AlphaPy Documentation](../../docs/index.html)
- [Example Notebooks](./notebooks/)
- [Community Forum](https://github.com/alphapy/alphapy/discussions)

## Conclusion

Shannon's Demon demonstrates how mathematical principles can create profitable trading strategies. By combining this classic approach with modern machine learning through AlphaPy Pro, you can develop sophisticated rebalancing strategies that adapt to changing market conditions.

Start with the basic implementation, understand the mechanics, then gradually add complexity through ML enhancements and multi-asset portfolios. Remember: the key to Shannon's Demon is discipline in rebalancing, not market timing.

Happy rebalancing!