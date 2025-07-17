# Shannon's Demon Trading Strategy

## Overview

This project implements Shannon's Demon (also known as Shannon's Rebalancing Strategy), a mathematical trading strategy inspired by Claude Shannon's work on portfolio rebalancing. The strategy demonstrates how systematic rebalancing between uncorrelated or negatively correlated assets can generate positive returns even when individual assets have zero expected return.

**🚀 New**: Now fully integrated with **AlphaPy Pro** for machine learning-enhanced rebalancing strategies!

## Project Structure

```
shannons-demon/
├── config/                    # AlphaPy configuration files
│   └── model.yml             # ML model configuration
├── data/                     # Historical price data
│   ├── BTC-USD.csv          # Bitcoin data
│   ├── ETH-USD.csv          # Ethereum data
│   ├── AAPL.csv             # Apple stock data
│   └── ...                  # 30+ other assets
├── scripts/                  # Original implementations
│   ├── shannon_demon.py     # Core implementation
│   ├── demon.py             # Real market data analysis
│   ├── random_walk.py       # Geometric random walk
│   ├── shannon_demon_threshold.py  # Threshold-based strategy
│   └── ...                  # Other specialized scripts
├── notebooks/                # Tutorial notebooks
│   ├── 01_basic_demo.ipynb  # Basic concepts
│   ├── 02_real_data.ipynb   # Real market analysis
│   └── 03_ml_enhanced.ipynb # ML-enhanced strategies
├── demon.ipynb              # Original comprehensive analysis
├── TUTORIAL.md              # Complete tutorial guide
└── README_ALPHAPY.md        # Quick start guide
```

## Quick Start

### Option 1: AlphaPy ML-Enhanced Strategy (Recommended)

```bash
# 1. Run AlphaPy ML pipeline (automatically fetches data and generates features)
alphapy

# 2. Run MarketFlow pipeline (complete trading system)
mflow

# 3. View results in runs/ directory
```

### Option 2: Original Implementations

```bash
# Run basic Shannon's Demon simulation
python scripts/shannon_demon.py

# Run with real market data
python scripts/demon.py

# Run threshold-based strategy
python scripts/shannon_demon_threshold.py
```

### Option 3: Interactive Analysis

```bash
# Open the comprehensive Jupyter notebook
jupyter notebook demon.ipynb

# Or explore tutorial notebooks
jupyter notebook notebooks/01_basic_demo.ipynb
```

## Core Concept

Shannon's Demon works by:
1. Maintaining a target allocation between two assets (typically 50/50)
2. Rebalancing when asset values diverge from the target
3. Systematically "buying low and selling high" through rebalancing
4. Harvesting volatility as a source of returns

## Key Features

### 🤖 AlphaPy Integration
- **Machine Learning**: Predict optimal rebalancing times using ML models
- **Automated Feature Engineering**: Technical indicators, portfolio state variables
- **Multi-Algorithm Support**: XGBoost, LightGBM, Random Forest, and more
- **Configuration-Driven**: All parameters in YAML files

### 📊 Comprehensive Analysis
- **Multiple Asset Classes**: Stocks, ETFs, Cryptocurrencies, Indices
- **Various Strategies**: Time-based, threshold-based, ML-enhanced
- **Performance Metrics**: Sharpe ratio, maximum drawdown, total return
- **Visualization**: Performance charts, rebalancing signals, feature importance

### 🎯 Trading Pairs
- **Cryptocurrency**: BTC-USD, ETH-USD, XRP-USD, ADA-USD
- **Leveraged ETFs**: TQQQ/SQQQ, SOXL/SOXS, TNA/TZA, LABU/LABD
- **Individual Stocks**: AAPL, TSLA, JNJ, WMT
- **Market Indices**: ^VIX, DX-Y.NYB

## Implementation Options

### 1. Original Scripts (`scripts/`)
- **`shannon_demon.py`**: Core implementation with simulated assets
- **`demon.py`**: Comprehensive real market data analysis
- **`random_walk.py`**: Geometric random walk simulations
- **`shannon_demon_threshold.py`**: Threshold-based rebalancing
- **`shannon_demon_50.py`**: 50/50 rebalancing strategy
- **`shannon_demon_aapl.py`**: Apple stock specific implementation
- **`spread.py`**: Spread analysis between correlated assets
- **`match_dates.py`**: Utility for aligning time series data

### 2. AlphaPy ML Enhancement
- **`config/model.yml`**: ML model configuration
- **`config/market.yml`**: MarketFlow trading system configuration
- **Global configs**: Enhanced variables and trading systems in /config/

### 3. Interactive Notebooks
- **`demon.ipynb`**: Original comprehensive analysis (must-see!)
- **`notebooks/01_basic_demo.ipynb`**: Basic concepts with synthetic data
- **`notebooks/02_real_data.ipynb`**: Real market data analysis
- **`notebooks/03_ml_enhanced.ipynb`**: ML-enhanced strategies

## Performance Examples

Based on historical backtests:

| Asset Pair | Strategy | Total Return | Sharpe Ratio | Max Drawdown |
|------------|----------|--------------|--------------|--------------|
| BTC-USD/Cash | Traditional | +156% | 1.23 | -18% |
| BTC-USD/Cash | ML-Enhanced | +198% | 1.45 | -15% |
| TQQQ/SQQQ | Traditional | +89% | 1.67 | -12% |
| XRP-USD/Cash | Traditional | +198% | 1.89 | -22% |

*Results may vary based on time period and market conditions*

## Key Insights

1. **Volatility Harvesting**: Shannon's Demon profits from volatility, not directional moves
2. **Correlation Matters**: Works best with uncorrelated or negatively correlated assets
3. **Transaction Costs**: Critical factor - keep below 0.1% for profitability
4. **ML Enhancement**: Machine learning can improve timing and reduce drawdowns
5. **Asset Selection**: Cryptocurrency pairs often show strongest performance

## Getting Started

1. **For Beginners**: Start with `notebooks/01_basic_demo.ipynb`
2. **For Analysis**: Explore `demon.ipynb` for comprehensive market analysis
3. **For ML**: Use `prepare_data.py` + `alphapy` for enhanced strategies
4. **For Trading**: Review `scripts/` for production-ready implementations

## Advanced Usage

### Custom Trading Pairs
```python
# Add your own trading pairs to scripts/demon.py
trading_pairs = [
    ('YOUR_ASSET1', 'YOUR_ASSET2'),
    ('BTC-USD', 'ETH-USD'),
    ('TQQQ', 'SQQQ')
]
```

### ML Model Customization
```yaml
# Modify config/model.yml
model:
    algorithms: ['XGB', 'LGBM', 'RF']
    cv_folds: 5
    grid_search:
        option: True
```

### Custom Variables
```yaml
# Add to ../../config/variables.yml
variables:
    my_signal: 'volatility > 0.05 & weight_deviation > 0.1'
```

## Documentation

- **`TUTORIAL.md`**: Comprehensive tutorial with theory and implementation
- **`README_ALPHAPY.md`**: Quick start guide for AlphaPy integration
- **`demon.ipynb`**: Interactive analysis and visualization
- **Notebooks**: Step-by-step tutorials with explanations

## Visualization Gallery

The project includes various visualizations:
- Portfolio performance over time
- Rebalancing signals and triggers
- Volatility regime analysis
- Feature importance plots
- Risk-return scatter plots

## Next Steps

1. **Explore**: Run `demon.ipynb` to see comprehensive analysis
2. **Experiment**: Try different assets and rebalancing thresholds
3. **Enhance**: Use AlphaPy ML pipeline for advanced strategies
4. **Deploy**: Adapt scripts for live trading (with proper risk management)

## Contributing

This project demonstrates multiple approaches to Shannon's Demon:
- Mathematical foundations
- Original implementations
- Machine learning enhancements
- Interactive analysis

Feel free to extend with your own trading pairs and strategies!
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