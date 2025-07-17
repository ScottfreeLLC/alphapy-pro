# Shannon's Demon Visualization Gallery

This directory contains visualizations generated from Shannon's Demon analysis.

## Performance Charts

### `demon.png`
Main Shannon's Demon performance comparison chart showing:
- Portfolio value over time
- Buy-and-hold vs rebalancing strategy
- Cumulative returns comparison

### Random Walk Simulations
- **`GRW1.png`** - Geometric Random Walk Simulation #1
- **`GRW2.png`** - Geometric Random Walk Simulation #2  
- **`GRW3.png`** - Geometric Random Walk Simulation #3

These demonstrate the theoretical foundation of Shannon's Demon with synthetic data where assets randomly double or halve.

## Asset-Specific Analysis

### Cryptocurrency
- **`btc-usd.png`** - Bitcoin vs USD performance analysis
- **`xrp-usd.png`** - XRP vs USD performance analysis

### Individual Stocks
- **`tsla.png`** - Tesla (TSLA) analysis
- **`jnj.png`** - Johnson & Johnson (JNJ) analysis
- **`wmt.png`** - Walmart (WMT) analysis

Each asset chart typically shows:
- Price evolution over time
- Shannon's Demon rebalancing points
- Performance comparison with buy-and-hold
- Cumulative returns analysis

## How to Generate More Visualizations

### Using Scripts
```bash
# Generate basic simulation plots
python scripts/shannon_demon.py

# Generate geometric random walk plots
python scripts/random_walk.py

# Generate real market data analysis
python scripts/demon.py
```

### Using Jupyter Notebooks
```bash
# Comprehensive analysis with multiple assets
jupyter notebook demon.ipynb

# Step-by-step tutorial
jupyter notebook notebooks/01_basic_demo.ipynb
```

### Using AlphaPy
```bash
# Generate ML-enhanced analysis plots
python prepare_data.py --symbol BTC-USD
alphapy --config config/model.yml
# Results saved to runs/run_YYYYMMDD_HHMMSS/plots/
```

## Chart Types

### Performance Charts
- Portfolio value over time
- Cumulative returns comparison
- Drawdown analysis
- Risk-return scatter plots

### Rebalancing Analysis
- Rebalancing trigger points
- Allocation drift over time
- Trade frequency analysis
- Transaction cost impact

### Market Analysis
- Volatility regime changes
- Correlation analysis
- Asset price movements
- Volume analysis

### ML Enhancement (AlphaPy)
- Feature importance plots
- Model performance curves
- Prediction confidence
- Classification metrics

## Customization

To generate custom visualizations:

1. **Modify existing scripts** in `scripts/` directory
2. **Create new analysis** in Jupyter notebooks
3. **Use AlphaPy configuration** to generate ML-specific plots
4. **Add custom plotting code** using matplotlib/seaborn

## Example Analysis

The `demon.ipynb` notebook provides the most comprehensive analysis including:
- Dynamic asset selection
- Real-time rebalancing simulation
- Performance visualization
- Trading pair analysis (TQQQ/SQQQ, LABU/LABD, etc.)
- Interactive parameter adjustment

## Future Enhancements

Potential additions to the visualization gallery:
- Interactive dashboard using Plotly/Dash
- Real-time performance monitoring
- Multi-asset portfolio analysis
- Risk decomposition charts
- Regime-based performance analysis
- Monte Carlo simulation results