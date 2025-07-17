# Shannon's Demon with AlphaPy Pro

Shannon's Demon rebalancing strategy enhanced with machine learning using AlphaPy Pro.

## Quick Start

### 1. Prepare Data

```bash
# Prepare Bitcoin data for ML training
python prepare_data.py --symbol BTC-USD --start 2018-01-01 --end 2023-12-31
```

This creates `train.csv` and `test.csv` with:
- Market data (OHLCV)
- Portfolio features (weight deviation, profit potential)  
- Technical indicators (RSI, volatility, returns)
- Target variable (`rebalance_signal`)

### 2. Run AlphaPy

```bash
# Run ML pipeline
alphapy --config config/model.yml

# Or use MarketFlow pipeline
mflow --config config/model.yml
```

### 3. View Results

Results are saved in `runs/run_YYYYMMDD_HHMMSS/`:
- `model/` - Trained models
- `output/` - Predictions and metrics
- `plots/` - Performance visualizations

## Configuration

The strategy is configured through YAML files:

- **`config/model.yml`** - Project-specific ML configuration
- **`../../config/variables.yml`** - Shannon's Demon variables
- **`../../config/systems.yml`** - Trading signals
- **`../../config/algos.yml`** - Algorithm parameters

## Key Features

### AlphaPy Integration
- **Automated Feature Engineering**: Technical indicators, lag features, portfolio variables
- **Multi-Algorithm Training**: XGBoost, LightGBM, Random Forest, AdaBoost, Gradient Boosting
- **Comprehensive Evaluation**: Cross-validation, feature importance, performance metrics
- **Configuration-Driven**: All parameters in YAML files

### Shannon's Demon Variables
- `weight_deviation` - Portfolio weight deviation from target
- `profit_potential` - Estimated profit from rebalancing
- `volatility_regime` - Market volatility classification
- `time_since_rebalance` - Days since last rebalance

### Trading Signals
- `shannon_basic` - Traditional threshold-based signals
- `shannon_profit` - Profit-optimized signals
- `shannon_regime` - Volatility regime-aware signals

## Model Performance

The ML model predicts when to rebalance based on:
- Market conditions (volatility, momentum)
- Portfolio state (weight deviation, time since rebalance)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Historical profitability patterns

## Example Workflow

1. **Data Preparation**: Generate training data with rebalancing labels
2. **Feature Engineering**: AlphaPy creates technical indicators and lag features
3. **Model Training**: Train multiple algorithms with hyperparameter tuning
4. **Evaluation**: Assess model performance with cross-validation
5. **Prediction**: Generate rebalancing signals for new data
6. **Backtesting**: Evaluate strategy performance vs buy-and-hold

## Advanced Usage

### Custom Variables

Add custom variables to `../../config/variables.yml`:

```yaml
variables:
    my_signal: 'volatility > 0.05 & weight_deviation > 0.1'
```

### Custom Systems

Define trading systems in `../../config/systems.yml`:

```yaml
my_system:
    signal_long: my_signal & profitsig
    signal_short: my_signal & ~profitsig
```

### Parameter Tuning

Modify algorithm parameters in `../../config/algos.yml`:

```yaml
XGB:
    grid:
        n_estimators: [100, 200, 300]
        max_depth: [3, 5, 7]
        learning_rate: [0.01, 0.1, 0.2]
```

## Notebooks

Interactive analysis notebooks:
- `notebooks/01_basic_demo.ipynb` - Basic Shannon's Demon concepts
- `notebooks/02_real_data.ipynb` - Real market data analysis
- `notebooks/03_ml_enhanced.ipynb` - ML enhancement with AlphaPy

## Documentation

- `TUTORIAL.md` - Comprehensive tutorial
- `README.md` - Technical documentation
- `prepare_data.py` - Data preparation script

## Performance Metrics

The strategy is evaluated using:
- **Classification Metrics**: F1, ROC-AUC, Precision, Recall
- **Trading Metrics**: Sharpe Ratio, Maximum Drawdown, Total Return
- **Feature Importance**: LOFO and built-in importance scores
- **Calibration**: Prediction confidence assessment

## Next Steps

1. Experiment with different assets and time periods
2. Tune hyperparameters for your specific use case
3. Add custom features and variables
4. Implement walk-forward analysis
5. Deploy for live trading (with proper risk management)

For detailed instructions, see `TUTORIAL.md`.