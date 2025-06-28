# Time Series Forecasting Project

## Overview

This project implements time series forecasting using the AlphaPy framework with specialized handling for temporal data. It's configured to predict future values based on historical patterns, exogenous variables, and time-based features.

## Objective

Build robust time series models that can:
- Handle multiple time series simultaneously (panel data)
- Incorporate exogenous variables for improved predictions
- Utilize lag features and time-based patterns
- Generate accurate one-step-ahead forecasts

## Dataset Structure

The training data (`train.csv`) contains:

- **unique_id**: Identifier for different time series (e.g., 'BE' for Belgium)
- **ds**: Datetime stamp (hourly frequency)
- **y**: Target variable to predict (appears to be energy/demand data)
- **Exogenous1, Exogenous2**: External predictor variables
- **day_0 to day_6**: One-hot encoded day of week features (0=Monday, 6=Sunday)

Example data shows hourly readings with values ranging from ~50 to 200+ for the target variable.

## Model Configuration

The project uses regression algorithms optimized for time series:

- **Algorithms**: CatBoost Regressor, LightGBM Regressor, Random Forest Regressor, XGBoost Regressor
- **Model Type**: Regression
- **Evaluation Metric**: Negative Mean Absolute Error
- **Cross-Validation**: 5-fold time series split

### Time Series Specific Settings

- **Option**: Enabled for temporal awareness
- **Date Index**: 'ds' column for time ordering
- **Forecast Horizon**: 1 step ahead
- **Lag Features**: 1 lag included (previous value)
- **Group ID**: 'unique_id' for panel data handling
- **Leaders**: Exogenous variables and day-of-week features preserved across time

### Feature Engineering

- **Date Transformations**: Automatic date part extraction (year, month, day, hour, etc.)
- **Scaling**: Standard scaling applied to features
- **Univariate Selection**: Top 50% of features selected
- **No PCA/Clustering**: Disabled to preserve temporal relationships

## How to Run

1. **Prerequisites**:
   ```bash
   # Ensure AlphaPy is installed
   pip install -e /path/to/alphapy-pro
   ```

2. **Navigate to the project directory**:
   ```bash
   cd /path/to/alphapy-pro/projects/time-series
   ```

3. **Run the AlphaPy pipeline**:
   ```bash
   alphapy
   ```

   Or use the market flow pipeline if working with financial time series:
   ```bash
   mflow
   ```

4. **Monitor progress**:
   - Check `alphapy.log` or `market_flow.log` for execution details
   - Results saved in timestamped directories under `runs/`

## Output Structure

Each run creates a directory (e.g., `runs/run_20240309_121700/`) containing:

- **config/**: Configuration used for the run
- **input/**: Copy of input data
- **model/**: Trained models and feature mappings
- **output/**: Predictions and forecasts
- **plots/**: Performance visualizations

## Time Series Considerations

### Data Preparation

1. **Temporal Ordering**: Data must be sorted by date within each group
2. **Missing Values**: Handle gaps in time series appropriately
3. **Stationarity**: Consider differencing or detrending if needed
4. **Seasonality**: Day-of-week features capture weekly patterns

### Model Evaluation

- **Walk-Forward Validation**: Models trained on past data, tested on future
- **No Data Leakage**: Careful handling of lag features and time splits
- **Group-Aware Splits**: Each unique_id treated separately in validation

### Feature Importance

With time series data, expect high importance for:
- Recent lag values
- Time-based features (hour of day, day of week)
- Exogenous variables if they have predictive power

## Tips for Improvement

1. **Add More Lags**: 
   - Increase `n_lags` in configuration
   - Consider multiple lag orders (1, 7, 24 for hourly data)

2. **Seasonal Features**:
   - Add month, quarter, or holiday indicators
   - Include interaction terms between time features

3. **Exogenous Variables**:
   - Weather data for energy demand
   - Economic indicators for financial series
   - Calendar events (holidays, special events)

4. **Advanced Techniques**:
   - Implement differencing for non-stationary series
   - Add moving averages or exponential smoothing features
   - Consider ensemble methods combining multiple horizons

5. **Hyperparameter Tuning**:
   - Enable grid search for algorithm optimization
   - Focus on tree depth and learning rate for gradient boosting

## Common Use Cases

This setup works well for:
- **Energy Demand Forecasting**: Hourly/daily consumption patterns
- **Sales Forecasting**: With weekly/monthly seasonality
- **Traffic Prediction**: Using historical patterns and events
- **Financial Time Series**: With appropriate risk considerations

## Troubleshooting

1. **Memory Issues**: Reduce features or use sampling for large datasets
2. **Poor Performance**: Check for data leakage or insufficient lag features
3. **Overfitting**: Reduce model complexity or add regularization
4. **Computational Time**: Use fewer algorithms or enable parallel processing