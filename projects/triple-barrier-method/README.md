# Triple Barrier Method for Financial Machine Learning

## Overview

This project implements the Triple Barrier Method, an advanced labeling technique for financial machine learning introduced by Marcos López de Prado. It addresses the challenge of defining profitable trading opportunities by using three barriers (profit target, stop loss, and time limit) to label financial data for supervised learning.

## Objective

The Triple Barrier Method aims to:
- Create more realistic labels for financial time series data
- Account for both profit-taking and stop-loss scenarios
- Incorporate time decay in trading positions
- Improve the signal-to-noise ratio in financial ML models

## Core Concept

The method sets three barriers around each trading entry point:

1. **Upper Barrier (Profit Target)**: Position closes with profit if price touches this level
2. **Lower Barrier (Stop Loss)**: Position closes with loss if price touches this level  
3. **Vertical Barrier (Time Limit)**: Position closes at market price when time expires

The first barrier touched determines the label (1 for profit, 0 for loss/neutral).

## Project Components

### Jupyter Notebooks

1. **Triple Barrier Method.ipynb**: Main implementation demonstrating:
   - TSLA stock data analysis
   - Bollinger Bands for entry signals
   - CUSUM filter for detecting price changes
   - Triple barrier labeling
   - Random Forest model training
   - Performance evaluation

2. **TBM Intraday.ipynb**: Adaptation for intraday trading scenarios

### Key Implementation Steps

1. **Data Preparation**:
   - Fetches historical stock data (TSLA example)
   - Calculates technical indicators (Bollinger Bands)
   - Generates entry signals

2. **Volatility Estimation**:
   - Computes exponentially weighted moving average of returns
   - Uses volatility to scale barriers dynamically

3. **Event Detection**:
   - CUSUM (Cumulative Sum) filter identifies significant price moves
   - Reduces noise by filtering out small fluctuations

4. **Triple Barrier Labeling**:
   - Sets profit/stop multipliers (default: 2x volatility)
   - Defines time horizon (default: 5 days)
   - Labels each event based on which barrier is touched first

5. **Model Training**:
   - Uses Random Forest classifier
   - Features include OHLCV data and technical indicators
   - Evaluates both primary model (entry signals) and secondary model (meta-labeling)

## How to Run

### Prerequisites

```bash
# Install required packages
pip install pandas numpy matplotlib scikit-learn yfinance
pip install -e /path/to/alphapy-pro  # For alphapy modules
```

### Running the Analysis

1. **Open Jupyter Notebook**:
   ```bash
   cd /path/to/alphapy-pro/projects/triple-barrier-method
   jupyter notebook
   ```

2. **Execute the main notebook**:
   - Open `Triple Barrier Method.ipynb`
   - Run all cells sequentially
   - Modify parameters as needed

### Key Parameters to Adjust

- **Stock Symbol**: Change `stock = 'TSLA'` to analyze different securities
- **Data History**: Adjust `data_history = 1000` for more/less historical data
- **Bollinger Bands**: Modify `window = 50` and `no_of_stdev = 1.5`
- **Barriers**: Change `pt_sl = [2, 2]` for different profit/stop ratios
- **Time Horizon**: Adjust `num_days = 5` in vertical barrier
- **Minimum Return**: Set `min_ret = 0.03` threshold

## Expected Output

### Visualizations
- Bollinger Bands with entry signals
- ROC curves for model performance
- Feature importance rankings

### Metrics
- **Classification Report**: Precision, recall, F1-score
- **Confusion Matrix**: True/false positives and negatives
- **Accuracy Score**: Overall prediction accuracy
- **ROC AUC**: Area under the ROC curve

### Labels Distribution
The notebook shows the distribution of:
- Side labels (1 for long, -1 for short)
- Binary outcomes (0 for loss, 1 for profit)

## Meta-Labeling Concept

The project demonstrates meta-labeling, where:
1. A primary model generates entry signals
2. The secondary model (Random Forest) predicts position sizing
3. This two-step process improves risk management

## Advantages

1. **Realistic Labels**: Accounts for how trades actually close in practice
2. **Risk Management**: Built-in stop-loss consideration
3. **Time Decay**: Recognizes that predictions lose value over time
4. **Flexible Framework**: Barriers can be adjusted per market conditions

## Limitations and Considerations

1. **Look-Ahead Bias**: Careful implementation needed to avoid future information leakage
2. **Transaction Costs**: Not included in basic implementation
3. **Market Impact**: Assumes positions can be entered/exited at observed prices
4. **Survivorship Bias**: Using current stock data may exclude delisted securities

## Extensions and Improvements

1. **Dynamic Barriers**: Adjust barriers based on market regime
2. **Asymmetric Barriers**: Different ratios for profit vs stop-loss
3. **Multiple Time Frames**: Combine signals from different horizons
4. **Feature Engineering**: Add more technical indicators
5. **Ensemble Methods**: Combine multiple models for better predictions

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- The AlphaPy framework's metalabel module implements core functionality

## Tips for Production Use

1. **Backtesting**: Always validate on out-of-sample data
2. **Paper Trading**: Test strategies without real money first
3. **Risk Limits**: Implement position sizing and portfolio limits
4. **Monitoring**: Track model performance degradation over time
5. **Market Conditions**: Adjust parameters for different volatility regimes