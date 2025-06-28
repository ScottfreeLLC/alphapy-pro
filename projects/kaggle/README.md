# Kaggle Titanic Competition

## Overview

This project implements machine learning models to predict passenger survival on the Titanic using the AlphaPy framework. It's configured to work with the classic Kaggle Titanic competition dataset.

## Objective

Build a predictive model that answers the question: "What sorts of people were more likely to survive?" using passenger data (name, age, gender, socio-economic class, etc).

## Dataset Description

The project uses three CSV files:

- **train.csv**: Training dataset with 891 passengers and their survival status
- **test.csv**: Test dataset with 418 passengers (survival status to be predicted)
- **gender_submission.csv**: Sample submission file showing the required format

### Key Features

- **PassengerId**: Unique identifier for each passenger
- **Survived**: Survival status (0 = No, 1 = Yes) - target variable
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Model Configuration

The project uses multiple algorithms configured in `config/model.yml`:

- **Algorithms**: CatBoost, LightGBM, Logistic Regression, Random Forest, XGBoost
- **Model Type**: Binary classification
- **Target Variable**: Survived
- **Evaluation Metric**: ROC AUC
- **Cross-Validation**: 3-fold
- **Class Balancing**: Enabled to handle imbalanced survival rates

### Feature Engineering

- **Factor Encoding**: Embarked port categorical variable
- **Clustering**: Automatic feature clustering (3-30 clusters)
- **Feature Counts**: Statistical count features
- **Target Encoding**: For categorical variables
- **Feature Scaling**: Standard scaling applied
- **Univariate Selection**: Top features based on f_classif scores
- **LOFO Importance**: Leave-One-Feature-Out importance analysis

## How to Run

1. **Prerequisites**:
   ```bash
   # Ensure AlphaPy is installed
   pip install -e /path/to/alphapy-pro
   ```

2. **Navigate to the project directory**:
   ```bash
   cd /path/to/alphapy-pro/projects/kaggle
   ```

3. **Run the AlphaPy pipeline**:
   ```bash
   alphapy
   ```

4. **Monitor progress**:
   - Check `alphapy.log` for detailed execution logs
   - Results will be saved in a timestamped directory under `runs/`

## Output Structure

Each run creates a timestamped directory (e.g., `runs/run_20250224_091440/`) containing:

- **config/**: Copy of the model configuration used
- **input/**: Copy of input data files
- **model/**: Trained model files and feature mappings
  - `model.pkl`: Serialized trained model
  - `feature_map.pkl`: Feature transformations
  - `model_metrics.csv`: Performance metrics
- **output/**: Predictions and rankings
  - `submission.csv`: Kaggle-ready submission file
  - `ranked_train.csv`: Training predictions with probabilities
  - `ranked_test.csv`: Test predictions with probabilities
- **plots/**: Visualization outputs
  - Confusion matrices for each algorithm
  - Learning curves
  - ROC curves
  - Calibration plots
  - Feature importance plots (LOFO)

## Performance Monitoring

The pipeline generates comprehensive performance metrics:

- **ROC AUC scores** for each algorithm
- **Confusion matrices** showing prediction accuracy
- **Learning curves** to detect overfitting
- **Calibration plots** for probability reliability
- **Feature importance** rankings

## Tips for Improvement

1. **Feature Engineering**: 
   - Extract titles from passenger names (Mr., Mrs., Miss., etc.)
   - Create family size features from SibSp and Parch
   - Bin continuous variables like Age and Fare

2. **Missing Data Handling**:
   - Age has significant missing values - consider imputation strategies
   - Cabin data is mostly missing - extract deck information where available

3. **Hyperparameter Tuning**:
   - Grid search is enabled with 20 iterations
   - Adjust search space in the main AlphaPy configuration

4. **Ensemble Methods**:
   - The pipeline already uses multiple algorithms
   - Consider stacking or blending predictions for better results

## Submission

To submit results to Kaggle:

1. Navigate to the latest run directory
2. Find `output/submission.csv`
3. Upload to Kaggle competition page

The submission file contains PassengerId and predicted Survived values in the required format.