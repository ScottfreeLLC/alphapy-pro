Introduction
============

**AlphaPy Pro** is an advanced machine learning framework for data scientists
and quantitative traders. Built on top of ``scikit-learn``, ``pandas``, and
modern ML libraries, it provides a comprehensive toolkit for feature engineering,
model development, and systematic trading. Here are just some of the things you
can do with AlphaPy Pro:

* Build and optimize ML models using ``scikit-learn``, ``XGBoost``, ``LightGBM``, and ``CatBoost``.
* Analyze financial markets with *MarketFlow* using multiple data providers.
* Implement advanced trading strategies with meta-labeling and the Triple Barrier Method.
* Develop and backtest trading systems with portfolio analysis.
* Implement custom domain-specific pipelines for specialized applications.

The ``alphapy`` package is the core platform providing the ML pipeline.
The *domain* pipeline MarketFlow (``mflow``) runs on top of ``alphapy``
for financial market analysis. As shown in the diagram below, we separate
the domain pipeline from the model pipeline. The domain pipeline transforms
raw application data into canonical form—training and testing sets—while
the model pipeline handles feature engineering, model training, and evaluation.
This architecture has been refined through numerous Kaggle competitions and
real-world trading applications.

.. image:: alphapy_pipeline.png
   :alt: AlphaPy Model Pipeline
   :width: 100%
   :align: center

Let's review all of the components in the diagram:

``Domain Pipeline``:
    This is the Python code that creates the standard training and
    testing data. For example, you may be combining different data
    frames or collecting time series data from an external feed.
    These data are transformed for input into the model pipeline.

``Domain YAML``: 
    AlphaPy uses configuration files written in YAML to give the
    data scientist maximum flexibility. Typically, you will have
    a standard YAML template for each domain or application.

``Training Data``: 
    The training data is an external file that is read as a
    pandas dataframe. For classification, one of the columns will
    represent the target or dependent variable.

``Testing Data``:  
    The testing data is an external file that is read as a pandas
    dataframe. For classification, the labels may or may not be
    included.

``Model Pipeline``: 
    This Python code is generic for running all classification or
    regression models. The pipeline begins with data and ends with
    a model object for new predictions.

``Model YAML``: 
    The configuration file has specific sections for running the
    model pipeline. Every aspect of creating a model is controlled
    through this file.

``Model Object``: 
    All models are saved to disk. You can load and run your trained
    model on new data in scoring mode.

Core Functionality
------------------

**AlphaPy** has been developed primarily for supervised learning
tasks. You can generate models for any classification or regression
problem.

* Binary Classification: classify elements into one of two groups
* Multiclass Classification: classify elements into multiple categories
* Regression: predict real values based on derived coefficients

Classification Algorithms:

* CatBoost (CATB)
* LightGBM (LGB)
* XGBoost (XGB) Binary and Multiclass
* Random Forests (RF)
* Extra Trees (EXT)
* Gradient Boosting (GB)
* Logistic Regression (LOGR)
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Naive Bayes (NB)
* AdaBoost (ADA)

Regression Algorithms:

* CatBoost Regressor
* LightGBM Regressor
* XGBoost Regressor
* Random Forest Regressor
* Extra Trees Regressor
* Gradient Boosting Regressor
* Linear Regression
* Ridge Regression
* Lasso Regression
* K-Nearest Neighbors Regressor

Key Features
------------

**AlphaPy Pro** includes several advanced features for modern ML workflows:

* **Meta-Labeling**: Triple Barrier Method for advanced financial ML
* **Feature Engineering**: Automated feature generation with clustering, interactions, and transformations
* **Feature Selection**: LOFO (Leave One Feature Out) importance and univariate selection
* **Model Calibration**: Probability calibration with sigmoid and isotonic methods
* **Advanced Visualization**: Learning curves, ROC curves, confusion matrices, and feature importance plots
* **Multiple Data Sources**: EODHD, Yahoo Finance, Polygon, IEX Cloud, and more
* **Grid Search**: Randomized and systematic hyperparameter optimization
* **Ensemble Methods**: Model blending and stacking

External Packages
-----------------

**AlphaPy Pro** leverages cutting-edge ML and data science packages:

* **Gradient Boosting**: XGBoost, LightGBM, CatBoost
* **Feature Engineering**: category_encoders, lofo-importance
* **Imbalanced Learning**: imbalanced-learn (SMOTE, ADASYN, etc.)
* **Calibration**: venn-abers for probability calibration
* **Market Data**: yfinance, polygon-api-client, pandas-datareader
* **Portfolio Analysis**: pyfolio (legacy), custom portfolio analytics
* **Visualization**: matplotlib, seaborn, plotly
* **Time Series**: statsmodels, arch
