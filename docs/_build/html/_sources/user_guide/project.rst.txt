Project Structure
=================

Overview
--------

AlphaPy Pro organizes machine learning projects into a standardized directory structure.
This structure ensures consistency, reproducibility, and easy management of experiments.

Basic Project Layout
--------------------

Every AlphaPy Pro project follows this structure::

    my_project/
    ├── config/
    │   └── model.yml           # Required: model configuration
    ├── data/
    │   ├── train.csv          # Training data
    │   └── test.csv           # Testing data (optional)
    └── runs/                  # Auto-created: experiment outputs
        └── run_YYYYMMDD_HHMMSS/
            ├── config/        # Configuration snapshot
            ├── input/         # Data snapshots
            ├── model/         # Trained models
            ├── output/        # Predictions
            └── plots/         # Visualizations

Creating a New Project
----------------------

1. **Create the project directory**::

    mkdir -p projects/my_project/{config,data}
    cd projects/my_project

2. **Create a model configuration**::

    # Copy from an example project
    cp ../kaggle/config/model.yml config/
    
    # Or use a minimal template (see below)

3. **Add your data files**::

    # Copy your training and test data
    cp /path/to/train.csv data/
    cp /path/to/test.csv data/

4. **Run the pipeline**::

    alphapy

Directory Details
-----------------

**config/**
    Contains YAML configuration files:
    
    * ``model.yml`` - Model pipeline configuration (required)
    * ``algos.yml`` - Algorithm hyperparameters (optional, uses defaults)
    * Additional domain-specific configs

**data/**
    Raw input data files:
    
    * Training data (required)
    * Testing data (optional for prediction)
    * Any supplementary data files
    * Supports CSV, TSV, and other delimited formats

**runs/**
    Auto-generated output directories:
    
    * Each run creates a timestamped subdirectory
    * Contains complete experiment artifacts
    * Preserves reproducibility

Model Configuration (model.yml)
-------------------------------

The ``model.yml`` file controls every aspect of the pipeline. Here's a comprehensive
example with all major sections:

.. code-block:: yaml

    # Project Configuration
    project:
        directory         : .                    # Project root (usually current dir)
        file_extension    : csv                  # Data file format
        submission_file   : 'submission'         # Kaggle submission template
        submit_probas     : False                # Submit probabilities vs labels

    # Model Training Configuration
    model:
        algorithms        : ['CATB', 'LGB', 'XGB', 'RF', 'LOGR']
        balance_classes   : True                 # Handle imbalanced data
        calibration       :
            option        : False                # Probability calibration
            type          : sigmoid              # sigmoid or isotonic
        cv_folds          : 5                    # Cross-validation folds
        estimators        : 100                  # Trees for ensemble methods
        grid_search       :
            option        : True                 # Enable hyperparameter search
            iterations    : 50                   # Number of search iterations
            random        : True                 # Random vs grid search
            subsample     : False                # Subsample for faster search
            sampling_pct  : 0.2                  # Subsample percentage
        pvalue_level      : 0.01                 # Feature selection p-value
        rfe               :
            option        : False                # Recursive feature elimination
            step          : 3                    # Features to remove per step
        scoring_function  : roc_auc              # Metric for model selection
        target            : target               # Target column name
        type              : classification        # classification or regression

    # Data Processing Configuration
    data:
        drop              : ['id', 'timestamp']  # Columns to drop
        features          : '*'                  # '*' for all, or list specific
        sampling          :
            option        : False                # Resample imbalanced classes
            method        : over_random          # SMOTE, ADASYN, etc.
            ratio         : 0.5                  # Target ratio
        sentinel          : -1                   # Missing value replacement
        separator         : ','                  # CSV delimiter
        shuffle           : True                 # Shuffle training data
        split             : 0.2                  # Validation split ratio

    # Feature Engineering Configuration
    features:
        clustering        :
            option        : True                 # Create cluster features
            increment     : 5                    # Cluster increment
            maximum       : 30                   # Max clusters
            minimum       : 5                    # Min clusters
        counts            :
            option        : True                 # Value count features
        encoding          :
            type          : target               # target, onehot, ordinal
        factors           : ['category1', 'category2']  # Categorical columns
        interactions      :
            option        : True                 # Polynomial interactions
            poly_degree   : 2                    # Interaction degree
            sampling_pct  : 10                   # Sample for efficiency
        lofo              :
            option        : True                 # LOFO importance
        pca               :
            option        : False                # Principal components
            increment     : 1
            maximum       : 10
            minimum       : 2
        scaling           :
            option        : True                 # Feature scaling
            type          : standard             # standard, minmax, robust
        text              :
            ngrams        : 2                    # For text features
            vectorize     : False                # TF-IDF vectorization
        univariate        :
            option        : True                 # Univariate selection
            percentage    : 50                   # Features to keep
            score_func    : f_classif            # Selection function

    # Pipeline Configuration
    pipeline:
        number_jobs       : -1                   # Parallel jobs (-1 = all CPUs)
        seed              : 42                   # Random seed
        verbosity         : 2                    # Logging level (0-3)

    # Visualization Configuration
    plots:
        calibration       : True                 # Calibration plots
        confusion_matrix  : True                 # Confusion matrices
        importances       : True                 # Feature importance
        learning_curve    : True                 # Learning curves
        roc_curve         : True                 # ROC curves

Configuration Sections
----------------------

**Project Section**
    Controls file I/O and submission formatting:
    
    * ``directory`` - Working directory (usually '.')
    * ``file_extension`` - Input file format
    * ``submission_file`` - Competition submission template
    * ``submit_probas`` - Output probabilities or labels

**Model Section**
    Core modeling parameters:
    
    * ``algorithms`` - List of ML algorithms to train
    * ``balance_classes`` - Handle class imbalance
    * ``calibration`` - Probability calibration options
    * ``grid_search`` - Hyperparameter optimization
    * ``scoring_function`` - Evaluation metric
    * ``type`` - Problem type (classification/regression)

**Data Section**
    Data preprocessing options:
    
    * ``drop`` - Features to remove
    * ``features`` - Features to use ('*' for all)
    * ``sampling`` - Resampling for imbalanced data
    * ``split`` - Train/validation split ratio
    * ``target`` - Target variable name

**Features Section**
    Feature engineering configuration:
    
    * ``clustering`` - K-means cluster features
    * ``encoding`` - Categorical encoding method
    * ``interactions`` - Polynomial features
    * ``lofo`` - Leave One Feature Out importance
    * ``scaling`` - Feature normalization
    * ``univariate`` - Statistical feature selection

**Pipeline Section**
    Execution parameters:
    
    * ``number_jobs`` - Parallelization (-1 for all cores)
    * ``seed`` - Random seed for reproducibility
    * ``verbosity`` - Logging detail level

**Plots Section**
    Visualization options:
    
    * Enable/disable specific plot types
    * All plots saved to runs/*/plots/

Algorithm Configuration (algos.yml)
-----------------------------------

The optional ``algos.yml`` file defines hyperparameter grids for each algorithm.
If not provided, sensible defaults are used. Example:

.. code-block:: yaml

    CATB:
        iterations: [100, 500, 1000]
        learning_rate: [0.01, 0.05, 0.1]
        depth: [4, 6, 8]
        l2_leaf_reg: [1, 3, 5]

    LGB:
        n_estimators: [100, 500, 1000]
        learning_rate: [0.01, 0.05, 0.1]
        num_leaves: [31, 63, 127]
        feature_fraction: [0.8, 0.9, 1.0]

    XGB:
        n_estimators: [100, 500, 1000]
        learning_rate: [0.01, 0.05, 0.1]
        max_depth: [3, 5, 7]
        subsample: [0.8, 0.9, 1.0]

Time Series Projects
--------------------

For time series analysis, add these configuration options:

.. code-block:: yaml

    model:
        time_series:
            option        : True
            date_index    : date                # Date column
            group_id      : symbol              # Group by column
            forecast      : 1                   # Forecast horizon
            n_lags        : 10                  # Lag features
            leaders       : []                  # Leading indicators

Best Practices
--------------

1. **Version Control** - Keep config files in git
2. **Data Management** - Store large data files outside the repo
3. **Experiment Tracking** - Use descriptive project names
4. **Configuration** - Start with defaults, tune incrementally
5. **Reproducibility** - Always set the random seed

Example Projects
----------------

The AlphaPy Pro repository includes several example projects:

* ``projects/kaggle/`` - Titanic competition starter
* ``projects/shannons-demon/`` - Trading strategy implementation
* ``projects/time-series/`` - Market prediction example
* ``projects/triple-barrier-method/`` - Advanced financial ML

Each example includes complete configuration files and sample data.