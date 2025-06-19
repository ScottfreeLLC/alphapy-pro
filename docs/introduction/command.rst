Command Line Interface
======================

The AlphaPy Pro Command Line Interface (CLI) provides a streamlined way
to run machine learning pipelines. Simply navigate to your project
directory and run the appropriate command.

Basic Usage
-----------

First, change to your project directory::

    cd path/to/project

Train models and generate predictions::

    alphapy

Core Commands
-------------

**alphapy** - Main ML Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage::

    alphapy [options]

Options:

* ``--train`` - Train new models and make predictions [Default]
* ``--predict`` - Make predictions using saved models
* ``--verbose`` - Enable verbose logging
* ``--debug`` - Enable debug mode with detailed output

Example::

    # Train models (default behavior)
    alphapy
    
    # Use a saved model for predictions
    alphapy --predict
    
    # Train with verbose output
    alphapy --train --verbose

**mflow** - MarketFlow Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For financial market analysis and trading systems::

    mflow [options]

Options:

* ``--train`` - Train new models [Default]
* ``--predict`` - Make predictions from saved models
* ``--tdate YYYY-MM-DD`` - Training start date (Default: earliest date)
* ``--pdate YYYY-MM-DD`` - Prediction date (Default: today)
* ``--lookback N`` - Number of days for lookback window
* ``--forecast N`` - Number of days to forecast

Examples::

    # Train a market model
    mflow
    
    # Train from a specific date
    mflow --tdate 2023-01-01
    
    # Make predictions for a specific date
    mflow --predict --pdate 2024-01-15
    
    # Use lookback window
    mflow --lookback 252 --forecast 20

**Custom Pipelines**
~~~~~~~~~~~~~~~~~~~

You can create custom domain-specific pipelines following the MarketFlow pattern.
See the AlphaPy Pro architecture documentation for implementation details.

Project Structure Requirements
------------------------------

Before running commands, ensure your project follows this structure::

    my_project/
    ├── config/
    │   └── model.yml       # Model configuration
    ├── data/
    │   ├── train.csv       # Training data
    │   └── test.csv        # Testing data (optional)
    └── runs/               # Output directory (auto-created)

Output Structure
----------------

After running a command, outputs are organized in timestamped directories::

    runs/
    └── run_YYYYMMDD_HHMMSS/
        ├── config/         # Configuration used
        ├── input/          # Input data snapshots
        ├── model/          # Trained models and metrics
        ├── output/         # Predictions and rankings
        └── plots/          # Visualizations

Common Workflows
----------------

**1. Quick Model Training**::

    cd projects/my_project
    alphapy

**2. Hyperparameter Tuning**:

First, edit ``config/model.yml`` to enable grid search::

    model:
        grid_search:
            option: True
            iterations: 100

Then run::

    alphapy --verbose

**3. Batch Predictions**::

    # Train once
    alphapy --train
    
    # Use the model for multiple prediction sets
    alphapy --predict

**4. Market Backtesting**::

    # Run market analysis with specific dates
    mflow --tdate 2020-01-01 --pdate 2023-12-31

Logging and Debugging
---------------------

AlphaPy Pro creates detailed logs:

* ``alphapy.log`` - Main pipeline log
* ``market_flow.log`` - MarketFlow specific log (when using mflow)

To increase log verbosity::

    alphapy --verbose --debug

Tips and Best Practices
-----------------------

1. **Always verify your data** before running models
2. **Start with small iterations** for grid search, then increase
3. **Use dated runs** for reproducibility
4. **Keep model.yml under version control**
5. **Review plots** in the output directory for insights

For more details on configuration options, see :doc:`../user_guide/project`.