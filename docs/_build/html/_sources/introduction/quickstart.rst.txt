Quick Start
===========

Install AlphaPy Pro from source::

    git clone https://github.com/ScottFreeLLC/alphapy-pro.git
    cd alphapy-pro
    pip install -e .

.. note:: Please refer to :doc:`install` for detailed installation instructions.

Running Your First Model
------------------------

1. **Navigate to the Kaggle example**::

    cd projects/kaggle

2. **Run the AlphaPy pipeline**::

    alphapy

This will train multiple models on the Titanic dataset and generate predictions.

3. **Check the results**::

    ls -la runs/

You'll see a new timestamped directory with your model outputs, plots, and predictions.

Working with Projects
---------------------

AlphaPy Pro organizes work into projects. Each project contains:

* ``config/model.yml`` - Model configuration
* ``data/`` - Input data files
* ``runs/`` - Output directories for each model run

Example Projects
----------------

The repository includes several example projects:

* **Kaggle** - Titanic survival prediction
* **Shannon's Demon** - Trading strategy implementation
* **Time Series** - Market prediction examples
* **Triple Barrier Method** - Advanced labeling for financial ML

Quick Examples
--------------

**Running a Market Analysis**::

    cd projects/time-series
    mflow

**Customizing Model Configuration**:

Edit ``config/model.yml`` to change algorithms, features, or parameters::

    model:
        algorithms: ['CATB', 'LGB', 'XGB']  # Choose algorithms
        cv_folds: 5                          # Cross-validation folds
        grid_search:
            option: True                     # Enable hyperparameter tuning
            iterations: 50                   # Number of iterations

**Creating a New Project**::

    mkdir projects/my_project
    cd projects/my_project
    mkdir config data
    
    # Copy a template configuration
    cp ../kaggle/config/model.yml config/
    
    # Add your data to the data/ directory
    # Run alphapy

Working with Notebooks
----------------------

You can also work with AlphaPy Pro in Jupyter notebooks:

.. code-block:: python

    from alphapy.model import create_model
    from alphapy.frame import read_frame
    
    # Load your data
    train_df = read_frame('data/train.csv')
    test_df = read_frame('data/test.csv')
    
    # Create and train models
    model = create_model(train_df, test_df, 'config/model.yml')

.. note:: AlphaPy Pro supports both command-line and notebook workflows. 
   We recommend using the command line for production runs and notebooks 
   for exploratory analysis.

Next Steps
----------

* Explore the example projects in ``projects/``
* Read the :doc:`../user_guide/pipelines` guide
* Learn about :doc:`../user_guide/market_flow` for trading applications
* Check out the :doc:`../tutorials/kaggle` tutorial