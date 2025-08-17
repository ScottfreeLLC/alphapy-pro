Installation
============

AlphaPy Pro requires Python 3.10 or higher. You can install it directly
from the source repository or via pip after it's published.

Development Installation
------------------------

To install AlphaPy Pro for development, clone the repository and install
in editable mode::

    git clone https://github.com/ScottFreeLLC/alphapy-pro.git
    cd alphapy-pro
    pip install -e .

This will install AlphaPy Pro along with all required dependencies.

Dependencies
------------

AlphaPy Pro will automatically install the following core dependencies:

* **Data Processing**: pandas>=2.0.0, numpy>=1.24.0
* **Machine Learning**: scikit-learn>=1.3.0
* **Gradient Boosting**: xgboost>=2.0.0, lightgbm>=4.0.0, catboost>=1.2
* **Visualization**: matplotlib>=3.7.0, seaborn>=0.12.0
* **Configuration**: pyyaml>=6.0

Additional dependencies for specific features:

* **Market Data**: yfinance, polygon-api-client, pandas-datareader
* **Feature Engineering**: category_encoders, lofo-importance
* **Imbalanced Learning**: imbalanced-learn
* **Calibration**: venn-abers
* **Portfolio Analysis**: pyfolio (optional, for legacy support)

Anaconda Python
---------------

If you're using Anaconda Python, you can create a dedicated environment
for AlphaPy Pro:

.. code-block:: bash

    conda create -n alphapy-pro python=3.9
    conda activate alphapy-pro
    
    # Install from conda-forge when available
    conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn pyyaml
    conda install -c conda-forge xgboost lightgbm catboost
    
    # Install remaining packages via pip
    pip install yfinance polygon-api-client lofo-importance
    pip install imbalanced-learn category_encoders venn-abers
    
    # Install AlphaPy Pro
    cd /path/to/alphapy-pro
    pip install -e .

Platform-Specific Notes
-----------------------

**macOS (Apple Silicon)**
    If you're on an M1/M2 Mac, some packages may require special handling.
    LightGBM and XGBoost should install correctly with recent versions.
    
**Windows**
    All packages should install correctly via pip. If you encounter issues
    with XGBoost, refer to the official XGBoost documentation.

**Linux**
    Standard installation should work without issues. Ensure you have
    Python development headers installed (python3-dev on Ubuntu/Debian).

Verifying Installation
----------------------

After installation, verify that AlphaPy Pro is correctly installed::

    # Check if the alphapy command is available
    alphapy --help
    
    # Check if the mflow command is available
    mflow --help
    
    # In Python, verify imports
    python -c "import alphapy; print(alphapy.__version__)"

Troubleshooting
---------------

If you encounter installation issues:

1. **Upgrade pip**: ``pip install --upgrade pip``
2. **Clear pip cache**: ``pip cache purge``
3. **Install with verbose output**: ``pip install -v -e .``
4. **Check for conflicting packages**: ``pip check``

For specific package issues, consult the documentation for that package:

* XGBoost: https://xgboost.readthedocs.io/
* LightGBM: https://lightgbm.readthedocs.io/
* CatBoost: https://catboost.ai/docs/