# AlphaPy Pro

|badge_pypi| |badge_downloads| |badge_docs|

**AlphaPy Pro** is an advanced machine learning framework designed for speculators and data scientists. Building on the foundation of the original AlphaPy, this professional edition offers enhanced features, improved performance, and enterprise-grade capabilities for financial modeling and prediction.

Written in Python with `scikit-learn`, `pandas`, and many other powerful libraries, AlphaPy Pro provides a comprehensive toolkit for feature engineering, model development, and portfolio analysis.

## Key Features

* **Advanced ML Pipeline**: Run machine learning models using `scikit-learn`, `XGBoost`, `LightGBM`, and `CatBoost`
* **Ensemble Methods**: Generate sophisticated blended and stacked ensembles
* **MarketFlow**: Specialized pipeline for financial market analysis and trading system development
* **Portfolio Analysis**: Comprehensive trading system backtesting and portfolio optimization
* **Configuration-Driven**: Flexible YAML-based configuration system
* **Feature Engineering**: Advanced feature creation, selection, and transformation tools
* **Time Series Support**: Built-in support for time series forecasting and analysis

## Architecture

### Core Components

- **alphapy/**: Main package with core ML functionality
- **config/**: YAML configuration files for algorithms, features, and systems
- **projects/**: Individual project workspaces with isolated configurations
- **docs/**: Comprehensive documentation and tutorials

### Pipeline Flow

1. **Data Ingestion**: Load and preprocess data from various sources
2. **Feature Engineering**: Create, transform, and select features
3. **Model Training**: Train and optimize multiple ML algorithms
4. **Ensemble Creation**: Combine models for improved performance
5. **Evaluation**: Generate comprehensive performance metrics and visualizations
6. **Deployment**: Export models and predictions for production use

## Quick Start

### Installation

#### From PyPI (Recommended)
```bash
# Install the latest stable version
pip install alphapy

# Install with optional dependencies
pip install alphapy[dev,docs]
```

#### Development Installation
```bash
# Clone the repository
git clone https://github.com/ScottFreeLLC/AlphaPy.git
cd AlphaPy

# Install in development mode with dev dependencies
pip install -e .[dev]
```

#### From Source
```bash
# Build and install using modern packaging
python -m build
pip install dist/alphapy-*.whl
```

### Configuration Setup

1. **Copy configuration templates**:
   ```bash
   cd config
   cp alphapy.yml.template alphapy.yml
   cp sources.yml.template sources.yml
   ```

2. **Edit configurations**:
   - Update `alphapy.yml` with your local directory paths
   - Add your API keys to `sources.yml` (keep this file secure!)

3. **See `config/README.md` for detailed setup instructions**

### Running AlphaPy Pro

```bash
# Main pipeline
alphapy

# Market analysis pipeline
mflow
```

## Project Structure

```
alphapy-pro/
├── alphapy/                 # Core package
│   ├── alphapy_main.py     # Main pipeline entry point
│   ├── mflow_main.py       # Market flow pipeline
│   ├── model.py            # Model management
│   ├── features.py         # Feature engineering
│   └── ...
├── config/                  # Configuration files
│   ├── alphapy.yml         # Main configuration (user-specific)
│   ├── sources.yml         # API keys (user-specific, gitignored)
│   ├── algos.yml           # ML algorithm definitions
│   ├── variables.yml       # Feature definitions
│   └── ...
├── projects/               # Project workspaces
│   ├── kaggle/            # Kaggle competition example
│   ├── time-series/       # Time series analysis
│   └── ...
└── docs/                  # Documentation
```

## Configuration System

AlphaPy Pro uses a comprehensive YAML-based configuration system:

- **`config/alphapy.yml`**: Main configuration with project paths
- **`config/sources.yml`**: API keys for data sources ⚠️ **Keep Secret!**
- **`config/algos.yml`**: ML algorithm definitions and hyperparameters
- **`config/variables.yml`**: Feature variable definitions
- **`config/groups.yml`**: Variable groupings for feature engineering
- **`config/systems.yml`**: Trading system definitions
- **`projects/*/config/model.yml`**: Project-specific model configurations

## MarketFlow

Specialized pipeline for financial market analysis featuring:

- **Multi-source data integration** (EOD Historical Data, Finnhub, IEX, Polygon, etc.)
- **Advanced technical indicators** and market features
- **Trading system development** and backtesting
- **Portfolio optimization** and risk analysis
- **Real-time prediction** capabilities


## Data Sources

AlphaPy Pro supports multiple data providers:

- **EOD Historical Data**: Historical market data
- **Finnhub**: Real-time market data and news
- **IEX Cloud**: Financial data platform
- **Polygon**: Stock market data API
- **Yahoo Finance**: Free market data
- **Custom sources**: Easily integrate your own data

## Examples

### Kaggle Competition
```bash
cd projects/kaggle
alphapy
```

### Time Series Analysis
```bash
cd projects/time-series
alphapy
```

### Market Analysis
```bash
cd projects/your-market-project
mflow
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

```bash
cd docs
make html
```

Documentation covers:
- **Installation and setup**
- **Configuration guide**
- **Feature engineering**
- **Model development**
- **Trading systems**
- **API reference**

## Development

### Build Documentation
```bash
cd docs
make html
```

### Clean Up Old Runs
```bash
./cleanup_runs.sh
```

### Testing
```bash
# Run tests with pytest
pytest

# Run tests with coverage
pytest --cov=alphapy --cov-report=html

# Run specific test files
pytest tests/test_version.py -v
```

## Requirements

- **Python 3.9+** (3.9, 3.10, 3.11 supported)
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computing
- **PyYAML**: Configuration file parsing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

### Optional Dependencies
- **pytest**: For running tests (install with `pip install alphapy[test]`)
- **black, isort, flake8**: Code quality tools (install with `pip install alphapy[dev]`)
- **sphinx**: Documentation building (install with `pip install alphapy[docs]`)

## Contributing

We welcome contributions to AlphaPy Pro! Here's how you can help:

1. **Fork the repository** and create your feature branch from `main`
2. **Make your changes** following the existing code style
3. **Add tests** for any new functionality
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description of your changes

### Development Setup

```bash
# Clone your fork
git clone https://github.com/ScottFreeLLC/AlphaPy.git
cd AlphaPy

# Install in development mode with all dev dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Set up configuration
cd config
cp alphapy.yml.template alphapy.yml
cp sources.yml.template sources.yml
# Edit with your settings
```

### Code Quality

This project uses several tools to maintain code quality:

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Check code style with flake8
flake8 .

# Run type checking with mypy
mypy alphapy/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Code Guidelines

- Follow PEP 8 style guidelines (enforced by flake8)
- Use black for code formatting
- Add docstrings to new functions and classes
- Include type hints where appropriate
- Write tests for new functionality
- Update CLAUDE.md if adding new development commands

## License

AlphaPy Pro is licensed under the Apache License 2.0. See `LICENSE` for details.

## Support

- **Issues**: Open an issue on GitHub for bug reports and feature requests
- **Documentation**: Check the `docs/` directory for comprehensive guides
- **Configuration**: See `config/README.md` for setup help

## Donations

If you find AlphaPy Pro valuable for your work, please consider supporting its development:

- **GitHub Sponsors**: [Sponsor this project](https://github.com/sponsors/your-username)
- **Buy Me a Coffee**: Support ongoing development
- **PayPal**: Direct donations welcome

Your support helps maintain and improve AlphaPy Pro for the entire community.

## Acknowledgments

AlphaPy Pro builds upon the foundation of the original AlphaPy framework, incorporating lessons learned and feature requests from the community. Special thanks to all contributors and users who have helped shape this professional edition.

---

*AlphaPy Pro - Professional Machine Learning for Financial Markets and Beyond*

.. |badge_pypi| image:: https://badge.fury.io/py/alphapy-pro.svg
.. |badge_docs| image:: https://readthedocs.org/projects/alphapy-pro/badge/?version=latest
.. |badge_downloads| image:: https://static.pepy.tech/badge/alphapy-pro