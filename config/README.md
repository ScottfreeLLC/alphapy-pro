# AlphaPy Configuration Guide

This directory contains configuration files for AlphaPy Pro. Some files contain sensitive information (like API keys) and are user-specific, while others define the behavior of the machine learning pipelines.

## Configuration Files Overview

### User-Specific Files (Not in Git)
These files contain personal settings and API keys. They are gitignored and must be created from templates:

- **`alphapy.yml`** - Main configuration with your local directory paths
- **`sources.yml`** - API keys for data sources (KEEP SECRET!)

### Shared Configuration Files (In Git)
These files define ML algorithms, features, and system behavior:

- **`algos.yml`** - Machine learning algorithm definitions and hyperparameters
- **`variables.yml`** - Feature variable definitions
- **`groups.yml`** - Variable groupings for feature engineering
- **`systems.yml`** - Trading system definitions

### Project-Specific Files
- **`model.yml`** - Located in each project directory, defines project-specific settings

## Quick Start

1. **Copy Templates**: Create your personal configuration files from templates:
   ```bash
   cp alphapy.yml.template alphapy.yml
   cp sources.yml.template sources.yml
   ```

2. **Edit `alphapy.yml`**: Update with your local directory paths:
   - Set `data_dir` to where you want to store market data
   - Set project roots for each pipeline (aflow, mflow)

3. **Edit `sources.yml`**: Add your API keys:
   - Get free/paid API keys from the providers listed in the template
   - Never commit this file or share your API keys!

4. **Optional**: Customize other configuration files as needed for your ML experiments

## Configuration Details

### alphapy.yml
Controls where AlphaPy stores data and projects:
- `data_dir`: Central location for all downloaded market data
- `aflow.project_root`: Location for general ML projects
- `mflow.project_root`: Location for market analysis projects

### sources.yml
Contains API credentials for data providers:
- Each source has its own section with required credentials
- Some sources require paid subscriptions (Finnhub Premium, Finviz Elite)
- Free sources: Yahoo Finance, Pandas DataReader

### algos.yml
Defines available ML algorithms:
- Each algorithm has a type (classifier/regressor)
- Hyperparameter grids for optimization
- Package dependencies (sklearn, xgboost, etc.)

### model.yml
Project-specific settings (one per project):
- Data specifications (features, target, train/test periods)
- Model selection and parameters
- Feature engineering settings
- Output preferences

## Best Practices

1. **Security**: Never commit `sources.yml` or share your API keys
2. **Paths**: Use absolute paths in `alphapy.yml` for reliability
3. **Backups**: Keep backups of your personal config files
4. **Templates**: When adding new config options, update the templates

## Troubleshooting

- **Missing config file**: Copy from template and fill in your values
- **Path errors**: Ensure all paths in `alphapy.yml` exist and are absolute
- **API errors**: Verify your API keys in `sources.yml` are correct
- **Import errors**: Check that required packages for algorithms are installed