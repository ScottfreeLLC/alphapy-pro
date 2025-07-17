# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT RULES
- **NO GUESSING**: Do not attempt solutions you're not certain about. If you don't know how to fix something, say so directly instead of wasting time with attempts that might not work.
- Be honest about limitations and knowledge gaps

## Overview

AlphaPy Pro is a machine learning framework designed for speculators and data scientists. It provides a flexible ML pipeline built on scikit-learn and pandas, with specialized pipelines for market analysis (MarketFlow) and sports prediction (SportFlow).

## Common Development Commands

### Build and Installation
```bash
# Install in development mode
pip install -e .

# Build distribution packages
python setup.py sdist bdist_wheel

# Create conda environment (if environment.yml exists)
conda env create -f environment.yml
```

### Running AlphaPy
```bash
# Main pipeline
alphapy

# Market flow pipeline  
mflow
```

### Documentation
```bash
# Build HTML documentation
cd docs
make html

# Clean documentation build
make clean
```

### Cleanup Utilities
```bash
# Remove old run directories (keeps most recent)
./utils/cleanup_runs.sh
```

## Architecture

### Core Components
- **alphapy/**: Main package containing all modules
  - `alphapy_main.py`: Main pipeline entry point
  - `mflow_main.py`: Market flow pipeline entry point
  - `model.py`: Core model management
  - `data.py`, `frame.py`: Data processing
  - `features.py`, `transforms.py`, `variables.py`: Feature engineering
  - `portfolio.py`, `system.py`: Trading functionality
  - `plots.py`: Visualization utilities

### Configuration System
All configurations use YAML format in the `config/` directory:
- `alphapy.yml`: Main configuration with project paths
- `algos.yml`: ML algorithm definitions and hyperparameters
- `model.yml`: Project-specific model configuration (in each project directory)
- `variables.yml`, `groups.yml`, `sources.yml`, `systems.yml`: Feature and data configurations

### Project Structure
Projects are organized under `projects/` with each containing:
- `config/model.yml`: Project-specific configuration
- `data/`: Input data files
- `runs/`: Output directories for each run (auto-generated)

### Pipeline Flow
1. Domain pipeline transforms raw data (market data, sports data, etc.)
2. Core ML pipeline trains and evaluates models
3. Models and results are saved to disk in timestamped run directories

## Key Development Notes

- The framework separates domain-specific logic from the core ML pipeline
- All major functionality is configuration-driven through YAML files
- Entry points are defined in setup.py: `alphapy` and `mflow`
- The project uses Apache License 2.0
- Warning suppression is in place for pandas and sklearn deprecations

## Development Environment

- Python Environment: Uses py312 environment located at `/Users/markconway/miniconda3/envs/py312/bin/python`