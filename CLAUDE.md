# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT RULES
- **NO GUESSING**: Do not attempt solutions you're not certain about. If you don't know how to fix something, say so directly instead of wasting time with attempts that might not work.
- Be honest about limitations and knowledge gaps
- **DO NOT suggest `uv run` or virtual environment activation**: After installing with `uv pip install -e ".[dev]"`, commands like `alphapy` and `mflow` work directly. Never tell the user to use `uv run` or `source .venv/bin/activate`.

## Overview

AlphaPy Pro is a machine learning framework designed for speculators and data scientists. It provides a flexible ML pipeline built on scikit-learn and pandas, with specialized pipelines for market analysis (MarketFlow).

## Common Development Commands

### Build and Installation
```bash
# Install package locally (editable mode) - commands work directly after this
pip install -e ".[dev]"

# Build distribution packages
python -m build

# Check package integrity
twine check dist/*
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
- Entry points are defined in pyproject.toml: `alphapy` and `mflow`
- The project uses Apache License 2.0
- Warning suppression is in place for pandas and sklearn deprecations

## Development Environment

- Python Version: 3.12+
- Development Installation: `pip install -e ".[dev]"` (editable install, commands work directly)

## Branching Workflow

- **main**: Production branch - stable releases only
- **develop**: Development branch - all new features and fixes should be tested here first
- **feature branches**: Create from develop for new features, merge back to develop

### Development Process
1. Always work on develop branch or feature branches created from develop
2. Test changes on develop before merging to main
3. Create releases from main branch only
4. Use PR workflow: feature → develop → main