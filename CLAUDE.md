# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT RULES
- **NO GUESSING**: Do not attempt solutions you're not certain about. If you don't know how to fix something, say so directly instead of wasting time with attempts that might not work.
- Be honest about limitations and knowledge gaps.
- **DO NOT suggest `uv run` or virtual environment activation**: After installing with `uv pip install -e ".[dev]"`, the `alphapy` command works directly.

## Overview

AlphaPy is a domain-agnostic machine learning pipeline framework. It provides a flexible ML pipeline built on scikit-learn, pandas, and polars.

In v4.0.0 (2026-04-26), the trading/markets/Alfi stack was split off into the private `alphapy-finance` repo. See `CHANGELOG.md` and the `v3.1.1-monolith` tag for the pre-split state. Sibling consumers:

- `alphapy-finance` (private) — trading, markets, Alfi platform.
- `alphapy-sports` (private) — sports betting prediction.

Both consume `alphapy-pro` via local editable install (`{ path = "../alphapy-pro", editable = true }`).

## Common Development Commands

### Build and Installation
```bash
pip install -e ".[dev]"
```

### Running AlphaPy
```bash
alphapy
```

There is only one entry point (`alphapy`). The `mflow` entry point moved to `alphapy-finance`.

### Documentation
```bash
cd docs
make html
```

### Cleanup Utilities
```bash
./utils/cleanup_runs.sh
```

## Architecture

### Core Components
- **`alphapy/`**: ML framework
  - `alphapy_main.py`: Pipeline entry point
  - `model.py`: Model management
  - `data.py`: Generic CSV/Parquet data loading
  - `frame.py`: Frame wrapper, polars/pandas abstraction
  - `features.py`, `transforms.py`, `variables.py`: Feature engineering
  - `estimators.py`: Estimator registry (sklearn, xgboost, catboost, lightgbm)
  - `optimize.py`: Hyperparameter optimization (Optuna)
  - `plots.py`: Visualization utilities
  - `globals.py`, `utilities.py`, `space.py`, `group.py`, `alias.py`, `calendrical.py`: Infrastructure

### Configuration System
All configurations use YAML format in the `config/` directory:
- `alphapy.yml`: Main configuration with project paths
- `algos.yml`: ML algorithm definitions and hyperparameters
- `model.yml.template`: Per-project template
- `variables.yml`, `groups.yml`: Feature/group definitions

### Project Structure
Projects under `projects/` (kaggle, pizza, time-series) contain:
- `config/model.yml`
- `data/`
- `runs/` (auto-generated)

### Pipeline Flow
1. Load `config/model.yml`
2. Ingest data → feature engineering → model training → evaluation
3. Models and results saved in timestamped run directories.

## Development Environment

- Python: 3.12+
- Editable install: `pip install -e ".[dev]"`

## Branching Workflow

- **main**: Production
- **develop**: Development
- **feature branches**: From develop, merge back to develop
