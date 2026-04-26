# AlphaPy

[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://scottfreellc.github.io/alphapy-pro/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

AlphaPy is a domain-agnostic machine learning pipeline framework for data scientists. It provides a YAML-driven workflow built on scikit-learn, pandas, and polars, with first-class support for XGBoost, CatBoost, LightGBM, and Optuna.

> **v4.0 note:** Trading, markets, and the Alfi platform have moved to the private `alphapy-finance` repository. See `CHANGELOG.md` and the `v3.1.1-monolith` tag for the pre-split state.

## Features

- **YAML-driven configuration** — declarative project setup, no boilerplate
- **Pluggable algorithms** — sklearn estimators plus XGBoost, CatBoost, LightGBM
- **Automated feature engineering** — encoding (category_encoders), scaling, selection (LOFO, RFECV, univariate)
- **Hyperparameter optimization** — Optuna with sklearn integration
- **Probability calibration** — Venn-Abers, sigmoid, isotonic
- **Imbalanced learning** — built-in SMOTE and class-weight strategies
- **Polars + pandas** — choose your DataFrame engine
- **Reproducible runs** — every training run lands in a timestamped `runs/` directory

## Installation

```bash
git clone https://github.com/ScottfreeLLC/alphapy-pro.git
cd alphapy-pro
pip install -e ".[dev]"
```

After install, the `alphapy` command is on your PATH directly.

## Quick start

Each ML project lives under `projects/<name>/` with its own `config/model.yml`. Three example projects ship with the repo:

```bash
cd projects/kaggle      # Titanic
alphapy

cd projects/pizza       # Toppings ranker
alphapy

cd projects/time-series # Generic time-series forecasting
alphapy
```

Outputs land in `projects/<name>/runs/run_YYYYMMDD_HHMMSS/`.

## Project structure

```
alphapy-pro/
├── alphapy/                 # ML framework
│   ├── alphapy_main.py     # Pipeline entry point
│   ├── model.py            # Model management
│   ├── data.py             # Generic CSV/Parquet loading
│   ├── frame.py            # Frame wrapper (polars/pandas)
│   ├── features.py         # Feature engineering
│   ├── transforms.py       # Generic transforms
│   ├── variables.py        # Declarative variable system
│   ├── estimators.py       # Estimator registry
│   ├── optimize.py         # Hyperparameter optimization
│   ├── plots.py            # Visualization
│   └── ...
├── config/                  # Global configs
│   ├── alphapy.yml         # Main config (paths)
│   ├── algos.yml           # Algorithm definitions
│   ├── variables.yml       # Variable definitions
│   ├── groups.yml          # Variable groups
│   └── model.yml.template  # Per-project template
├── projects/               # Example projects
│   ├── kaggle/
│   ├── pizza/
│   └── time-series/
├── docs/                   # Sphinx documentation
└── tests/                  # Test suite
```

## Configuration

- `config/alphapy.yml` — global paths
- `config/algos.yml` — algorithm definitions and hyperparameter grids
- `config/variables.yml` — feature variable expressions
- `config/groups.yml` — feature groupings
- `projects/<name>/config/model.yml` — per-project model config (target, algorithms, CV, optimization, encoding, scaling, selection)

## Documentation

[https://scottfreellc.github.io/alphapy-pro/](https://scottfreellc.github.io/alphapy-pro/)

```bash
cd docs
make html
```

## Downstream consumers

`alphapy-pro` is consumed as a library by domain-specific repos (private):

- **`alphapy-finance`** — trading, markets, Alfi (FastAPI/React) platform
- **`alphapy-sports`** — sports betting prediction

Both depend on `alphapy-pro` via `{ path = "../alphapy-pro", editable = true }` in their `pyproject.toml`.

## License

Apache-2.0. See `LICENSE`.
