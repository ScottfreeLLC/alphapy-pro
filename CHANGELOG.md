# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2026-04-26

### BREAKING

Trading, markets, and the Alfi platform have moved to the private `alphapy-finance` repository. See tag `v3.1.1-monolith` for the pre-split state.

### Removed
- `alphapy/mflow_main.py`, `alphapy/system.py`, `alphapy/backtest.py`, `alphapy/metalabel.py` — moved to `alphapy_finance/`.
- `alphapy/portfolio/`, `alphapy/indicators/`, `alphapy/data_sources/` — moved to `alphapy_finance/`.
- Market-data helpers in `alphapy/data.py` (`get_market_data`, `get_yahoo_data`, `get_polygon_data`, `get_alpaca_data`, `get_eodhd_data`, `convert_data`, `convert_offset`, `assign_global_data`, `standardize_data`, `resample_ohlcv`) — moved to `alphapy_finance/`.
- Top-level `app/` (Alfi UI/backend/agent) and `agent/` (legacy trading agent) — moved.
- 13 trading project examples (crypto, intraday, orb, v-system, vwap, shannons-demon, numerai, two-sigma, trade-gpt, price-encoding, triple-barrier-method, metalabel, ranker) — moved.
- Trading scripts (`run_agent.py`, `run_research.py`, `sync_tradingagents.py`, `scripts/markets/`) — moved.
- Trading tests (`tests/agent/`, `tests/app_agent/`, `tests/backend/`, `tests/integration/`, finance-specific files in `tests/alphapy/`) — moved.
- Markets docs (`docs/PLAN-markets.md`, `docs/markets/`) — moved.
- Trading config files (`sources.yml.template`, `systems.yml`, `groups.yml`, `indicators.yml`) — moved.
- Entry points `mflow`, `scott`, `research` — moved to alphapy-finance.
- ~30 dependencies removed from `pyproject.toml` (alpaca-py, vectorbt, fastapi, langchain*, yfinance, polygon-api-client, etc.).

### Changed
- `alphapy/__init__.py` exports `__version__` and a stable `__all__` list.
- `alphapy/data.py` no longer module-imports `yfinance` or `data_sources`.

### Migration
Repos that depended on `alphapy-pro` for finance code should switch to `alphapy-finance`:
```toml
# pyproject.toml
[tool.uv.sources]
alphapy-finance = { path = "../alphapy-finance", editable = true }
```

## [Unreleased]

### Added
- Modern packaging with pyproject.toml
- Comprehensive testing framework with pytest
- GitHub Actions CI/CD workflows
- Code quality tools (black, isort, flake8, mypy)
- Pre-commit hooks for code quality
- Dynamic versioning system
- Automated PyPI publishing with trusted publishing
- Developer documentation

### Changed
- Migrated from setup.py to pyproject.toml for packaging
- Updated documentation for modern Python packaging standards
- Enhanced security by removing API keys from repository history
- Updated Python version support to 3.10, 3.11, 3.12 (dropped 3.9)

### Security
- Removed sensitive API keys and credentials from git history
- Added security scanning to prevent future credential leaks

## [3.0.0] - 2024-XX-XX

### Added
- AlphaPy Pro machine learning framework for speculators and data scientists
- MarketFlow pipeline for financial market analysis
- Flexible ML pipeline built on scikit-learn and pandas
- Support for Python 3.9, 3.10, and 3.11

### Features
- Core ML pipeline with model management
- Feature engineering and data processing utilities
- Portfolio management and trading systems
- Visualization tools for analysis
- Configuration-driven approach with YAML files
- Command-line interfaces: `alphapy` and `mflow`

[Unreleased]: https://github.com/ScottFreeLLC/AlphaPy/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/ScottFreeLLC/AlphaPy/releases/tag/v3.0.0