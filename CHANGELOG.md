# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- AlphaPy machine learning framework for speculators and data scientists
- MarketFlow pipeline for financial market analysis
- SportFlow pipeline for sports prediction
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