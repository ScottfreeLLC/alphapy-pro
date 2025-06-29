# Contributing to AlphaPy Pro

Thank you for your interest in contributing to AlphaPy Pro! This guide will help you get started with development.

## Development Setup

### Prerequisites

- Python 3.10+ (3.10, 3.11, or 3.12)
- Git
- pip

### Setting Up Your Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/alphapy-pro.git
   cd alphapy-pro
   ```

2. **Install in development mode**:
   ```bash
   # Install with all development dependencies
   pip install -e .[dev]
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Set up configuration**:
   ```bash
   cd config
   cp alphapy.yml.template alphapy.yml
   cp sources.yml.template sources.yml
   # Edit these files with your local settings
   ```

5. **Verify installation**:
   ```bash
   python -c "import alphapy; print(alphapy.__version__)"
   pytest tests/test_version.py -v
   ```

## Development Workflow

For the complete Git workflow guide, see [docs/git-workflow.md](docs/git-workflow.md).

### Quick Start

1. **Create an issue** describing your feature or bug fix
2. **Create a branch** from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/issue-123-description
   ```
3. **Make your changes** following the code quality standards below
4. **Create a Pull Request** to merge back to `develop`
5. **Address review feedback** and get approval
6. **Squash and merge** when ready

### Code Quality Standards

This project maintains high code quality standards using automated tools:

#### Code Formatting
```bash
# Format all code with black
black .

# Sort imports with isort
isort .
```

#### Linting
```bash
# Check code style with flake8
flake8 .

# Run type checking (optional but recommended)
mypy alphapy/
```

#### Pre-commit Hooks
All code quality checks run automatically before commits:
```bash
# Run all pre-commit hooks manually
pre-commit run --all-files
```

### Testing

#### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=alphapy --cov-report=html

# Run specific test files
pytest tests/test_utilities.py -v

# Run tests matching a pattern
pytest -k "test_version" -v
```

#### Writing Tests
- Place tests in the `tests/` directory
- Follow the naming convention: `test_*.py`
- Use descriptive test names and docstrings
- Test both success and failure cases
- Use pytest fixtures for common setup

Example test structure:
```python
"""Test description module."""
import pytest
from alphapy.utilities import some_function


class TestSomeFunction:
    """Test suite for some_function."""
    
    def test_some_function_with_valid_input(self):
        """Test some_function with valid input."""
        result = some_function("valid_input")
        assert result == "expected_output"
    
    def test_some_function_with_invalid_input(self):
        """Test some_function raises error with invalid input."""
        with pytest.raises(ValueError):
            some_function("invalid_input")
```

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation if needed

3. **Run tests and quality checks**:
   ```bash
   # Run tests
   pytest

   # Run code quality checks
   black .
   isort .
   flake8 .
   
   # Or let pre-commit handle it
   pre-commit run --all-files
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

5. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Guidelines

Use clear, descriptive commit messages:

- **Add**: New features or functionality
- **Fix**: Bug fixes
- **Update**: Improvements to existing features
- **Remove**: Deletion of code or features
- **Docs**: Documentation changes
- **Test**: Test additions or modifications

Examples:
```
Add: version management system with semantic versioning
Fix: handle edge case in valid_date function
Update: modernize packaging with pyproject.toml
Test: add comprehensive tests for utilities module
```

## Code Style Guidelines

### Python Code Style

- Follow PEP 8 (enforced by flake8)
- Use black for code formatting (line length: 88 characters)
- Use isort for import sorting
- Add type hints where possible
- Write descriptive docstrings for public functions and classes

### Documentation Style

- Use Google-style docstrings
- Include type information in docstrings
- Provide examples for complex functions
- Keep documentation up to date with code changes

Example docstring:
```python
def process_data(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Process input data for machine learning.
    
    Args:
        data: Input DataFrame containing features and target
        target: Name of the target column
        
    Returns:
        A tuple containing (features, target) where features is a DataFrame
        and target is a Series.
        
    Raises:
        ValueError: If target column is not found in data.
        
    Example:
        >>> df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        >>> features, target = process_data(df, 'y')
        >>> len(features.columns)
        1
    """
```

## Project Structure

Understanding the project structure helps with navigation:

```
AlphaPy/
├── alphapy/                 # Main package
│   ├── __init__.py         # Package initialization with version
│   ├── alphapy_main.py     # Main pipeline entry point
│   ├── mflow_main.py       # Market flow pipeline
│   ├── model.py            # Model management
│   ├── features.py         # Feature engineering
│   ├── utilities.py        # Utility functions
│   └── ...
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py         # Pytest configuration and fixtures
│   ├── test_version.py     # Version tests
│   ├── test_utilities.py   # Utility function tests
│   └── ...
├── config/                  # Configuration files
│   ├── alphapy.yml.template # Main configuration template
│   ├── sources.yml.template # API keys template
│   └── ...
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
│   └── bump_version.py     # Version management
├── .github/workflows/       # CI/CD pipelines
├── pyproject.toml          # Modern Python packaging configuration
├── setup.py               # Legacy setup (maintained for compatibility)
└── README.md              # Project overview
```

## Release Process

### Version Management

This project uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Creating a Release

1. **Update version**:
   ```bash
   python scripts/bump_version.py [major|minor|patch]
   ```

2. **Update CHANGELOG.md** with release notes

3. **Commit and tag**:
   ```bash
   git commit -am "Bump version to X.Y.Z"
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   ```

4. **Push changes**:
   ```bash
   git push origin main --tags
   ```

5. **GitHub Actions will automatically publish to PyPI**

## Getting Help

- **Issues**: Open an issue on GitHub for bug reports or feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Documentation**: Check the `docs/` directory for comprehensive guides
- **Code Review**: Pull requests receive thorough code review and feedback

## Security

- Never commit API keys or sensitive information
- Use the template files for configuration
- Report security vulnerabilities privately through GitHub Security Advisories

Thank you for contributing to AlphaPy! 🚀