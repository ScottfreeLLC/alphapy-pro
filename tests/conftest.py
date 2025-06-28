"""
Pytest configuration and shared fixtures for AlphaPy tests.
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data_dir():
    """Path to sample data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_config_dir():
    """Path to sample config directory."""
    return Path(__file__).parent / "config"


@pytest.fixture
def mock_alphapy_config(temp_dir):
    """Create a mock AlphaPy configuration for testing."""
    config = {
        'directory': temp_dir,
        'file_extension': 'csv',
        'separator': ',',
        'target': 'target',
        'algorithms': ['rf', 'xgb'],
        'cv_folds': 3,
        'lag_period': 1,
        'leaders': 1,
        'predict_mode': False,
        'predict_history': False,
        'score_validation': False,
        'split': 0.4,
        'test_size': 0.2,
        'validation_size': 0.2,
    }
    return config