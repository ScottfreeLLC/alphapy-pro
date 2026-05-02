"""
Pytest configuration and shared fixtures for AlphaPy tests.
"""
import asyncio
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ============================================================================
# Async Support
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Directory Fixtures
# ============================================================================

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
def memory_temp_dir(tmp_path):
    """Create temporary directory for memory tool tests."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


# ============================================================================
# Mock Model Fixtures
# ============================================================================

@pytest.fixture
def mock_xgb_model(tmp_path):
    """Create a mock model with the on-disk layout AlphaPy expects."""
    from sklearn.ensemble import GradientBoostingClassifier
    import joblib

    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)

    model_dir = tmp_path / "model"
    config_dir = tmp_path / "config"
    model_dir.mkdir()
    config_dir.mkdir()

    joblib.dump(model, model_dir / "xgb_predictor.pkl")

    feature_names = [f"feature_{i}" for i in range(10)]
    feature_map = {name: i for i, name in enumerate(feature_names)}
    joblib.dump(feature_map, model_dir / "feature_map.pkl")

    config = {"target": "target", "algorithms": ["xgb"]}
    import yaml
    with open(config_dir / "model.yml", "w") as f:
        yaml.dump(config, f)

    return {
        "run_dir": tmp_path,
        "model": model,
        "feature_names": feature_names,
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_alphapy_config(temp_dir):
    """Create a mock AlphaPy configuration for testing."""
    return {
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
