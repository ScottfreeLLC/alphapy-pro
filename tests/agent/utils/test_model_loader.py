"""Tests for agent.utils.model_loader module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agent.utils.model_loader import ModelLoader


class TestModelLoaderInit:
    """Tests for ModelLoader initialization."""

    def test_init_with_path(self, tmp_path):
        """Test initialization with path."""
        loader = ModelLoader(run_dir=tmp_path, algo="xgb")

        assert loader.run_dir == tmp_path
        assert loader.algo == "xgb"
        assert loader.is_loaded is False

    def test_init_default_algo(self, tmp_path):
        """Test default algorithm is xgb."""
        loader = ModelLoader(run_dir=tmp_path)

        assert loader.algo == "xgb"


class TestModelLoaderLoad:
    """Tests for load method."""

    def test_load_success(self, mock_xgb_model):
        """Test successful model loading."""
        loader = ModelLoader(run_dir=mock_xgb_model["run_dir"], algo="xgb")
        loader.load()

        assert loader.is_loaded is True
        assert loader.predictor is not None
        assert loader.feature_names is not None

    def test_load_missing_model_dir(self, tmp_path):
        """Test error when model directory is missing."""
        loader = ModelLoader(run_dir=tmp_path, algo="xgb")

        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            loader.load()

    def test_load_missing_predictor(self, tmp_path):
        """Test error when predictor file is missing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        loader = ModelLoader(run_dir=tmp_path, algo="xgb")

        with pytest.raises(FileNotFoundError, match="No predictor file found"):
            loader.load()

    def test_load_returns_self(self, mock_xgb_model):
        """Test load returns self for chaining."""
        loader = ModelLoader(run_dir=mock_xgb_model["run_dir"], algo="xgb")
        result = loader.load()

        assert result is loader


class TestModelLoaderPredict:
    """Tests for predict method."""

    @pytest.fixture
    def loaded_loader(self, mock_xgb_model):
        """Create a loaded model loader."""
        loader = ModelLoader(run_dir=mock_xgb_model["run_dir"], algo="xgb")
        loader.load()
        return loader

    def test_predict_not_loaded(self, tmp_path):
        """Test predict raises error when model not loaded."""
        loader = ModelLoader(run_dir=tmp_path, algo="xgb")

        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.predict(pd.DataFrame())

    def test_predict_returns_tuple(self, loaded_loader, mock_xgb_model):
        """Test predict returns (predictions, probabilities)."""
        feature_names = mock_xgb_model["feature_names"]
        df = pd.DataFrame(
            np.random.randn(10, len(feature_names)),
            columns=feature_names
        )

        preds, probas = loaded_loader.predict(df)

        assert preds is not None
        assert len(preds) == 10
        assert probas is not None

    def test_predict_handles_missing_features(self, loaded_loader, mock_xgb_model):
        """Test predict handles missing features."""
        # Create df with only some features
        df = pd.DataFrame({
            mock_xgb_model["feature_names"][0]: [1.0, 2.0, 3.0]
        })

        # Should not raise, will fill missing with NaN
        preds, _ = loaded_loader.predict(df)
        assert len(preds) == 3


class TestModelLoaderPredictLatest:
    """Tests for predict_latest method."""

    @pytest.fixture
    def loaded_loader(self, mock_xgb_model):
        """Create a loaded model loader."""
        loader = ModelLoader(run_dir=mock_xgb_model["run_dir"], algo="xgb")
        loader.load()
        return loader

    def test_predict_latest_returns_dict(self, loaded_loader, mock_xgb_model):
        """Test predict_latest returns dictionary."""
        feature_names = mock_xgb_model["feature_names"]
        df = pd.DataFrame(
            np.random.randn(10, len(feature_names)),
            columns=feature_names
        )

        result = loaded_loader.predict_latest(df)

        assert isinstance(result, dict)
        assert "prediction" in result

    def test_predict_latest_includes_probability(self, loaded_loader, mock_xgb_model):
        """Test predict_latest includes probability for classifiers."""
        feature_names = mock_xgb_model["feature_names"]
        df = pd.DataFrame(
            np.random.randn(10, len(feature_names)),
            columns=feature_names
        )

        result = loaded_loader.predict_latest(df)

        assert "probability" in result


class TestModelLoaderInfo:
    """Tests for model information methods."""

    @pytest.fixture
    def loaded_loader(self, mock_xgb_model):
        """Create a loaded model loader."""
        loader = ModelLoader(run_dir=mock_xgb_model["run_dir"], algo="xgb")
        loader.load()
        return loader

    def test_get_model_info_unloaded(self, tmp_path):
        """Test get_model_info when not loaded."""
        loader = ModelLoader(run_dir=tmp_path, algo="xgb")
        info = loader.get_model_info()

        assert info["is_loaded"] is False
        assert info["n_features"] == 0

    def test_get_model_info_loaded(self, loaded_loader):
        """Test get_model_info when loaded."""
        info = loaded_loader.get_model_info()

        assert info["is_loaded"] is True
        assert info["n_features"] > 0
        assert "model_type" in info

    def test_get_feature_importance_unloaded(self, tmp_path):
        """Test feature importance when not loaded."""
        loader = ModelLoader(run_dir=tmp_path, algo="xgb")
        result = loader.get_feature_importance()

        assert result is None

    def test_get_feature_importance_loaded(self, loaded_loader):
        """Test feature importance when loaded."""
        result = loaded_loader.get_feature_importance()

        assert result is not None
        assert "feature" in result.columns
        assert "importance" in result.columns
