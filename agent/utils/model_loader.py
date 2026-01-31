"""Model loader for trained AlphaPy models."""

import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage trained AlphaPy models for prediction.

    Handles loading XGBoost (or other sklearn-compatible) models
    from AlphaPy run directories.
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        algo: str = "xgb",
    ):
        """Initialize model loader.

        Args:
            run_dir: Path to AlphaPy run directory containing model files
            algo: Algorithm name (e.g., 'xgb', 'rf', 'lgbm')
        """
        self.run_dir = Path(run_dir)
        self.algo = algo

        self.predictor = None
        self.feature_map: Optional[dict] = None
        self.feature_names: Optional[list[str]] = None
        self.model_config: Optional[dict] = None

        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def load(self) -> "ModelLoader":
        """Load model and feature map from run directory.

        Returns:
            Self for method chaining.
        """
        model_dir = self.run_dir / "model"

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Find predictor file
        predictor_pattern = f"{self.algo}*.pkl"
        predictor_files = list(model_dir.glob(predictor_pattern))

        if not predictor_files:
            # Try without algorithm prefix
            predictor_files = list(model_dir.glob("*predictor*.pkl"))

        if not predictor_files:
            raise FileNotFoundError(
                f"No predictor file found matching {predictor_pattern} in {model_dir}"
            )

        predictor_file = predictor_files[0]
        logger.info(f"Loading predictor from: {predictor_file}")
        self.predictor = joblib.load(predictor_file)

        # Load feature map if available
        feature_map_file = model_dir / "feature_map.pkl"
        if feature_map_file.exists():
            logger.info(f"Loading feature map from: {feature_map_file}")
            self.feature_map = joblib.load(feature_map_file)
            self.feature_names = list(self.feature_map.keys())
        else:
            # Try to get feature names from model
            if hasattr(self.predictor, "feature_names_in_"):
                self.feature_names = list(self.predictor.feature_names_in_)
            elif hasattr(self.predictor, "get_booster"):
                # XGBoost specific
                self.feature_names = self.predictor.get_booster().feature_names
            logger.warning("No feature_map.pkl found, using model's feature names")

        # Load model config if available
        config_file = self.run_dir / "config" / "model.yml"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                self.model_config = yaml.safe_load(f)

        self._is_loaded = True
        logger.info(
            f"Model loaded: {len(self.feature_names or [])} features, "
            f"algorithm: {self.algo}"
        )

        return self

    def predict(
        self,
        df: pd.DataFrame,
        return_proba: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate predictions from feature DataFrame.

        Args:
            df: DataFrame with feature columns
            return_proba: If True, return probabilities for classifiers

        Returns:
            Tuple of (predictions, probabilities). Probabilities is None
            for regressors or if return_proba is False.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Select and order features to match training
        if self.feature_names:
            # Check for missing features
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                logger.warning(f"Missing features: {missing}")
                # Fill missing with NaN (model may handle this)
                for feat in missing:
                    df[feat] = np.nan

            X = df[self.feature_names].values
        else:
            logger.warning("No feature names available, using all numeric columns")
            X = df.select_dtypes(include=[np.number]).values

        # Handle NaN values
        if np.isnan(X).any():
            logger.warning("NaN values in features, filling with 0")
            X = np.nan_to_num(X, nan=0.0)

        # Get predictions
        predictions = self.predictor.predict(X)

        # Get probabilities if classifier
        probabilities = None
        if return_proba and hasattr(self.predictor, "predict_proba"):
            try:
                proba = self.predictor.predict_proba(X)
                # For binary classification, get probability of positive class
                if proba.ndim == 2 and proba.shape[1] == 2:
                    probabilities = proba[:, 1]
                else:
                    probabilities = proba
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")

        return predictions, probabilities

    def predict_latest(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """Predict on the latest row only.

        Args:
            df: DataFrame with feature columns

        Returns:
            Dictionary with prediction and probability for the latest row.
        """
        # Get just the last row
        last_row = df.tail(1)
        preds, probas = self.predict(last_row)

        result = {
            "prediction": int(preds[0]) if len(preds) > 0 else None,
        }

        if probas is not None and len(probas) > 0:
            result["probability"] = float(probas[0])

        return result

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the model.

        Returns:
            DataFrame with feature names and importance scores,
            or None if not available.
        """
        if not self._is_loaded:
            return None

        importance = None

        # Try XGBoost style
        if hasattr(self.predictor, "feature_importances_"):
            importance = self.predictor.feature_importances_
        elif hasattr(self.predictor, "get_booster"):
            try:
                booster = self.predictor.get_booster()
                importance_dict = booster.get_score(importance_type="gain")
                if self.feature_names:
                    importance = [
                        importance_dict.get(f, 0) for f in self.feature_names
                    ]
            except Exception:
                pass

        if importance is not None and self.feature_names:
            return pd.DataFrame({
                "feature": self.feature_names,
                "importance": importance,
            }).sort_values("importance", ascending=False)

        return None

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model metadata.
        """
        info = {
            "run_dir": str(self.run_dir),
            "algorithm": self.algo,
            "is_loaded": self._is_loaded,
            "n_features": len(self.feature_names) if self.feature_names else 0,
        }

        if self._is_loaded and self.predictor:
            info["model_type"] = type(self.predictor).__name__

            # Check if classifier or regressor
            if hasattr(self.predictor, "predict_proba"):
                info["model_class"] = "classifier"
            else:
                info["model_class"] = "regressor"

            # Get n_estimators if available
            if hasattr(self.predictor, "n_estimators"):
                info["n_estimators"] = self.predictor.n_estimators

        return info
