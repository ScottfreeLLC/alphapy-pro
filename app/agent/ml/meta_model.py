"""Meta-labeling model: XGBoost classifier that predicts probability of signal success."""

import logging
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models")


class MetaModel:
    """
    Meta-labeling model from de Prado's AFML.

    Primary signals come from skill evaluations (LLM).
    This model takes: signal direction + feature vector + LLM confidence
    and predicts: probability of success + suggested size multiplier.

    Uses XGBoost for tabular financial data.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names = None
        self.model_path = model_path
        self._loaded = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        signal_confidence: Optional[pd.Series] = None,
        signal_direction: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Train the meta-labeling model.

        Args:
            X: Feature matrix (from features.build_feature_matrix)
            y: Labels from triple barrier method (+1/-1/0)
            signal_confidence: LLM confidence for each signal (optional)
            signal_direction: Signal direction (+1 long, -1 short) (optional)

        Returns:
            Dict with training metrics
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("xgboost not installed. Run: uv add xgboost")
            return {"error": "xgboost not installed"}

        # Add signal features if available
        train_X = X.copy()
        if signal_confidence is not None:
            train_X["signal_confidence"] = signal_confidence.reindex(X.index).fillna(0.5)
        if signal_direction is not None:
            train_X["signal_direction"] = signal_direction.reindex(X.index).fillna(1)

        # Convert labels to binary: success (1) vs failure (0)
        y_binary = (y == 1).astype(int)

        # Align indices
        common_idx = train_X.index.intersection(y_binary.index)
        train_X = train_X.loc[common_idx]
        y_binary = y_binary.loc[common_idx]

        if len(train_X) < 50:
            return {"error": f"Insufficient training data: {len(train_X)} samples (need 50+)"}

        self.feature_names = list(train_X.columns)

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=len(y_binary[y_binary == 0]) / max(1, len(y_binary[y_binary == 1])),
            random_state=42,
            eval_metric="logloss",
        )

        self.model.fit(train_X.values, y_binary.values)
        self._loaded = True

        # Training metrics
        train_pred = self.model.predict(train_X.values)
        train_proba = self.model.predict_proba(train_X.values)[:, 1]

        accuracy = (train_pred == y_binary.values).mean()
        precision = train_pred[train_pred == 1].sum() / max(1, train_pred.sum())

        metrics = {
            "samples": len(train_X),
            "features": len(self.feature_names),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "positive_rate": round(float(y_binary.mean()), 4),
            "avg_probability": round(float(train_proba.mean()), 4),
        }

        logger.info(f"MetaModel trained: {metrics}")
        return metrics

    def predict(
        self,
        features: Dict,
        signal_confidence: float = 0.5,
        signal_direction: int = 1,
    ) -> Tuple[float, float]:
        """
        Predict probability of success and suggested size multiplier.

        Args:
            features: Flat dict of feature values (from FeatureEngine.compute_features_df)
            signal_confidence: LLM confidence
            signal_direction: +1 for long, -1 for short

        Returns:
            (probability_of_success, suggested_size_multiplier)
        """
        if not self._loaded or self.model is None:
            return 0.5, 1.0  # Neutral defaults

        # Build feature vector
        feature_dict = dict(features)
        feature_dict["signal_confidence"] = signal_confidence
        feature_dict["signal_direction"] = signal_direction

        # Align with training features
        row = []
        for fname in self.feature_names:
            row.append(feature_dict.get(fname, 0.0))

        X = np.array([row])
        proba = float(self.model.predict_proba(X)[0, 1])

        # Size multiplier: scale from 0.5x (low prob) to 2x (high prob)
        size_mult = 0.5 + (proba * 1.5)
        size_mult = round(max(0.5, min(2.0, size_mult)), 2)

        return round(proba, 4), size_mult

    def save(self, path: Optional[str] = None):
        """Save model to disk."""
        if not self._loaded or self.model is None:
            logger.warning("No model to save")
            return

        try:
            import joblib
        except ImportError:
            logger.error("joblib not installed")
            return

        save_path = path or self.model_path
        if not save_path:
            os.makedirs(MODEL_DIR, exist_ok=True)
            save_path = os.path.join(MODEL_DIR, f"meta_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, save_path)
        logger.info(f"MetaModel saved to {save_path}")

    def load(self, path: str):
        """Load model from disk."""
        try:
            import joblib
        except ImportError:
            logger.error("joblib not installed")
            return

        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self._loaded = True
        logger.info(f"MetaModel loaded from {path} ({len(self.feature_names)} features)")

    @property
    def is_loaded(self) -> bool:
        return self._loaded
