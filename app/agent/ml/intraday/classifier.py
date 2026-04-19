"""XGBoost multiclass classifier for intraday pattern detection.

Outputs:
- Predicted pattern (IntradayPattern enum)
- Probability for each pattern class
- Top-3 most likely patterns

Persisted via joblib to app/data/models/intraday_classifier_{date}.joblib
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .patterns import IntradayPattern, NUM_CLASSES, PATTERN_NAMES

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "data", "models",
)


class IntradayClassifier:
    """
    XGBoost multiclass classifier for intraday pattern detection.

    Uses multi:softprob objective for probability outputs with class weighting
    to handle imbalanced pattern frequencies (NO_PATTERN dominates).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self._loaded = False
        self.training_metrics: Optional[Dict] = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_X: Optional[pd.DataFrame] = None,
        eval_y: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Train the multiclass classifier.

        Args:
            X: Feature matrix (from build_intraday_features / build_multi_session_features).
            y: IntradayPattern integer labels.
            eval_X: Optional held-out feature matrix for evaluation.
            eval_y: Optional held-out labels.

        Returns:
            Dict with training and evaluation metrics.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("xgboost not installed. Run: uv add xgboost")
            return {"error": "xgboost not installed"}

        # Align
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) < 100:
            return {"error": f"Insufficient training data: {len(X)} samples (need 100+)"}

        self.feature_names = list(X.columns)

        # Ensure all classes are represented (XGBoost requires contiguous labels)
        # Add a tiny synthetic row per missing class with near-zero weight
        present_classes = set(y.unique())
        all_classes = set(range(NUM_CLASSES))
        missing = all_classes - present_classes
        if missing:
            pad_rows = pd.DataFrame(
                np.zeros((len(missing), len(self.feature_names))),
                columns=self.feature_names,
            )
            pad_labels = pd.Series(list(missing))
            X = pd.concat([X, pad_rows], ignore_index=True)
            y = pd.concat([y, pad_labels], ignore_index=True)

        # Compute class weights (inverse frequency)
        class_counts = y.value_counts()
        total = len(y)
        sample_weights = y.map(lambda c: total / (NUM_CLASSES * max(1, class_counts.get(c, 1))))
        # Near-zero weight for synthetic padding rows
        if missing:
            sample_weights.iloc[-len(missing):] = 1e-6

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            num_class=NUM_CLASSES,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        )

        self.model.fit(
            X.values, y.values,
            sample_weight=sample_weights.values,
        )
        self._loaded = True

        # Training metrics
        metrics = self._compute_metrics(X, y, prefix="train")

        # Evaluation metrics
        if eval_X is not None and eval_y is not None:
            eval_common = eval_X.index.intersection(eval_y.index)
            eval_X = eval_X.loc[eval_common]
            eval_y = eval_y.loc[eval_common]
            if len(eval_X) > 0:
                eval_metrics = self._compute_metrics(eval_X, eval_y, prefix="eval")
                metrics.update(eval_metrics)

        metrics["n_features"] = len(self.feature_names)
        metrics["n_classes"] = NUM_CLASSES

        self.training_metrics = metrics
        logger.info(f"IntradayClassifier trained: {metrics}")
        return metrics

    def predict(self, features: pd.DataFrame) -> List[Dict]:
        """
        Predict patterns for a batch of bars.

        Args:
            features: Feature matrix (same columns as training).

        Returns:
            List of prediction dicts with: pattern, probability, top_3.
        """
        if not self._loaded or self.model is None:
            return [_default_prediction() for _ in range(len(features))]

        X = self._align_features(features)
        probas = self.model.predict_proba(X)

        results = []
        for i in range(len(probas)):
            row_proba = probas[i]
            pred_class = int(np.argmax(row_proba))
            pred_proba = float(row_proba[pred_class])

            # Top 3
            top_3_idx = np.argsort(row_proba)[-3:][::-1]
            top_3 = [
                {
                    "pattern": IntradayPattern(idx).name,
                    "probability": round(float(row_proba[idx]), 4),
                }
                for idx in top_3_idx
            ]

            results.append({
                "pattern": IntradayPattern(pred_class).name,
                "probability": round(pred_proba, 4),
                "top_3": top_3,
                "all_probabilities": {
                    IntradayPattern(c).name: round(float(row_proba[c]), 4)
                    for c in range(NUM_CLASSES)
                },
            })

        return results

    def predict_single(self, features: Dict) -> Dict:
        """
        Predict pattern for a single bar.

        Args:
            features: Flat dict of feature values.

        Returns:
            Prediction dict with: pattern, probability, top_3.
        """
        if not self._loaded or self.model is None:
            return _default_prediction()

        row = [features.get(fname, 0.0) for fname in self.feature_names]
        X = np.array([row])
        probas = self.model.predict_proba(X)[0]

        pred_class = int(np.argmax(probas))
        pred_proba = float(probas[pred_class])

        top_3_idx = np.argsort(probas)[-3:][::-1]
        top_3 = [
            {
                "pattern": IntradayPattern(idx).name,
                "probability": round(float(probas[idx]), 4),
            }
            for idx in top_3_idx
        ]

        return {
            "pattern": IntradayPattern(pred_class).name,
            "probability": round(pred_proba, 4),
            "top_3": top_3,
        }

    def save(self, path: Optional[str] = None) -> str:
        """Save model to disk. Returns the save path."""
        if not self._loaded or self.model is None:
            logger.warning("No model to save")
            return ""

        try:
            import joblib
        except ImportError:
            logger.error("joblib not installed")
            return ""

        save_path = path or os.path.join(
            MODEL_DIR,
            f"intraday_classifier_{datetime.now().strftime('%Y%m%d')}.joblib",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "created_at": datetime.now().isoformat(),
        }, save_path)

        logger.info(f"IntradayClassifier saved to {save_path}")
        return save_path

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
        self.training_metrics = data.get("training_metrics")
        self._loaded = True
        logger.info(
            f"IntradayClassifier loaded from {path} "
            f"({len(self.feature_names)} features, {NUM_CLASSES} classes)"
        )

    @staticmethod
    def find_latest_model() -> Optional[str]:
        """Find the most recent saved model."""
        if not os.path.exists(MODEL_DIR):
            return None

        models = sorted([
            f for f in os.listdir(MODEL_DIR)
            if f.startswith("intraday_classifier_") and f.endswith(".joblib")
        ])
        if models:
            return os.path.join(MODEL_DIR, models[-1])
        return None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _align_features(self, features: pd.DataFrame) -> np.ndarray:
        """Align feature columns with training features."""
        aligned = pd.DataFrame(index=features.index)
        for fname in self.feature_names:
            aligned[fname] = features[fname] if fname in features.columns else 0.0
        return aligned.values

    def _compute_metrics(self, X: pd.DataFrame, y: pd.Series, prefix: str) -> Dict:
        """Compute classification metrics."""
        from sklearn.metrics import accuracy_score, classification_report

        pred = self.model.predict(X.values)
        accuracy = accuracy_score(y.values, pred)

        # Per-class metrics
        report = classification_report(
            y.values, pred,
            labels=list(range(NUM_CLASSES)),
            target_names=[IntradayPattern(i).name for i in range(NUM_CLASSES)],
            output_dict=True,
            zero_division=0,
        )

        # Extract per-pattern precision/recall
        per_pattern = {}
        for i in range(NUM_CLASSES):
            name = IntradayPattern(i).name
            if name in report:
                per_pattern[name] = {
                    "precision": round(report[name]["precision"], 4),
                    "recall": round(report[name]["recall"], 4),
                    "f1": round(report[name]["f1-score"], 4),
                    "support": int(report[name]["support"]),
                }

        return {
            f"{prefix}_accuracy": round(float(accuracy), 4),
            f"{prefix}_samples": len(X),
            f"{prefix}_per_pattern": per_pattern,
            f"{prefix}_class_distribution": y.value_counts().to_dict(),
        }


def _default_prediction() -> Dict:
    """Return a default prediction when no model is loaded."""
    return {
        "pattern": IntradayPattern.NO_PATTERN.name,
        "probability": 1.0,
        "top_3": [{"pattern": IntradayPattern.NO_PATTERN.name, "probability": 1.0}],
    }
