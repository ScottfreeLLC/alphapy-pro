"""Composite confidence scoring with calibration."""

import logging
from typing import Dict, Optional

from ..core.signal import TradeSignal
from .performance import PerformanceTracker

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Composite confidence score combining:
      1. Claude's raw confidence from skill evaluation
      2. Historical accuracy of the generating skill
      3. Risk/reward quality of the signal
      4. ML meta-label probability (if model loaded)

    Calibrated over time based on actual outcomes.
    """

    def __init__(self, performance: PerformanceTracker, ml_model=None):
        self.performance = performance
        self.ml_model = ml_model

        # Weights with ML component
        if ml_model and ml_model.is_loaded:
            self.weight_raw = 0.35
            self.weight_skill_accuracy = 0.25
            self.weight_risk_reward = 0.15
            self.weight_ml = 0.25
        else:
            # 3-component fallback
            self.weight_raw = 0.50
            self.weight_skill_accuracy = 0.30
            self.weight_risk_reward = 0.20
            self.weight_ml = 0.0

    def set_ml_model(self, ml_model):
        """Set or update the ML model and recalculate weights."""
        self.ml_model = ml_model
        if ml_model and ml_model.is_loaded:
            self.weight_raw = 0.35
            self.weight_skill_accuracy = 0.25
            self.weight_risk_reward = 0.15
            self.weight_ml = 0.25
        else:
            self.weight_raw = 0.50
            self.weight_skill_accuracy = 0.30
            self.weight_risk_reward = 0.20
            self.weight_ml = 0.0

    def score(self, signal: TradeSignal) -> float:
        """
        Calculate a composite confidence score for a signal.

        Returns adjusted confidence between 0.0 and 1.0.
        """
        # 1. Raw confidence from Claude
        raw = signal.confidence

        # 2. Historical skill accuracy
        skill_accuracy = self._get_skill_accuracy(signal.skill_name)

        # 3. Risk/reward quality (normalize: 1.5 R:R = 0.5, 3.0+ = 1.0)
        rr = signal.risk_reward_ratio
        rr_score = min(1.0, max(0.0, (rr - 1.0) / 2.0))

        # 4. ML meta-label probability
        ml_score = 0.5  # Neutral default
        if self.ml_model and self.ml_model.is_loaded:
            features = signal.metadata.get("ml_features", {})
            direction = 1 if signal.direction.value == "long" else -1
            ml_proba, _ = self.ml_model.predict(features, raw, direction)
            ml_score = ml_proba
            signal.metadata["ml_probability"] = ml_proba

        composite = (
            self.weight_raw * raw
            + self.weight_skill_accuracy * skill_accuracy
            + self.weight_risk_reward * rr_score
            + self.weight_ml * ml_score
        )

        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        logger.debug(
            f"Confidence score for {signal.symbol}/{signal.skill_name}: "
            f"raw={raw:.2f}, skill_acc={skill_accuracy:.2f}, rr={rr_score:.2f}, "
            f"ml={ml_score:.2f} → {composite:.2f}"
        )
        return round(composite, 3)

    def _get_skill_accuracy(self, skill_name: str) -> float:
        """Get historical win rate for a skill. Returns 0.5 if no history."""
        skill_metrics = self.performance.get_skill_metrics()
        metrics = skill_metrics.get(skill_name)

        if not metrics or metrics.get("total_trades", 0) < 5:
            return 0.5  # Neutral until we have enough data

        return metrics.get("win_rate", 0.5)

    def calibrate(self, signal: TradeSignal, outcome_profitable: bool):
        """
        Record outcome for future calibration.

        This is called after a trade closes to improve future scoring.
        The actual calibration happens implicitly via performance.get_skill_metrics().
        """
        logger.debug(
            f"Calibration data: {signal.skill_name} confidence={signal.confidence} "
            f"profitable={outcome_profitable}"
        )
