"""Signal Generator tool for ML-based trading signals."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import polars as pl

from .base import Tool
from ..utils.feature_calculator import FeatureCalculator
from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class SignalGeneratorTool(Tool):
    """Tool for generating trading signals using ML models.

    Uses talipp indicators and XGBoost models to generate
    buy/sell signals with probability scores.
    """

    name: str = "generate_signals"
    description: str = """
Generates trading signals using the trained ML model.
Takes bar data and produces signals with probabilities for each symbol.
Returns signals filtered by probability threshold.
Use this after fetching market data to determine trading opportunities.
"""
    input_schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "bar_data": {
                "type": "string",
                "description": "JSON string of bar data from market_data tool",
            },
            "prob_min": {
                "type": "number",
                "default": 0.55,
                "minimum": 0.5,
                "maximum": 1.0,
                "description": "Minimum probability threshold for signals",
            },
            "signal_type": {
                "type": "string",
                "enum": ["long_only", "short_only", "both"],
                "default": "long_only",
                "description": "Type of signals to generate",
            },
        },
        "required": ["bar_data"],
    })

    _feature_calculator: FeatureCalculator = field(default=None, repr=False)
    _model_loader: ModelLoader = field(default=None, repr=False)
    _run_dir: str = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize feature calculator and model loader."""
        if self._feature_calculator is None:
            self._feature_calculator = FeatureCalculator()

    def load_model(self, run_dir: str, algo: str = "xgb") -> None:
        """Load the ML model from a run directory.

        Args:
            run_dir: Path to AlphaPy run directory
            algo: Algorithm name
        """
        self._run_dir = run_dir
        self._model_loader = ModelLoader(run_dir, algo)
        self._model_loader.load()
        logger.info(f"Model loaded from {run_dir}")

    async def execute(
        self,
        bar_data: str,
        prob_min: float = 0.55,
        signal_type: str = "long_only",
    ) -> str:
        """Generate trading signals from bar data.

        Returns:
            JSON string with signals for each symbol.
        """
        if self._model_loader is None or not self._model_loader.is_loaded:
            return json.dumps({
                "error": "Model not loaded. Call load_model() first.",
                "signals": {},
            })

        try:
            # Parse bar data
            data = json.loads(bar_data)

            signals = {}
            for symbol, symbol_data in data.items():
                if symbol.startswith("_"):  # Skip metadata
                    continue

                if "bars" not in symbol_data:
                    continue

                # Convert bars to Polars DataFrame
                bars = symbol_data["bars"]
                if not bars:
                    continue

                df = pl.DataFrame(bars)

                # Compute indicators
                df_features = self._feature_calculator.compute_single(df, symbol)

                # Get prediction for latest bar
                result = self._model_loader.predict_latest(df_features)

                prediction = result.get("prediction")
                probability = result.get("probability", 0.0)

                # Determine signal
                signal = self._determine_signal(
                    prediction=prediction,
                    probability=probability,
                    prob_min=prob_min,
                    signal_type=signal_type,
                )

                signals[symbol] = {
                    "prediction": prediction,
                    "probability": round(probability, 4) if probability else None,
                    "signal": signal,
                    "latest_close": symbol_data.get("latest_close"),
                    "latest_time": symbol_data.get("latest_time"),
                }

            # Filter to only actionable signals
            actionable = {
                k: v for k, v in signals.items()
                if v["signal"] in ["long", "short"]
            }

            return json.dumps({
                "signals": signals,
                "actionable": actionable,
                "summary": {
                    "total_symbols": len(signals),
                    "long_signals": sum(1 for v in signals.values() if v["signal"] == "long"),
                    "short_signals": sum(1 for v in signals.values() if v["signal"] == "short"),
                    "no_signal": sum(1 for v in signals.values() if v["signal"] == "none"),
                    "prob_threshold": prob_min,
                },
            })

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return json.dumps({
                "error": str(e),
                "signals": {},
            })

    def _determine_signal(
        self,
        prediction: Optional[int],
        probability: Optional[float],
        prob_min: float,
        signal_type: str,
    ) -> str:
        """Determine signal from prediction and probability.

        Returns:
            Signal string: "long", "short", or "none"
        """
        if prediction is None or probability is None:
            return "none"

        # For binary classification:
        # prediction = 1 typically means positive (long)
        # prediction = 0 typically means negative (short or no trade)

        if prediction == 1 and probability >= prob_min:
            if signal_type in ["long_only", "both"]:
                return "long"
        elif prediction == 0:
            # For short signals, we look at probability of being wrong
            short_prob = 1 - probability
            if short_prob >= prob_min and signal_type in ["short_only", "both"]:
                return "short"

        return "none"
