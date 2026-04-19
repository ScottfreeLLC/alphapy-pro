"""LLM integration (via LiteLLM) for skill evaluation and signal synthesis."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import litellm

from ..config import AgentConfig
from .signal import SignalDirection, TradeSignal

logger = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = """You are Alfi, an AI trading agent. You evaluate trading strategies (called "skills") against current market data to determine if trade conditions are met.

IMPORTANT: The market data includes pre-computed technical indicators under the "indicators" key and a human-readable "trend_summary". USE THESE — do not attempt to compute indicators from raw bars.

Key indicator fields available:
- indicators.trend: sma20, sma50, ema20, ema50, price_vs_sma20_pct, above_sma20, above_sma50, golden_cross
- indicators.momentum: rsi_14, rsi_zone, macd_line, macd_signal, macd_histogram, macd_bullish, roc_10
- indicators.volatility: bb_upper, bb_middle, bb_lower, bb_pct_b, bb_position, atr_14, atr_pct, hist_volatility_20d
- indicators.volume: volume_sma20, volume_ratio, volume_surge, obv, obv_trend, volume_trend
- indicators.price_context: daily_change_pct, high_20d, low_20d, dist_from_20d_high_pct, near_20d_high, near_20d_low

Use "trend_summary" for quick orientation, then check specific indicator values to evaluate the strategy conditions.

You MUST respond with valid JSON only. No markdown, no code fences, no extra text.

Response schema:
{
  "conditions_met": true/false,
  "confidence": 0.0-1.0,
  "direction": "long" or "short",
  "entry_price": number,
  "stop_loss": number,
  "take_profit": number,
  "reasoning": "Brief explanation of why conditions are/aren't met"
}

If conditions are NOT met, set confidence to 0 and use current price for entry/stop/target fields."""

SYNTHESIS_SYSTEM_PROMPT = """You are Alfi, an AI trading agent synthesizing multiple trade signals into a coherent plan.

Given a set of trade signals from different strategies, identify:
1. Confirming signals (multiple strategies agree)
2. Conflicting signals (strategies disagree)
3. Overall portfolio-level assessment

You MUST respond with valid JSON only:
{
  "assessment": "Brief overall assessment",
  "top_signals": [
    {
      "symbol": "TICKER",
      "direction": "long/short",
      "confidence": 0.0-1.0,
      "reasoning": "Why this is the top pick"
    }
  ],
  "conflicts": ["List of conflicting symbols"],
  "skip_symbols": ["Symbols to avoid right now"]
}"""


class ReasoningClient:
    """Wraps LLM API via LiteLLM for skill evaluation and signal synthesis."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def evaluate_skill(
        self,
        skill_markdown: str,
        market_data: Dict[str, Any],
        symbol: str,
    ) -> Optional[Dict]:
        """
        Send skill + market data to Claude for evaluation.

        Returns parsed JSON with conditions_met, confidence, entry/stop/target, reasoning.
        """
        user_prompt = f"""## Symbol: {symbol}

## Market Data
```json
{json.dumps(market_data, indent=2, default=str)}
```

## Strategy (Skill)
{skill_markdown}

Evaluate whether this strategy's entry conditions are currently met for {symbol}.
Respond with JSON only."""

        try:
            response = litellm.completion(
                model=self.config.eval_model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            text = response.choices[0].message.content.strip()
            # Strip code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
                text = text.strip()

            result = json.loads(text)
            logger.info(
                f"Skill eval for {symbol}: conditions_met={result.get('conditions_met')}, "
                f"confidence={result.get('confidence')}"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM API error evaluating {symbol}: {e}")
            return None

    async def aevaluate_skill(
        self,
        skill_markdown: str,
        market_data: Dict[str, Any],
        symbol: str,
    ) -> Optional[Dict]:
        """Async wrapper — runs evaluate_skill in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(
            self.evaluate_skill, skill_markdown, market_data, symbol
        )

    def synthesize_signals(self, signals: List[TradeSignal]) -> Optional[Dict]:
        """
        Synthesize multiple signals into a coherent trade plan using Opus.

        Used when there are many signals to reconcile into a portfolio-level view.
        """
        if not signals:
            return None

        signals_data = [s.to_dict() for s in signals]
        user_prompt = f"""## Active Trade Signals
```json
{json.dumps(signals_data, indent=2, default=str)}
```

Synthesize these signals into a coherent trading plan. Identify confirmations, conflicts, and top picks."""

        try:
            response = litellm.completion(
                model=self.config.synthesis_model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
                text = text.strip()

            return json.loads(text)

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Signal synthesis failed: {e}")
            return None

    def build_market_snapshot(self, bars_data: Dict, pattern_data: Optional[Dict] = None) -> Dict:
        """
        Build a structured market snapshot for Claude from raw bar data.

        Args:
            bars_data: OHLCV bar data (list of dicts or similar)
            pattern_data: Optional pivot pattern analysis results
        """
        snapshot = {
            "bars": bars_data,
        }
        if pattern_data:
            snapshot["patterns"] = {
                "detected": pattern_data.get("summary", {}).get("detected_patterns", []),
                "sentiment": pattern_data.get("summary", {}).get("overall_sentiment", "neutral"),
                "bullish_count": pattern_data.get("summary", {}).get("bullish_patterns_count", 0),
                "bearish_count": pattern_data.get("summary", {}).get("bearish_patterns_count", 0),
            }
        return snapshot
