"""FeatureEngine: enriches raw OHLCV snapshots with computed technical indicators."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from .indicators import (
    build_trend_summary,
    compute_momentum,
    compute_price_context,
    compute_trend,
    compute_vwap,
    compute_volatility,
    compute_volume,
)

logger = logging.getLogger(__name__)

# Minimum bars needed for longest indicator (SMA50 + buffer)
MIN_BARS = 55


class FeatureEngine:
    """
    Computes technical indicators from raw OHLCV bar data and produces
    enriched snapshots for the LLM evaluator.

    Indicators computed:
    - Trend: SMA(20), SMA(50), EMA(20), EMA(50), price position
    - Momentum: RSI(14), MACD(12,26,9), ROC(10)
    - Volatility: Bollinger Bands(20,2), ATR(14), 20-day hist vol
    - Volume: Volume SMA(20), ratio, OBV, trend
    - Price context: distance from 20d high/low, daily change %
    - trend_summary: human-readable prose summary
    """

    def compute_features(self, snapshot: Dict) -> Dict:
        """
        Enrich a single symbol's snapshot with computed indicators.

        Args:
            snapshot: Dict with keys like 'symbol', 'current_price', and a bars
                      key (bars_1d, bars_5min, etc.) containing OHLCV bar dicts.

        Returns:
            The snapshot dict enriched with 'indicators' and 'trend_summary' keys,
            and bars trimmed to 5 most recent.
        """
        symbol = snapshot.get("symbol", "UNKNOWN")
        timeframe = snapshot.get("timeframe", "1d")

        # Find bars key: bars_5min, bars_1d, etc.
        bars_key = f"bars_{timeframe}"
        bars = snapshot.get(bars_key, snapshot.get("bars_1d", []))

        # Intraday needs fewer bars for indicators
        min_bars = 20 if timeframe != "1d" else MIN_BARS

        if len(bars) < min_bars:
            logger.debug(f"{symbol}: only {len(bars)} bars, need {min_bars} — skipping indicators")
            return snapshot

        try:
            df = self._bars_to_df(bars)
            if df is None or len(df) < min_bars:
                return snapshot

            trend = compute_trend(df)
            momentum = compute_momentum(df)
            volatility = compute_volatility(df)
            volume = compute_volume(df)
            price_ctx = compute_price_context(df)

            indicators = {
                "trend": trend,
                "momentum": momentum,
                "volatility": volatility,
                "volume": volume,
                "price_context": price_ctx,
            }

            # Add VWAP for intraday timeframes
            if timeframe != "1d":
                indicators["vwap"] = compute_vwap(df)

            current_price = snapshot.get("current_price", df["close"].iloc[-1])

            summary = build_trend_summary(
                symbol=symbol,
                price=current_price,
                trend=trend,
                momentum=momentum,
                volatility=volatility,
                volume=volume,
                price_ctx=price_ctx,
            )

            # Price encoding (last 5 bars as token string)
            try:
                from ..ml.encoding import PriceEncoder
                pe = PriceEncoder(period=20)
                indicators["encoding"] = pe.get_last_n_encoded(df, n=5)
            except Exception:
                pass

            snapshot["indicators"] = indicators
            snapshot["trend_summary"] = summary

            # Trim bars to 5 most recent (LLM doesn't need full history)
            snapshot[bars_key] = bars[-5:]

        except Exception as e:
            logger.error(f"FeatureEngine error for {symbol}: {e}", exc_info=True)

        return snapshot

    def compute_features_batch(self, snapshots: Dict[str, Dict]) -> Dict[str, Dict]:
        """Enrich all symbol snapshots in a batch."""
        for symbol in snapshots:
            snapshots[symbol] = self.compute_features(snapshots[symbol])
        return snapshots

    def compute_features_df(self, df: pd.DataFrame) -> Dict:
        """
        Compute all indicators from a raw DataFrame.

        Returns a flat dict of all indicator values. Useful for ML pipelines.
        """
        if df is None or len(df) < MIN_BARS:
            return {}

        result = {}
        result.update(_flatten("trend", compute_trend(df)))
        result.update(_flatten("momentum", compute_momentum(df)))
        result.update(_flatten("volatility", compute_volatility(df)))
        result.update(_flatten("volume", compute_volume(df)))
        result.update(_flatten("price", compute_price_context(df)))
        return result

    @staticmethod
    def _bars_to_df(bars: List[Dict]) -> pd.DataFrame:
        """Convert list of bar dicts to a pandas DataFrame."""
        try:
            df = pd.DataFrame(bars)
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
            return df
        except Exception as e:
            logger.error(f"Failed to convert bars to DataFrame: {e}")
            return None


def _flatten(prefix: str, d: Dict) -> Dict:
    """Flatten a nested dict with a prefix."""
    return {f"{prefix}_{k}": v for k, v in d.items()}
