"""Feature calculator using AlphaPy multi-source indicator system.

Supports indicators from multiple sources with prefixes:
    ap:  - AlphaPy transforms.py (76 functions, default)
    ta:  - TA-Lib (200+ indicators)
    pta: - pandas-ta (130+ indicators)
    vbt: - VectorBT (backtesting indicators)

Examples:
    # Using string specs (recommended)
    calc = FeatureCalculator(indicators=[
        "rsi_14",              # AlphaPy RSI (default source)
        "ma_close_20",         # AlphaPy moving average
        "ap:ema_close_10",     # Explicit AlphaPy source
        "ta:MACD_12_26_9",     # TA-Lib MACD
        "pta:supertrend_10_3", # pandas-ta supertrend
    ])
"""

import logging
from typing import Optional

import polars as pl

from alphapy.indicators import IndicatorEngine

logger = logging.getLogger(__name__)


class FeatureCalculator:
    """Calculate features using the multi-source indicator system.

    Supports indicators from AlphaPy, TA-Lib, pandas-ta, and VectorBT.
    Uses Polars DataFrames throughout for performance.

    All indicators are specified as strings matching transforms.py function signatures:
        - "rsi_14"           -> rsi(df, 14)
        - "ma_close_20"      -> ma(df, 'close', 20)
        - "ema_high_10"      -> ema(df, 'high', 10)
        - "ap:bbands_close_20_2" -> bbands(df, 'close', 20, 2)
    """

    # Default intraday indicators for real-time trading
    # Using string format matching transforms.py function signatures
    # Note: atr_14 alias auto-resolves truerange dependency
    DEFAULT_INDICATORS = [
        # Momentum
        "rsi_2",
        "rsi_5",
        "rsi_14",
        # Moving averages
        "ma_close_5",
        "ma_close_10",
        "ma_close_20",
        "ma_close_50",
        "ema_close_5",
        "ema_close_10",
        "ema_close_20",
        # Volatility (atr auto-resolves truerange)
        "atr_14",
        # Price patterns
        "gap",
        "hlrange",
    ]

    def __init__(
        self,
        indicators: Optional[list[str]] = None,
    ):
        """Initialize feature calculator.

        Args:
            indicators: List of indicator strings. Uses defaults if None.
        """
        self.indicators = indicators or self.DEFAULT_INDICATORS
        self._engine = IndicatorEngine()

    def compute_single(
        self,
        df: pl.DataFrame,
        symbol: Optional[str] = None,
    ) -> pl.DataFrame:
        """Compute indicators for a single symbol's DataFrame.

        Args:
            df: Polars DataFrame with OHLCV columns
            symbol: Symbol name (for logging)

        Returns:
            DataFrame with computed indicators added.
        """
        if df.is_empty():
            logger.warning(f"Empty dataframe for {symbol or 'unknown'}")
            return df

        return self._engine.compute(df, self.indicators)

    def compute_parallel(
        self,
        frames: dict[str, pl.DataFrame],
    ) -> dict[str, pl.DataFrame]:
        """Compute indicators for multiple symbols.

        Args:
            frames: Dictionary mapping symbol to Polars DataFrame

        Returns:
            Dictionary mapping symbol to DataFrame with indicators.
        """
        logger.info(f"Computing indicators for {len(frames)} symbols")

        results = {}
        for symbol, df in frames.items():
            try:
                results[symbol] = self.compute_single(df, symbol)
            except Exception as e:
                logger.error(f"Error computing indicators for {symbol}: {e}")
                results[symbol] = df

        return results

    def get_latest_features(
        self,
        df: pl.DataFrame,
        n_rows: int = 1,
    ) -> pl.DataFrame:
        """Get the latest N rows of features.

        Args:
            df: DataFrame with computed features
            n_rows: Number of rows to return

        Returns:
            DataFrame with only feature columns for the last N rows.
        """
        ohlcv_cols = {"datetime", "open", "high", "low", "close", "volume", "vwap"}
        feature_cols = [c for c in df.columns if c not in ohlcv_cols]

        return df.select(feature_cols).tail(n_rows)

    def validate_features(self, df: pl.DataFrame) -> dict:
        """Validate that indicators were computed successfully.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results.
        """
        # For string-based indicators, the column name is typically the indicator string
        # after stripping the source prefix (ap:, ta:, etc.)
        expected = set()
        for ind in self.indicators:
            if isinstance(ind, str):
                # Strip source prefix if present
                col_name = ind.split(":")[-1] if ":" in ind else ind
                expected.add(col_name)
            else:
                # Should not happen with string-only API, but handle gracefully
                expected.add(str(ind))

        present = [c for c in expected if c in df.columns]
        missing = [c for c in expected if c not in df.columns]

        # Check for all-null columns
        null_cols = []
        for col in present:
            if df[col].null_count() == len(df):
                null_cols.append(col)

        return {
            "valid": len(missing) == 0 and len(null_cols) == 0,
            "present": present,
            "missing": missing,
            "all_null": null_cols,
            "total_expected": len(expected),
        }
