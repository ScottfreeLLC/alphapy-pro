"""Unified indicator calculation engine with multi-source support.

Supports indicators from:
- ap: AlphaPy transforms.py (76 functions) with talipp streaming when available
- ta: TA-Lib (200+ indicators)
- pta: pandas-ta (130+ indicators)
- vbt: VectorBT

Uses source prefixes for explicit source selection:
    engine.compute(df, ["ap:rsi_14", "ta:MACD_12_26_9", "vbt:atr_14"])

Unprefixed indicators default to AlphaPy (ap:):
    engine.compute(df, ["rsi_14", "ma_close_20"])  # → ap:rsi_14, ap:ma_close_20
"""

import logging
from typing import Optional

import polars as pl

try:
    from talipp.indicators import (
        SMA, EMA, RSI, MACD, BB, ATR, ADX, OBV, VWAP,
        Stoch, CCI, Williams, ROC,
        DEMA, TEMA, KAMA, ZLEMA,
        Aroon, ParabolicSAR,
    )
    from talipp.ohlcv import OHLCV
    TALIPP_AVAILABLE = True
except ImportError:
    TALIPP_AVAILABLE = False

logger = logging.getLogger(__name__)


# Indicator configuration: maps name -> (talipp_class, input_type, output_type)
# input_type: "close" = just close prices, "ohlcv" = full OHLCV bars
# output_type: "single" = one value, "multi" = multiple named values
TALIPP_INDICATORS = {
    # Trend indicators (input: close)
    "sma": {"class": SMA, "input": "close", "output": "single"},
    "ema": {"class": EMA, "input": "close", "output": "single"},
    "dema": {"class": DEMA, "input": "close", "output": "single"},
    "tema": {"class": TEMA, "input": "close", "output": "single"},
    "kama": {"class": KAMA, "input": "close", "output": "single"},
    "zlema": {"class": ZLEMA, "input": "close", "output": "single"},

    # Momentum indicators
    "rsi": {"class": RSI, "input": "close", "output": "single"},
    "roc": {"class": ROC, "input": "close", "output": "single"},
    "macd": {
        "class": MACD,
        "input": "close",
        "output": "multi",
        "outputs": ["macd", "signal", "histogram"],
        "attrs": ["macd", "signal", "histogram"],
    },
    "stoch": {
        "class": Stoch,
        "input": "ohlcv",
        "output": "multi",
        "outputs": ["k", "d"],
        "attrs": ["k", "d"],
    },
    "cci": {"class": CCI, "input": "ohlcv", "output": "single"},
    "williams": {"class": Williams, "input": "ohlcv", "output": "single"},
    "adx": {
        "class": ADX,
        "input": "ohlcv",
        "output": "multi",
        "outputs": ["adx", "plus_di", "minus_di"],
        "attrs": ["adx", "plus_di", "minus_di"],
    },
    "aroon": {
        "class": Aroon,
        "input": "ohlcv",
        "output": "multi",
        "outputs": ["up", "down"],
        "attrs": ["up", "down"],
    },

    # Volatility indicators
    "bollinger": {
        "class": BB,
        "input": "close",
        "output": "multi",
        "outputs": ["upper", "middle", "lower"],
        "attrs": ["ub", "cb", "lb"],
    },
    "atr": {"class": ATR, "input": "ohlcv", "output": "single"},
    "parabolic_sar": {"class": ParabolicSAR, "input": "ohlcv", "output": "single"},

    # Volume indicators
    "obv": {"class": OBV, "input": "ohlcv", "output": "single"},
    "vwap": {"class": VWAP, "input": "ohlcv", "output": "single"},
}

# Aliases for indicator names
INDICATOR_ALIASES = {
    "bb": "bollinger",
    "psar": "parabolic_sar",
}


def get_indicator_config(name: str) -> Optional[dict]:
    """Get indicator configuration by name or alias."""
    name = name.lower()
    name = INDICATOR_ALIASES.get(name, name)
    return TALIPP_INDICATORS.get(name)


def list_available_indicators() -> list[str]:
    """List all available indicator names."""
    return sorted(TALIPP_INDICATORS.keys())


class IndicatorEngine:
    """Unified indicator calculation engine with multi-source support.

    Routes indicator requests to appropriate sources:
    - ap: AlphaPy transforms.py (with talipp streaming when available)
    - ta: TA-Lib
    - pta: pandas-ta
    - vbt: VectorBT

    Uses talipp for O(1) incremental updates when available.
    All input/output is Polars DataFrames.

    Example:
        engine = IndicatorEngine()

        # Multi-source computation with prefixes
        df = engine.compute(df, [
            "ap:rsi_14",           # AlphaPy
            "ta:MACD_12_26_9",     # TA-Lib
            "pta:supertrend_10_3", # pandas-ta
            "vbt:rsi_14",          # VectorBT
            "ma_close_20",         # No prefix → defaults to ap:
        ])

        # AlphaPy indicators use transforms.py functions with positional args
        df = engine.compute(df, [
            "rsi_14",              # rsi(df, 14)
            "ma_close_20",         # ma(df, 'close', 20)
            "ema_high_10",         # ema(df, 'high', 10)
        ])
    """

    def __init__(self):
        """Initialize the indicator engine."""
        self._streaming_indicators: dict[str, dict] = {}

    def compute(
        self,
        df: pl.DataFrame,
        indicators: list[str],
    ) -> pl.DataFrame:
        """Compute indicators on a DataFrame with multi-source support.

        Args:
            df: Polars DataFrame with OHLCV columns
            indicators: List of indicator strings with optional source prefix:
                       - "rsi_14", "ma_close_20" (default to alphapy)
                       - "ap:bbands_high_20_2.5" (explicit alphapy)
                       - "ta:MACD_12_26_9" (TA-Lib)
                       - "pta:supertrend_10_3" (pandas-ta)
                       - "vbt:rsi_14" (VectorBT)

        Returns:
            DataFrame with indicator columns added
        """
        from .registry import parse_indicator_string

        result = df.clone()

        for ind in indicators:
            try:
                if not isinstance(ind, str):
                    logger.warning(f"Indicator must be a string, got: {type(ind)}")
                    continue

                # Parse source-prefixed string
                source, name, args = parse_indicator_string(ind)
                result = self._compute_from_source(result, source, name, args, ind)
            except Exception as e:
                logger.error(f"Error computing {ind}: {e}")

        return result

    def _compute_from_source(
        self,
        df: pl.DataFrame,
        source: str,
        name: str,
        args: list | dict,
        original_string: str,
    ) -> pl.DataFrame:
        """Route computation to the appropriate source.

        Args:
            df: DataFrame with OHLCV data
            source: Source name (alphapy, talib, pandas_ta, vectorbt)
            name: Indicator name
            args: For alphapy: list of positional args
                  For external: dict of keyword args
            original_string: Original indicator string for error messages

        Returns:
            DataFrame with indicator column added
        """
        if source == "alphapy":
            # alphapy uses positional args (list)
            return self._compute_alphapy(df, name, args, original_string)
        elif source == "talib":
            # External sources use keyword args (dict)
            return self._compute_talib(df, name, args, original_string)
        elif source == "pandas_ta":
            return self._compute_pandas_ta(df, name, args, original_string)
        elif source == "vectorbt":
            return self._compute_vectorbt(df, name, args, original_string)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _compute_alphapy(
        self,
        df: pl.DataFrame,
        name: str,
        args: list,
        original_string: str,
    ) -> pl.DataFrame:
        """Compute AlphaPy indicator.

        Uses transforms.py which has the authoritative function signatures
        and defaults. No hardcoded defaults here - args list is passed through.

        Args:
            df: Polars DataFrame with OHLCV data
            name: Indicator name (e.g., "rsi", "ma", "ema")
            args: Positional arguments list (e.g., [14], ["close", 20])
            original_string: Original indicator string for column naming
        """
        # Always use transforms.py - it has the real defaults
        return self._compute_transforms(df, name, args, original_string)

    def _resolve_dependencies(
        self,
        df: pl.DataFrame,
        args: list,
    ) -> pl.DataFrame:
        """Auto-resolve indicator dependencies by computing missing column indicators.

        When an indicator like atr_14 expands to ma(df, 'truerange', 14), the 'truerange'
        column must exist. This method detects such dependencies and computes them first.

        Known indicator column patterns that can be auto-computed:
        - truerange: truerange(df)
        - gap: gap(df)
        - hlrange: hlrange(df)
        - etc.

        Args:
            df: DataFrame to check and potentially add dependencies to
            args: Arguments list from the parsed indicator string

        Returns:
            DataFrame with any missing dependency columns added
        """
        try:
            from alphapy import transforms
        except ImportError:
            return df

        # Known single-word column indicators that can be auto-computed
        AUTO_COMPUTABLE = {
            "truerange", "gap", "hlrange", "netreturn", "logreturn",
            "higher", "lower", "highest", "lowest", "diff",
        }

        for arg in args:
            if isinstance(arg, str) and arg not in df.columns:
                # Check if this looks like an indicator that can be computed
                base_name = arg.lower()

                # Check if it's a known auto-computable indicator
                if base_name in AUTO_COMPUTABLE:
                    func = getattr(transforms, base_name, None)
                    if func is not None:
                        logger.debug(f"Auto-computing dependency: {base_name}")
                        try:
                            pdf = df.to_pandas()
                            result = func(pdf)
                            if result is not None:
                                import numpy as np
                                if hasattr(result, "values"):
                                    values = result.values
                                else:
                                    values = np.array(result)
                                df = df.with_columns(pl.Series(base_name, values))
                        except Exception as e:
                            logger.debug(f"Could not auto-compute {base_name}: {e}")

        return df

    def _compute_transforms(
        self,
        df: pl.DataFrame,
        name: str,
        args: list,
        original_string: str,
    ) -> pl.DataFrame:
        """Call transforms.py function with positional args.

        Matches vfunc calling convention from variables.py:
            func(df, *args)

        Handles all return types:
        - pandas.Series (most common)
        - numpy.ndarray (diff, ttmsqueeze*)
        - pandas.DataFrame (dateparts, vwap, runstest)
        - Scalars (row-wise helpers - raises error)

        Also handles length-changing functions by padding with NaN.
        Auto-resolves dependencies when needed.
        """
        import numpy as np
        import pandas as pd

        from .registry import (
            is_row_wise_function,
            is_length_changing_function,
        )

        try:
            from alphapy import transforms
        except ImportError as e:
            raise ValueError(f"Could not import alphapy.transforms: {e}")

        # Auto-resolve any missing column dependencies
        df = self._resolve_dependencies(df, args)

        # Check for row-wise helper functions
        if is_row_wise_function(name):
            raise ValueError(
                f"'{name}' is a row-wise helper function that returns scalars. "
                f"Use it with df.apply(transforms.{name}, axis=1, args=(...)) instead."
            )

        func = getattr(transforms, name, None)
        if func is None:
            raise ValueError(f"Unknown AlphaPy indicator: {name}")

        # Convert to pandas for transforms.py
        pdf = df.to_pandas()
        original_length = len(df)

        # Call the function with positional args (like vfunc does)
        result = func(pdf, *args)

        # Generate column name from original string (strip source prefix)
        col_name = original_string.split(":")[-1] if ":" in original_string else original_string

        # Handle None result
        if result is None:
            logger.warning(f"Function {name} returned None")
            return df

        # Handle scalar result (shouldn't happen if row_wise check passed)
        if isinstance(result, (int, float, bool)) and not isinstance(result, np.ndarray):
            raise ValueError(
                f"'{name}' returned a scalar value. "
                f"This function is designed for row-wise apply, not batch computation."
            )

        # Handle DataFrame result (multi-column)
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                new_col_name = f"{col_name}_{col}" if col != name else col_name
                values = result[col].values
                df = df.with_columns(pl.Series(new_col_name, values))
            return df

        # Get values from Series or array
        if isinstance(result, np.ndarray):
            values = result
        elif hasattr(result, "values"):
            values = result.values
        else:
            values = np.array(result)

        # Handle length-changing functions (pad with NaN at beginning)
        if len(values) < original_length:
            if is_length_changing_function(name):
                padding = np.full(original_length - len(values), np.nan)
                values = np.concatenate([padding, values])
            else:
                logger.warning(
                    f"Function {name} returned {len(values)} elements "
                    f"but expected {original_length}. Padding with NaN."
                )
                padding = np.full(original_length - len(values), np.nan)
                values = np.concatenate([padding, values])

        # Add result to DataFrame
        return df.with_columns(pl.Series(col_name, values))

    def _compute_talib(
        self,
        df: pl.DataFrame,
        name: str,
        params: dict,
        original_string: str,
    ) -> pl.DataFrame:
        """Compute TA-Lib indicator."""
        from .external import get_talib_adapter, ExternalAdapterError

        adapter = get_talib_adapter()
        if not adapter.available:
            raise ExternalAdapterError(
                "TA-Lib is not installed. Install with: pip install TA-Lib "
                "(requires C library)"
            )

        result = adapter.compute(df, name, **params)

        if isinstance(result, pl.DataFrame):
            # Multi-column result (e.g., MACD)
            for col in result.columns:
                df = df.with_columns(result[col].alias(col))
            return df
        else:
            # Single column result
            col_name = f"ta_{name.lower()}_{params.get('timeperiod', '')}"
            return df.with_columns(result.alias(col_name))

    def _compute_pandas_ta(
        self,
        df: pl.DataFrame,
        name: str,
        params: dict,
        original_string: str,
    ) -> pl.DataFrame:
        """Compute pandas-ta indicator."""
        from .external import get_pandas_ta_adapter, ExternalAdapterError

        adapter = get_pandas_ta_adapter()
        if not adapter.available:
            raise ExternalAdapterError(
                "pandas-ta is not installed. Install with: pip install pandas-ta"
            )

        result = adapter.compute(df, name, **params)

        if isinstance(result, pl.DataFrame):
            # Multi-column result
            for col in result.columns:
                df = df.with_columns(result[col].alias(col))
            return df
        else:
            # Single column result
            col_name = f"pta_{name}_{params.get('length', '')}"
            return df.with_columns(result.alias(col_name))

    def _compute_vectorbt(
        self,
        df: pl.DataFrame,
        name: str,
        params: dict,
        original_string: str,
    ) -> pl.DataFrame:
        """Compute VectorBT indicator."""
        from .external import get_vectorbt_adapter, ExternalAdapterError

        adapter = get_vectorbt_adapter()
        if not adapter.available:
            raise ExternalAdapterError(
                "VectorBT is not installed. Install with: pip install vectorbt"
            )

        result = adapter.compute(df, name, **params)

        if isinstance(result, pl.DataFrame):
            # Multi-column result
            for col in result.columns:
                df = df.with_columns(result[col].alias(f"vbt_{col}"))
            return df
        else:
            # Single column result
            col_name = f"vbt_{name}_{params.get('window', '')}"
            return df.with_columns(result.alias(col_name))

    def _pad_values(self, indicator, length: int) -> list:
        """Pad indicator values to match DataFrame length."""
        values = list(indicator)
        padding = length - len(values)
        return [None] * padding + values

    def _pad_list(self, values: list, length: int) -> list:
        """Pad a list to match DataFrame length."""
        padding = length - len(values)
        return [None] * padding + values

# Convenience functions

def add_indicators(
    df: pl.DataFrame,
    indicators: list[str],
) -> pl.DataFrame:
    """Add indicators to a DataFrame.

    Args:
        df: Polars DataFrame with OHLCV data
        indicators: List of indicator strings like "rsi_14", "ma_close_20"

    Returns:
        DataFrame with indicator columns added

    Example:
        df = add_indicators(df, ["rsi_14", "ma_close_20", "ema_high_10"])
        df = add_indicators(df, ["ta:RSI_14", "pta:supertrend_10_3"])
    """
    engine = IndicatorEngine()
    return engine.compute(df, indicators)


