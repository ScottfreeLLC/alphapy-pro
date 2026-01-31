"""External library adapters for technical analysis indicators.

This module provides adapters that allow using external TA libraries
(pandas-ta, TA-Lib) through the unified indicator registry interface.

The adapters handle:
- Polars <-> Pandas conversion for pandas-ta
- Polars <-> NumPy conversion for TA-Lib
- Parameter mapping between library conventions
"""

import logging
from typing import Any, Callable, Optional, Union

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pandas as pd
except ImportError:
    pd = None


logger = logging.getLogger(__name__)


class ExternalAdapterError(Exception):
    """Error in external library adapter."""

    pass


class PandasTAAdapter:
    """Adapter for pandas-ta library.

    pandas-ta provides 130+ technical analysis indicators built on pandas.
    This adapter converts Polars DataFrames to/from pandas for compatibility.

    Usage:
        adapter = PandasTAAdapter()
        result = adapter.compute(df, "rsi", length=14)
    """

    def __init__(self):
        """Initialize the pandas-ta adapter."""
        self._module = None
        self._available = False
        self._load_module()

    def _load_module(self):
        """Attempt to load pandas-ta module."""
        try:
            import pandas_ta as ta

            self._module = ta
            self._available = True
            logger.debug("pandas-ta adapter initialized")
        except ImportError:
            logger.debug("pandas-ta not available")

    @property
    def available(self) -> bool:
        """Check if pandas-ta is available."""
        return self._available

    def list_indicators(self) -> list[str]:
        """List all available pandas-ta indicators.

        Returns:
            List of indicator function names
        """
        if not self._available:
            return []

        # pandas-ta organizes indicators by category
        indicators = []
        for category in ["candles", "cycles", "momentum", "overlap",
                         "performance", "statistics", "trend", "volatility", "volume"]:
            cat_module = getattr(self._module, category, None)
            if cat_module:
                for name in dir(cat_module):
                    if not name.startswith("_") and callable(getattr(cat_module, name, None)):
                        indicators.append(name)

        return sorted(set(indicators))

    def compute(
        self,
        df: "pl.DataFrame",
        indicator: str,
        **params,
    ) -> Union["pl.Series", "pl.DataFrame"]:
        """Compute a pandas-ta indicator on a Polars DataFrame.

        Args:
            df: Polars DataFrame with OHLCV columns
            indicator: Name of the indicator function
            **params: Parameters to pass to the indicator

        Returns:
            Polars Series or DataFrame with computed values

        Raises:
            ExternalAdapterError: If computation fails
        """
        if not self._available:
            raise ExternalAdapterError("pandas-ta is not installed")

        if pl is None or pd is None:
            raise ExternalAdapterError("polars and pandas are required")

        try:
            # Convert to pandas
            pdf = df.to_pandas()

            # Get the indicator function
            func = getattr(self._module, indicator, None)
            if func is None:
                # Try to find in submodules
                for category in ["momentum", "overlap", "trend", "volatility", "volume"]:
                    cat_module = getattr(self._module, category, None)
                    if cat_module:
                        func = getattr(cat_module, indicator, None)
                        if func:
                            break

            if func is None:
                raise ExternalAdapterError(f"Unknown pandas-ta indicator: {indicator}")

            # Map common parameter names
            param_map = {"p": "length", "period": "length", "c": "close"}
            mapped_params = {}
            for k, v in params.items():
                mapped_params[param_map.get(k, k)] = v

            # Call the indicator
            # pandas-ta expects column names in lowercase
            pdf.columns = pdf.columns.str.lower()
            result = func(pdf, **mapped_params)

            if result is None:
                raise ExternalAdapterError(f"pandas-ta {indicator} returned None")

            # Convert back to Polars
            if isinstance(result, pd.DataFrame):
                return pl.from_pandas(result)
            elif isinstance(result, pd.Series):
                return pl.Series(result.name or indicator, result.values)
            else:
                raise ExternalAdapterError(f"Unexpected result type: {type(result)}")

        except Exception as e:
            if isinstance(e, ExternalAdapterError):
                raise
            raise ExternalAdapterError(f"pandas-ta error: {e}") from e


class TALibAdapter:
    """Adapter for TA-Lib library.

    TA-Lib is a C library with Python bindings providing 200+ indicators.
    This adapter converts Polars data to NumPy arrays for compatibility.

    Note: TA-Lib requires separate installation of the C library.

    Usage:
        adapter = TALibAdapter()
        result = adapter.compute(df, "RSI", timeperiod=14)
    """

    def __init__(self):
        """Initialize the TA-Lib adapter."""
        self._module = None
        self._available = False
        self._load_module()

    def _load_module(self):
        """Attempt to load TA-Lib module."""
        try:
            import talib

            self._module = talib
            self._available = True
            logger.debug("TA-Lib adapter initialized")
        except ImportError:
            logger.debug("TA-Lib not available (requires C library installation)")

    @property
    def available(self) -> bool:
        """Check if TA-Lib is available."""
        return self._available

    def list_indicators(self) -> list[str]:
        """List all available TA-Lib indicators.

        Returns:
            List of indicator function names
        """
        if not self._available:
            return []

        return sorted(self._module.get_functions())

    def list_groups(self) -> dict[str, list[str]]:
        """List indicators grouped by category.

        Returns:
            Dictionary mapping group name to list of indicators
        """
        if not self._available:
            return {}

        groups = {}
        for group in self._module.get_function_groups():
            groups[group] = self._module.get_function_groups()[group]
        return groups

    def compute(
        self,
        df: "pl.DataFrame",
        indicator: str,
        **params,
    ) -> Union["pl.Series", "pl.DataFrame"]:
        """Compute a TA-Lib indicator on a Polars DataFrame.

        Args:
            df: Polars DataFrame with OHLCV columns
            indicator: Name of the indicator function (e.g., "RSI", "MACD")
            **params: Parameters to pass to the indicator

        Returns:
            Polars Series or DataFrame with computed values

        Raises:
            ExternalAdapterError: If computation fails
        """
        if not self._available:
            raise ExternalAdapterError(
                "TA-Lib is not installed. Install with: pip install TA-Lib "
                "(requires C library: brew install ta-lib on macOS)"
            )

        if pl is None:
            raise ExternalAdapterError("polars is required")

        try:
            # Get the indicator function
            func = getattr(self._module, indicator.upper(), None)
            if func is None:
                raise ExternalAdapterError(f"Unknown TA-Lib indicator: {indicator}")

            # Extract numpy arrays from DataFrame
            arrays = {}
            col_map = {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
            for polars_col, talib_name in col_map.items():
                if polars_col in df.columns:
                    arrays[talib_name] = df[polars_col].to_numpy().astype(np.float64)

            # Map common parameter names
            param_map = {"p": "timeperiod", "period": "timeperiod", "length": "timeperiod"}
            mapped_params = {}
            for k, v in params.items():
                mapped_params[param_map.get(k, k)] = v

            # Determine what input the function needs
            # Most TA-Lib functions use one of: close, high/low/close, open/high/low/close
            import inspect
            sig = inspect.signature(func)
            func_params = list(sig.parameters.keys())

            # Build input arguments
            input_args = []
            for p in func_params:
                p_lower = p.lower()
                if p_lower in arrays:
                    input_args.append(arrays[p_lower])
                elif p_lower == "real":
                    input_args.append(arrays.get("close", arrays.get("open")))
                elif p_lower == "real0":
                    input_args.append(arrays.get("high", arrays.get("close")))
                elif p_lower == "real1":
                    input_args.append(arrays.get("low", arrays.get("close")))
                else:
                    break  # Non-data parameter

            # Call the function
            result = func(*input_args, **mapped_params)

            # Handle multiple outputs (e.g., MACD returns macd, signal, hist)
            if isinstance(result, tuple):
                # Get output names
                output_names = self._module.abstract.Function(indicator.upper()).output_names
                columns = {}
                for i, (name, values) in enumerate(zip(output_names, result)):
                    columns[f"{indicator.lower()}_{name}"] = values
                return pl.DataFrame(columns)
            else:
                return pl.Series(indicator.lower(), result)

        except Exception as e:
            if isinstance(e, ExternalAdapterError):
                raise
            raise ExternalAdapterError(f"TA-Lib error: {e}") from e


class CustomModuleAdapter:
    """Adapter for user-defined custom indicator modules.

    This adapter allows loading indicators from arbitrary Python modules,
    enabling users to add their own indicators via configuration.

    Usage:
        adapter = CustomModuleAdapter("mymodule.indicators")
        result = adapter.compute(df, "my_custom_indicator", param1=10)
    """

    def __init__(self, module_path: str):
        """Initialize with a module path.

        Args:
            module_path: Dotted module path (e.g., "mypackage.indicators")
        """
        self._module_path = module_path
        self._module = None
        self._available = False
        self._load_module()

    def _load_module(self):
        """Attempt to load the custom module."""
        try:
            import importlib

            self._module = importlib.import_module(self._module_path)
            self._available = True
            logger.debug(f"Custom module loaded: {self._module_path}")
        except ImportError as e:
            logger.debug(f"Custom module not available: {self._module_path} - {e}")

    @property
    def available(self) -> bool:
        """Check if the custom module is available."""
        return self._available

    def list_indicators(self) -> list[str]:
        """List callable functions in the module.

        Returns:
            List of function names
        """
        if not self._available:
            return []

        import inspect

        return [
            name
            for name, obj in inspect.getmembers(self._module, inspect.isfunction)
            if not name.startswith("_")
        ]

    def compute(
        self,
        df: "pl.DataFrame",
        indicator: str,
        **params,
    ) -> Union["pl.Series", "pl.DataFrame"]:
        """Compute an indicator from the custom module.

        Args:
            df: Polars DataFrame with OHLCV columns
            indicator: Name of the function
            **params: Parameters to pass

        Returns:
            Polars Series or DataFrame

        Raises:
            ExternalAdapterError: If computation fails
        """
        if not self._available:
            raise ExternalAdapterError(f"Module not available: {self._module_path}")

        func = getattr(self._module, indicator, None)
        if func is None:
            raise ExternalAdapterError(
                f"Function {indicator} not found in {self._module_path}"
            )

        try:
            result = func(df, **params)

            # Ensure result is Polars
            if pd is not None and isinstance(result, (pd.DataFrame, pd.Series)):
                if isinstance(result, pd.DataFrame):
                    return pl.from_pandas(result)
                return pl.Series(result.name or indicator, result.values)
            elif isinstance(result, np.ndarray):
                return pl.Series(indicator, result)

            return result
        except Exception as e:
            raise ExternalAdapterError(f"Custom indicator error: {e}") from e


class VectorBTAdapter:
    """Adapter for VectorBT library.

    VectorBT provides fast vectorized indicators optimized for backtesting.
    It includes technical analysis indicators with a focus on portfolio
    simulation and strategy testing.

    Note: VectorBT requires pandas and numpy.

    Usage:
        adapter = VectorBTAdapter()
        result = adapter.compute(df, "RSI", window=14)
    """

    def __init__(self):
        """Initialize the VectorBT adapter."""
        self._module = None
        self._available = False
        self._load_module()

    def _load_module(self):
        """Attempt to load VectorBT module."""
        try:
            import vectorbt as vbt

            self._module = vbt
            self._available = True
            logger.debug("VectorBT adapter initialized")
        except ImportError:
            logger.debug("VectorBT not available")

    @property
    def available(self) -> bool:
        """Check if VectorBT is available."""
        return self._available

    def list_indicators(self) -> list[str]:
        """List all available VectorBT indicators.

        Returns:
            List of indicator names
        """
        if not self._available:
            return []

        # VectorBT indicators are in various namespaces
        indicators = []
        try:
            # Main indicator classes
            if hasattr(self._module, "indicators"):
                for name in dir(self._module.indicators):
                    if not name.startswith("_"):
                        obj = getattr(self._module.indicators, name, None)
                        if obj is not None and hasattr(obj, "run"):
                            indicators.append(name)

            # Also check for common indicators at top level
            common_indicators = ["RSI", "MACD", "BBANDS", "ATR", "SMA", "EMA", "STOCH"]
            for name in common_indicators:
                if hasattr(self._module, name) or name.upper() in [i.upper() for i in indicators]:
                    if name not in indicators:
                        indicators.append(name)

        except Exception as e:
            logger.debug(f"Error listing VectorBT indicators: {e}")

        return sorted(indicators)

    def compute(
        self,
        df: "pl.DataFrame",
        indicator: str,
        **params,
    ) -> Union["pl.Series", "pl.DataFrame"]:
        """Compute a VectorBT indicator on a Polars DataFrame.

        Args:
            df: Polars DataFrame with OHLCV columns
            indicator: Name of the indicator (e.g., "RSI", "MACD")
            **params: Parameters to pass to the indicator

        Returns:
            Polars Series or DataFrame with computed values

        Raises:
            ExternalAdapterError: If computation fails
        """
        if not self._available:
            raise ExternalAdapterError(
                "VectorBT is not installed. Install with: pip install vectorbt"
            )

        if pl is None or pd is None:
            raise ExternalAdapterError("polars and pandas are required")

        try:
            # Convert to pandas
            pdf = df.to_pandas()
            pdf.columns = pdf.columns.str.lower()

            # Map common parameter names
            param_map = {"p": "window", "period": "window", "timeperiod": "window"}
            mapped_params = {}
            for k, v in params.items():
                mapped_params[param_map.get(k, k)] = v

            # Get the indicator class from vectorbt
            ind_class = None

            # Try different locations for the indicator
            indicator_upper = indicator.upper()
            indicator_lower = indicator.lower()

            # Check in vbt.indicators namespace
            if hasattr(self._module, "indicators"):
                ind_class = getattr(self._module.indicators, indicator_upper, None)
                if ind_class is None:
                    ind_class = getattr(self._module.indicators, indicator_lower, None)

            # Check at top level
            if ind_class is None:
                ind_class = getattr(self._module, indicator_upper, None)

            if ind_class is None:
                raise ExternalAdapterError(f"Unknown VectorBT indicator: {indicator}")

            # Determine what input the indicator needs
            close = pdf["close"].values if "close" in pdf.columns else None
            high = pdf["high"].values if "high" in pdf.columns else None
            low = pdf["low"].values if "low" in pdf.columns else None
            open_arr = pdf["open"].values if "open" in pdf.columns else None
            volume = pdf["volume"].values if "volume" in pdf.columns else None

            # Call the indicator based on its type
            if hasattr(ind_class, "run"):
                # Most VectorBT indicators use .run() method
                if indicator_upper in ("RSI", "SMA", "EMA", "ROC"):
                    result = ind_class.run(close, **mapped_params)
                elif indicator_upper in ("MACD",):
                    result = ind_class.run(close, **mapped_params)
                elif indicator_upper in ("ATR", "BBANDS", "STOCH"):
                    result = ind_class.run(high, low, close, **mapped_params)
                else:
                    # Try with close as default
                    result = ind_class.run(close, **mapped_params)
            else:
                raise ExternalAdapterError(
                    f"VectorBT indicator {indicator} does not have a run() method"
                )

            # Convert result to Polars
            if hasattr(result, "to_frame"):
                result_df = result.to_frame()
                return pl.from_pandas(result_df)
            elif hasattr(result, "values"):
                return pl.Series(indicator.lower(), result.values)
            else:
                return pl.Series(indicator.lower(), np.array(result))

        except Exception as e:
            if isinstance(e, ExternalAdapterError):
                raise
            raise ExternalAdapterError(f"VectorBT error: {e}") from e


# Module-level adapter instances (lazy initialization)
_pandas_ta_adapter: Optional[PandasTAAdapter] = None
_talib_adapter: Optional[TALibAdapter] = None
_vectorbt_adapter: Optional[VectorBTAdapter] = None
_custom_adapters: dict[str, CustomModuleAdapter] = {}


def get_pandas_ta_adapter() -> PandasTAAdapter:
    """Get the pandas-ta adapter singleton.

    Returns:
        PandasTAAdapter instance
    """
    global _pandas_ta_adapter
    if _pandas_ta_adapter is None:
        _pandas_ta_adapter = PandasTAAdapter()
    return _pandas_ta_adapter


def get_talib_adapter() -> TALibAdapter:
    """Get the TA-Lib adapter singleton.

    Returns:
        TALibAdapter instance
    """
    global _talib_adapter
    if _talib_adapter is None:
        _talib_adapter = TALibAdapter()
    return _talib_adapter


def get_vectorbt_adapter() -> VectorBTAdapter:
    """Get the VectorBT adapter singleton.

    Returns:
        VectorBTAdapter instance
    """
    global _vectorbt_adapter
    if _vectorbt_adapter is None:
        _vectorbt_adapter = VectorBTAdapter()
    return _vectorbt_adapter


def get_custom_adapter(module_path: str) -> CustomModuleAdapter:
    """Get or create a custom module adapter.

    Args:
        module_path: Dotted module path

    Returns:
        CustomModuleAdapter instance
    """
    if module_path not in _custom_adapters:
        _custom_adapters[module_path] = CustomModuleAdapter(module_path)
    return _custom_adapters[module_path]


def compute_external(
    df: "pl.DataFrame",
    source: str,
    indicator: str,
    **params,
) -> Union["pl.Series", "pl.DataFrame"]:
    """Compute an indicator from an external source.

    This is a convenience function that routes to the appropriate adapter.

    Args:
        df: Polars DataFrame
        source: Source library ("pandas_ta", "talib", "vectorbt", or custom module path)
        indicator: Indicator function name
        **params: Indicator parameters

    Returns:
        Computed indicator values

    Raises:
        ExternalAdapterError: If computation fails
    """
    if source == "pandas_ta":
        return get_pandas_ta_adapter().compute(df, indicator, **params)
    elif source == "talib":
        return get_talib_adapter().compute(df, indicator, **params)
    elif source == "vectorbt":
        return get_vectorbt_adapter().compute(df, indicator, **params)
    else:
        return get_custom_adapter(source).compute(df, indicator, **params)
