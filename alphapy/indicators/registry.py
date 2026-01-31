"""Central registry for all technical indicators.

This module provides a unified registry that discovers and manages technical
indicators from multiple sources:
- alphapy.transforms (built-in functions)
- talipp (streaming indicators)
- External libraries (pandas-ta, TA-Lib)
- User-defined custom indicators

The registry provides a single API for indicator lookup and computation
regardless of the underlying implementation.
"""

import importlib
import inspect
import logging
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional, Union

import yaml

from .spec import (
    IndicatorCategory,
    IndicatorParam,
    IndicatorSource,
    IndicatorSpec,
    COLUMN_PARAM,
    PERIOD_PARAM,
    STD_DEV_PARAM,
)


logger = logging.getLogger(__name__)


# Mapping of indicator names to categories
INDICATOR_CATEGORIES = {
    # Trend indicators
    "adx": IndicatorCategory.TREND,
    "aroon": IndicatorCategory.TREND,
    "diplus": IndicatorCategory.TREND,
    "diminus": IndicatorCategory.TREND,
    "psar": IndicatorCategory.TREND,
    # Momentum indicators
    "rsi": IndicatorCategory.MOMENTUM,
    "macd": IndicatorCategory.MOMENTUM,
    "stoch": IndicatorCategory.MOMENTUM,
    "cci": IndicatorCategory.MOMENTUM,
    "williams": IndicatorCategory.MOMENTUM,
    "roc": IndicatorCategory.MOMENTUM,
    "mfi": IndicatorCategory.MOMENTUM,
    "momentum": IndicatorCategory.MOMENTUM,
    # Volatility indicators
    "atr": IndicatorCategory.VOLATILITY,
    "bbands": IndicatorCategory.VOLATILITY,
    "bblower": IndicatorCategory.VOLATILITY,
    "bbupper": IndicatorCategory.VOLATILITY,
    "natr": IndicatorCategory.VOLATILITY,
    "truerange": IndicatorCategory.VOLATILITY,
    "volatility": IndicatorCategory.VOLATILITY,
    # Volume indicators
    "obv": IndicatorCategory.VOLUME,
    "vwap": IndicatorCategory.VOLUME,
    "pvt": IndicatorCategory.VOLUME,
    "ad": IndicatorCategory.VOLUME,
    # Overlap/Moving averages
    "ma": IndicatorCategory.OVERLAP,
    "ema": IndicatorCategory.OVERLAP,
    "sma": IndicatorCategory.OVERLAP,
    "dema": IndicatorCategory.OVERLAP,
    "tema": IndicatorCategory.OVERLAP,
    "kama": IndicatorCategory.OVERLAP,
    "zlema": IndicatorCategory.OVERLAP,
    "wma": IndicatorCategory.OVERLAP,
    # Statistical
    "zscore": IndicatorCategory.STATISTICAL,
    "percentile": IndicatorCategory.STATISTICAL,
    "rank": IndicatorCategory.STATISTICAL,
    "normalize": IndicatorCategory.STATISTICAL,
    # Calendar
    "bizday": IndicatorCategory.CALENDAR,
    "rdate": IndicatorCategory.CALENDAR,
}

# Indicators that require OHLCV data (not just close)
OHLCV_INDICATORS = {
    "adx", "atr", "aroon", "cci", "diplus", "diminus", "mfi",
    "obv", "psar", "stoch", "truerange", "vwap", "williams",
}

# talipp streaming class mapping
TALIPP_STREAMING = {
    "sma": "SMA",
    "ema": "EMA",
    "rsi": "RSI",
    "macd": "MACD",
    "bb": "BB",
    "atr": "ATR",
    "adx": "ADX",
    "stoch": "Stoch",
    "cci": "CCI",
    "obv": "OBV",
    "vwap": "VWAP",
    "dema": "DEMA",
    "tema": "TEMA",
    "kama": "KAMA",
    "zlema": "ZLEMA",
    "roc": "ROC",
    "williams": "Williams",
    "aroon": "Aroon",
    "psar": "ParabolicSAR",
}


class IndicatorRegistry:
    """Central registry for all technical indicators.

    This class provides:
    - Auto-discovery of transforms.py functions
    - Loading of external library adapters
    - Configuration-based indicator definitions
    - Unified lookup by name or alias
    - Parameter parsing from string notation (e.g., "rsi_14")

    Usage:
        # Initialize and discover
        registry = IndicatorRegistry()
        registry.discover_transforms()

        # Get indicator by name
        spec = registry.get("rsi")
        result = spec.compute(df, p=14)

        # Parse shorthand notation
        spec, params = registry.parse("ma_close_20")
        result = spec.compute(df, **params)
    """

    _instance: Optional["IndicatorRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "IndicatorRegistry":
        """Singleton pattern for registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry (only runs once due to singleton)."""
        if self._initialized:
            return

        self._indicators: dict[str, IndicatorSpec] = {}
        self._aliases: dict[str, str] = {}
        self._external_modules: dict[str, ModuleType] = {}
        self._config: dict[str, Any] = {}

        self._initialized = True

    @classmethod
    def reset(cls):
        """Reset the singleton for testing purposes."""
        if cls._instance is not None:
            cls._instance._indicators.clear()
            cls._instance._aliases.clear()
            cls._instance._external_modules.clear()
            cls._instance._config.clear()
            cls._initialized = False

    def register(self, spec: IndicatorSpec) -> None:
        """Register an indicator specification.

        Args:
            spec: IndicatorSpec to register
        """
        self._indicators[spec.name.lower()] = spec

        # Register aliases
        for alias in spec.aliases:
            self._aliases[alias.lower()] = spec.name.lower()

        logger.debug(f"Registered indicator: {spec.name}")

    def unregister(self, name: str) -> bool:
        """Unregister an indicator by name.

        Args:
            name: Indicator name to unregister

        Returns:
            True if found and removed, False otherwise
        """
        name_lower = name.lower()
        if name_lower in self._indicators:
            spec = self._indicators.pop(name_lower)
            # Remove aliases
            for alias in spec.aliases:
                self._aliases.pop(alias.lower(), None)
            return True
        return False

    def get(self, name: str) -> Optional[IndicatorSpec]:
        """Get indicator specification by name or alias.

        Args:
            name: Indicator name or alias

        Returns:
            IndicatorSpec if found, None otherwise
        """
        name_lower = name.lower()

        # Direct lookup
        if name_lower in self._indicators:
            return self._indicators[name_lower]

        # Alias lookup
        if name_lower in self._aliases:
            return self._indicators[self._aliases[name_lower]]

        return None

    def has(self, name: str) -> bool:
        """Check if an indicator exists by name or alias.

        Args:
            name: Indicator name or alias

        Returns:
            True if indicator exists
        """
        return self.get(name) is not None

    def list_indicators(
        self,
        category: Optional[IndicatorCategory] = None,
        source: Optional[IndicatorSource] = None,
    ) -> list[str]:
        """List all registered indicator names.

        Args:
            category: Filter by category (optional)
            source: Filter by source (optional)

        Returns:
            List of indicator names
        """
        result = []
        for name, spec in self._indicators.items():
            if category and spec.category != category:
                continue
            if source and spec.source != source:
                continue
            result.append(name)
        return sorted(result)

    def list_aliases(self) -> dict[str, str]:
        """Get mapping of all aliases to canonical names.

        Returns:
            Dictionary mapping alias -> canonical name
        """
        return dict(self._aliases)

    def parse(self, indicator_str: str) -> tuple[Optional[IndicatorSpec], dict[str, Any]]:
        """Parse indicator string notation into spec and parameters.

        Supports formats:
        - "rsi_14" -> rsi with period=14
        - "ma_close_20" -> ma with column=close, period=20
        - "bbands_high_10_2.0" -> bbands with column=high, period=10, sd=2.0

        Args:
            indicator_str: Indicator string to parse

        Returns:
            Tuple of (IndicatorSpec, params dict), or (None, {}) if not found
        """
        parts = indicator_str.lower().split("_")
        if not parts:
            return None, {}

        # First part is always the indicator name
        name = parts[0]
        spec = self.get(name)

        if spec is None:
            # Try two-part name (e.g., "true_range" -> "truerange")
            if len(parts) >= 2:
                name = parts[0] + parts[1]
                spec = self.get(name)
                parts = [name] + parts[2:]

            if spec is None:
                return None, {}

        params = {}
        remaining = parts[1:]

        # Parse remaining parts based on indicator parameters
        param_specs = spec.params
        param_idx = 0

        for part in remaining:
            # Try to interpret as a number
            try:
                if "." in part:
                    value = float(part)
                else:
                    value = int(part)

                # Assign to next numeric parameter
                for i, p in enumerate(param_specs[param_idx:], param_idx):
                    if p.param_type in (int, float):
                        params[p.name] = value
                        param_idx = i + 1
                        break
            except ValueError:
                # String value - likely a column name
                for i, p in enumerate(param_specs[param_idx:], param_idx):
                    if p.param_type == str:
                        params[p.name] = part
                        param_idx = i + 1
                        break

        return spec, params

    def discover_transforms(self) -> int:
        """Auto-discover indicator functions from alphapy.transforms.

        Inspects the transforms module and creates IndicatorSpec for each
        function that looks like an indicator (takes df as first param).

        Returns:
            Number of indicators discovered
        """
        try:
            from alphapy import transforms as transforms_module
        except ImportError as e:
            logger.warning(f"Could not import transforms module: {e}")
            return 0

        count = 0
        for name, func in inspect.getmembers(transforms_module, inspect.isfunction):
            # Skip private functions
            if name.startswith("_"):
                continue

            # Check if it takes a DataFrame as first param
            sig = inspect.signature(func)
            params = list(sig.parameters.values())

            if not params or params[0].name not in ("df", "frame", "data"):
                continue

            # Create parameter specs from function signature
            param_specs = self._extract_params(params[1:])

            # Determine category
            category = INDICATOR_CATEGORIES.get(name, IndicatorCategory.OTHER)

            # Check if OHLCV is required
            requires_ohlcv = name in OHLCV_INDICATORS

            # Check for streaming support
            streaming_class = None
            supports_streaming = False
            if name in TALIPP_STREAMING:
                supports_streaming = True
                try:
                    from talipp import indicators as talipp_ind
                    streaming_class = getattr(talipp_ind, TALIPP_STREAMING[name], None)
                except ImportError:
                    pass

            # Extract description from docstring
            description = ""
            if func.__doc__:
                # Get first line of docstring
                description = func.__doc__.strip().split("\n")[0]
                # Remove r-string prefix markers if present
                description = description.lstrip("r").strip('"\'')

            spec = IndicatorSpec(
                name=name,
                category=category,
                source=IndicatorSource.TRANSFORMS,
                description=description,
                params=param_specs,
                requires_ohlcv=requires_ohlcv,
                supports_streaming=supports_streaming,
                compute_func=func,
                streaming_class=streaming_class,
            )

            self.register(spec)
            count += 1

        logger.info(f"Discovered {count} indicators from transforms module")
        return count

    def _extract_params(self, params: list[inspect.Parameter]) -> list[IndicatorParam]:
        """Extract IndicatorParam specs from function parameters.

        Args:
            params: List of inspect.Parameter objects

        Returns:
            List of IndicatorParam specifications
        """
        result = []

        for param in params:
            # Get default value
            if param.default is inspect.Parameter.empty:
                default = None
            else:
                default = param.default

            # Infer type from default value
            if default is None:
                param_type = str
            elif isinstance(default, bool):
                param_type = bool
            elif isinstance(default, int):
                param_type = int
            elif isinstance(default, float):
                param_type = float
            else:
                param_type = str

            # Common parameter descriptions
            descriptions = {
                "p": "Lookback period",
                "c": "Column name",
                "sd": "Standard deviations",
                "n": "Number of periods",
                "fast": "Fast period",
                "slow": "Slow period",
                "signal": "Signal period",
            }

            result.append(IndicatorParam(
                name=param.name,
                param_type=param_type,
                default=default if default is not None else (14 if param.name == "p" else "close"),
                description=descriptions.get(param.name, ""),
            ))

        return result

    def load_from_config(self, config_path: Union[str, Path]) -> int:
        """Load indicator definitions from YAML configuration.

        Args:
            config_path: Path to indicators.yml

        Returns:
            Number of indicators loaded
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return 0

        with open(config_path) as f:
            self._config = yaml.safe_load(f) or {}

        count = 0

        # Load external library configurations
        external_libs = self._config.get("external_libraries", {})
        for lib_name, lib_config in external_libs.items():
            if lib_config.get("enabled", False):
                try:
                    module = importlib.import_module(lib_config.get("module", lib_name))
                    self._external_modules[lib_name] = module
                    logger.info(f"Loaded external library: {lib_name}")
                except ImportError as e:
                    logger.debug(f"External library not available: {lib_name} - {e}")

        # Load custom indicator definitions
        custom_indicators = self._config.get("custom_indicators") or {}
        for ind_name, ind_config in custom_indicators.items():
            try:
                spec = self._create_external_spec(ind_name, ind_config)
                if spec:
                    self.register(spec)
                    count += 1
            except Exception as e:
                logger.error(f"Error loading custom indicator {ind_name}: {e}")

        # Load aliases from config
        aliases = self._config.get("aliases", {})
        for alias, canonical in aliases.items():
            self._aliases[alias.lower()] = canonical.lower()

        logger.info(f"Loaded {count} custom indicators from config")
        return count

    def _create_external_spec(
        self,
        name: str,
        config: dict[str, Any],
    ) -> Optional[IndicatorSpec]:
        """Create an IndicatorSpec from external library config.

        Args:
            name: Indicator name
            config: Configuration dictionary

        Returns:
            IndicatorSpec or None if creation fails
        """
        source_name = config.get("source", "custom")

        # Determine source enum
        source_map = {
            "pandas_ta": IndicatorSource.PANDAS_TA,
            "talib": IndicatorSource.TALIB,
            "custom": IndicatorSource.CUSTOM,
        }
        source = source_map.get(source_name, IndicatorSource.CUSTOM)

        # Build parameter specs from config
        params = []
        for pname, pconfig in config.get("params", {}).items():
            if isinstance(pconfig, dict):
                params.append(IndicatorParam(
                    name=pname,
                    param_type={"int": int, "float": float, "str": str, "bool": bool}.get(
                        pconfig.get("type", "int"), int
                    ),
                    default=pconfig.get("default"),
                    description=pconfig.get("description", ""),
                ))
            else:
                # Simple value = default
                param_type = type(pconfig)
                params.append(IndicatorParam(
                    name=pname,
                    param_type=param_type,
                    default=pconfig,
                ))

        return IndicatorSpec(
            name=name,
            category=IndicatorCategory(config.get("category", "other")),
            source=source,
            description=config.get("description", ""),
            params=params,
            aliases=config.get("aliases", []),
            outputs=config.get("outputs", [name]),
            requires_ohlcv=config.get("requires_ohlcv", False),
            external_module=config.get("module", source_name),
            external_func=config.get("function", name),
        )

    def get_external_module(self, name: str) -> Optional[ModuleType]:
        """Get a loaded external module by name.

        Args:
            name: Module name (e.g., "pandas_ta")

        Returns:
            Module if loaded, None otherwise
        """
        return self._external_modules.get(name)


# Source prefix mapping for unified indicator API
SOURCE_PREFIXES = {
    "ap:": "alphapy",
    "ta:": "talib",
    "pta:": "pandas_ta",
    "vbt:": "vectorbt",
}

DEFAULT_SOURCE = "alphapy"


def parse_source_prefix(indicator: str) -> tuple[str, str]:
    """Parse source prefix from indicator string.

    Args:
        indicator: String like "ta:RSI_14" or "ap:ma_close_20" or "rsi_14"

    Returns:
        Tuple of (source, indicator_without_prefix)

    Examples:
        "ta:RSI_14" → ("talib", "RSI_14")
        "ap:ma_close_20" → ("alphapy", "ma_close_20")
        "rsi_14" → ("alphapy", "rsi_14")  # default to alphapy
    """
    for prefix, source_name in SOURCE_PREFIXES.items():
        if indicator.startswith(prefix):
            return source_name, indicator[len(prefix):]
    return DEFAULT_SOURCE, indicator


def parse_indicator_string(indicator: str) -> tuple[str, str, list | dict]:
    """Parse source-prefixed indicator string into components.

    This is the main entry point for parsing indicator specifications
    that may include source prefixes.

    Args:
        indicator: String like "ta:RSI_14" or "ap:ma_close_20" or "rsi_14"

    Returns:
        Tuple of (source, name, args_or_params)
        - For alphapy: args is a list of positional arguments
        - For external sources: args is a dict of keyword arguments

    Examples:
        "ta:RSI_14" → ("talib", "RSI", {"timeperiod": 14})
        "ap:ma_close_20" → ("alphapy", "ma", ["close", 20])
        "rsi_14" → ("alphapy", "rsi", [14])
        "pta:supertrend_10_3" → ("pandas_ta", "supertrend", {"length": 10, "multiplier": 3.0})
    """
    # First extract source
    source, remaining = parse_source_prefix(indicator)

    # Parse the indicator name and parameters
    parts = remaining.split("_")
    if not parts:
        return source, "", [] if source == "alphapy" else {}

    name = parts[0]
    params = {}

    if source == "talib":
        # TA-Lib uses uppercase names and "timeperiod" param
        name = name.upper()
        remaining_parts = parts[1:]

        # TA-Lib parameter mapping
        if len(remaining_parts) >= 1:
            try:
                params["timeperiod"] = int(remaining_parts[0])
            except ValueError:
                pass
        # MACD: ta:MACD_12_26_9
        if len(remaining_parts) >= 3 and name in ("MACD", "STOCH", "BBANDS"):
            try:
                if name == "MACD":
                    params["fastperiod"] = int(remaining_parts[0])
                    params["slowperiod"] = int(remaining_parts[1])
                    params["signalperiod"] = int(remaining_parts[2])
                elif name == "BBANDS":
                    params["timeperiod"] = int(remaining_parts[0])
                    params["nbdevup"] = float(remaining_parts[1])
                    params["nbdevdn"] = float(remaining_parts[2]) if len(remaining_parts) > 2 else float(remaining_parts[1])
            except (ValueError, IndexError):
                pass
        return source, name, params

    elif source == "pandas_ta":
        # pandas-ta uses lowercase and "length" param
        name = name.lower()
        remaining_parts = parts[1:]

        if len(remaining_parts) >= 1:
            try:
                params["length"] = int(remaining_parts[0])
            except ValueError:
                # Might be a column name
                params["close"] = remaining_parts[0]
                if len(remaining_parts) >= 2:
                    try:
                        params["length"] = int(remaining_parts[1])
                    except ValueError:
                        pass
        # Additional params for specific indicators
        if len(remaining_parts) >= 2 and name == "supertrend":
            try:
                params["length"] = int(remaining_parts[0])
                params["multiplier"] = float(remaining_parts[1])
            except (ValueError, IndexError):
                pass
        return source, name, params

    elif source == "vectorbt":
        # VectorBT uses various conventions
        name = name.lower()
        remaining_parts = parts[1:]

        if len(remaining_parts) >= 1:
            try:
                params["window"] = int(remaining_parts[0])
            except ValueError:
                pass
        return source, name, params

    else:  # alphapy
        # Use positional args parser (like vparse in variables.py)
        name, args = parse_transforms_string(remaining)
        return source, name, args


# Module-level convenience functions

_registry: Optional[IndicatorRegistry] = None


def get_registry() -> IndicatorRegistry:
    """Get the global indicator registry instance.

    Returns:
        IndicatorRegistry singleton
    """
    global _registry
    if _registry is None:
        _registry = IndicatorRegistry()
    return _registry


def discover_all() -> int:
    """Discover all available indicators.

    This discovers transforms.py functions and loads config if available.

    Returns:
        Total number of indicators discovered
    """
    registry = get_registry()
    count = registry.discover_transforms()

    # Try to load config from standard location
    config_paths = [
        Path("config/indicators.yml"),
        Path(__file__).parent.parent.parent / "config" / "indicators.yml",
    ]
    for path in config_paths:
        if path.exists():
            count += registry.load_from_config(path)
            break

    return count


def get_indicator(name: str) -> Optional[IndicatorSpec]:
    """Get an indicator by name or alias.

    Args:
        name: Indicator name or alias

    Returns:
        IndicatorSpec if found
    """
    return get_registry().get(name)


def parse_indicator(indicator_str: str) -> tuple[Optional[IndicatorSpec], dict[str, Any]]:
    """Parse indicator string notation.

    Args:
        indicator_str: String like "rsi_14" or "ma_close_20"

    Returns:
        Tuple of (IndicatorSpec, params)
    """
    return get_registry().parse(indicator_str)


def list_all_indicators() -> dict[str, list[str]]:
    """List all available indicators grouped by source.

    Returns:
        Dictionary mapping source name to list of indicator names:
        {
            "alphapy": ["adx", "bbands", "ema", "ma", "rsi", ...],  # 76
            "talib": ["RSI", "MACD", "ATR", "BBANDS", ...],  # 200+
            "pandas_ta": ["rsi", "supertrend", "vwap", ...],  # 130+
            "vectorbt": ["RSI", "MACD", "ATR", ...],
        }
    """
    from .external import (
        get_pandas_ta_adapter,
        get_talib_adapter,
        get_vectorbt_adapter,
    )

    # Discover AlphaPy transforms
    registry = get_registry()
    if not registry._indicators:
        registry.discover_transforms()

    result = {
        "alphapy": sorted(registry.list_indicators()),
        "talib": [],
        "pandas_ta": [],
        "vectorbt": [],
    }

    # Get TA-Lib indicators
    try:
        talib_adapter = get_talib_adapter()
        if talib_adapter.available:
            result["talib"] = talib_adapter.list_indicators()
    except Exception as e:
        logger.debug(f"Could not list TA-Lib indicators: {e}")

    # Get pandas-ta indicators
    try:
        pandas_ta_adapter = get_pandas_ta_adapter()
        if pandas_ta_adapter.available:
            result["pandas_ta"] = pandas_ta_adapter.list_indicators()
    except Exception as e:
        logger.debug(f"Could not list pandas-ta indicators: {e}")

    # Get VectorBT indicators
    try:
        vbt_adapter = get_vectorbt_adapter()
        if vbt_adapter.available:
            result["vectorbt"] = vbt_adapter.list_indicators()
    except Exception as e:
        logger.debug(f"Could not list VectorBT indicators: {e}")

    return result


def list_alphapy_indicators() -> list[str]:
    """List all AlphaPy transforms.py indicators.

    Returns:
        List of indicator names (76 total)
    """
    registry = get_registry()
    if not registry._indicators:
        registry.discover_transforms()
    return sorted(registry.list_indicators())


# =============================================================================
# Transforms.py Metadata System
# =============================================================================

# Special handling flags that can't be auto-detected from signatures
SPECIAL_FUNCTIONS = {
    # Length-changing functions (return fewer elements than input)
    "diff": {"length_change": True},

    # Row-wise helper functions (return scalars, designed for df.apply())
    "c2max": {"row_wise": True},
    "c2min": {"row_wise": True},
    "gtval0": {"row_wise": True},

    # Multi-column return functions (return DataFrames)
    "dateparts": {"returns": "dataframe"},
    "bizday": {"returns": "dataframe"},
    "timeparts": {"returns": "dataframe"},
    "runstest": {"returns": "dataframe"},
    "texplode": {"returns": "dataframe"},

    # Special column handling (vwap returns DataFrame but single column)
    "vwap": {"returns": "dataframe"},
}

# Cache for transforms metadata
_transforms_metadata_cache: Optional[dict] = None


def get_transforms_metadata() -> dict:
    """Auto-discover transforms.py function signatures.

    Uses Python's inspect module to automatically detect function parameters
    and their default values. Results are cached for efficiency.

    Returns:
        Dictionary mapping function names to their metadata:
        {
            "gap": {"params": [], "defaults": {}},
            "rsi": {"params": ["p"], "defaults": {"p": 14}},
            "ma": {"params": ["c", "p"], "defaults": {"c": "close", "p": 20}},
        }
    """
    global _transforms_metadata_cache
    if _transforms_metadata_cache is not None:
        return _transforms_metadata_cache

    try:
        from alphapy import transforms
    except ImportError:
        logger.warning("Could not import alphapy.transforms for metadata discovery")
        return {}

    metadata = {}
    for name in dir(transforms):
        if name.startswith('_'):
            continue
        func = getattr(transforms, name)
        if not callable(func):
            continue

        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            continue

        params = []
        defaults = {}

        for pname, param in sig.parameters.items():
            if pname == 'df':  # Skip the DataFrame parameter
                continue
            params.append(pname)
            if param.default is not inspect.Parameter.empty:
                defaults[pname] = param.default

        # Merge with special handling flags
        func_metadata = {"params": params, "defaults": defaults}
        if name in SPECIAL_FUNCTIONS:
            func_metadata.update(SPECIAL_FUNCTIONS[name])

        metadata[name] = func_metadata

    _transforms_metadata_cache = metadata
    logger.debug(f"Auto-discovered metadata for {len(metadata)} transforms.py functions")
    return metadata


def ensure_aliases_loaded() -> None:
    """Ensure aliases from variables.yml are loaded into Alias.aliases.

    This function loads the aliases section from config/variables.yml
    into the Alias class, making them available for indicator parsing.

    This is idempotent - calling it multiple times is safe.
    """
    try:
        from alphapy.alias import Alias
    except ImportError:
        logger.debug("alphapy.alias not available, skipping alias loading")
        return

    # Only load if not already loaded
    if Alias.aliases:
        return

    # Try to load from config/variables.yml
    import os
    config_paths = [
        Path("config/variables.yml"),
        Path(__file__).parent.parent.parent / "config" / "variables.yml",
        Path(os.environ.get("ALPHAPY_ROOT", "")) / "config" / "variables.yml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    var_specs = yaml.safe_load(f) or {}
                aliases = var_specs.get('aliases', {})
                for name, expr in aliases.items():
                    Alias(name, expr)
                logger.debug(f"Loaded {len(aliases)} aliases from {config_path}")
                return
            except Exception as e:
                logger.debug(f"Could not load aliases from {config_path}: {e}")

    logger.debug("No variables.yml found, aliases not loaded")


def parse_transforms_string(indicator: str) -> tuple[str, list]:
    """Parse transforms.py indicator string into name and positional args.

    Matches vparse behavior from variables.py - simple positional arg parsing
    with alias expansion from variables.yml.
    This is the proven approach that has worked for 10+ years.

    Args:
        indicator: Indicator string like "rsi_14" or "ma_close_20" or "atr_14"

    Returns:
        Tuple of (function_name, args_list)

    Examples:
        "gap" → ("gap", [])
        "rsi_14" → ("rsi", [14])
        "ma_close_20" → ("ma", ["close", 20])
        "ema_high_10" → ("ema", ["high", 10])
        "diff_close_1" → ("diff", ["close", 1])
        "bbands_close_20_2.5" → ("bbands", ["close", 20, 2.5])
        "atr_14" → ("ma", ["truerange", 14])  # alias expansion
        "smac_20" → ("ma", ["close", 20])     # alias expansion
    """
    # Ensure aliases are loaded from variables.yml
    ensure_aliases_loaded()

    parts = indicator.lower().split("_")
    if not parts:
        return "", []

    name = parts[0]
    remaining_parts = parts[1:]

    # Check for alias expansion (like vparse does in variables.py)
    try:
        from alphapy.alias import get_alias
        alias = get_alias(name)
        if alias:
            # Expand alias: "atr" → "ma_truerange"
            alias_parts = alias.split("_")
            name = alias_parts[0]
            # Prepend alias params to remaining parts
            remaining_parts = alias_parts[1:] + remaining_parts
    except ImportError:
        # alphapy.alias not available, skip alias expansion
        pass

    # Convert remaining parts to int/float/str (like vfunc does in variables.py)
    args = []
    for part in remaining_parts:
        try:
            if "." in part:
                args.append(float(part))
            else:
                args.append(int(part))
        except ValueError:
            args.append(part)  # String (column name)

    return name, args


def is_row_wise_function(name: str) -> bool:
    """Check if a function is a row-wise helper (returns scalar).

    Args:
        name: Function name

    Returns:
        True if function returns scalar and is designed for df.apply()
    """
    metadata = get_transforms_metadata()
    func_meta = metadata.get(name, {})
    return func_meta.get("row_wise", False)


def is_length_changing_function(name: str) -> bool:
    """Check if a function changes the output length.

    Args:
        name: Function name

    Returns:
        True if function returns fewer elements than input
    """
    metadata = get_transforms_metadata()
    func_meta = metadata.get(name, {})
    return func_meta.get("length_change", False)


def returns_dataframe(name: str) -> bool:
    """Check if a function returns a DataFrame (multi-column).

    Args:
        name: Function name

    Returns:
        True if function returns DataFrame
    """
    metadata = get_transforms_metadata()
    func_meta = metadata.get(name, {})
    return func_meta.get("returns") == "dataframe"


# =============================================================================
# External Library Metadata Discovery
# =============================================================================

# Caches for external library metadata
_talib_metadata_cache: Optional[dict] = None
_pandas_ta_metadata_cache: Optional[dict] = None
_talipp_metadata_cache: Optional[dict] = None


def get_talib_metadata() -> dict:
    """Auto-discover TA-Lib function signatures.

    Dynamically introspects TA-Lib at import time using the abstract API.
    Results are cached for efficiency.

    Returns:
        Dictionary mapping function names to their metadata:
        {
            "RSI": {"params": ["timeperiod"], "defaults": {"timeperiod": 14}},
            "MACD": {"params": ["fastperiod", "slowperiod", "signalperiod"],
                     "defaults": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}},
        }
    """
    global _talib_metadata_cache
    if _talib_metadata_cache is not None:
        return _talib_metadata_cache

    try:
        import talib
        from talib import abstract
    except ImportError:
        logger.debug("TA-Lib not available for metadata discovery")
        return {}

    metadata = {}

    try:
        for func_name in talib.get_functions():
            try:
                # Use abstract API to get function info
                func_info = abstract.Function(func_name)

                params = []
                defaults = {}

                # Get parameter info from the abstract function
                for param_name, param_info in func_info.parameters.items():
                    params.append(param_name)
                    # param_info is the default value
                    if param_info is not None:
                        defaults[param_name] = param_info

                # Get input names (for multi-input indicators)
                input_names = list(func_info.input_names.keys()) if func_info.input_names else []

                # Get output names (for multi-output indicators)
                output_names = func_info.output_names if func_info.output_names else [func_name.lower()]

                metadata[func_name] = {
                    "params": params,
                    "defaults": defaults,
                    "inputs": input_names,
                    "outputs": output_names,
                }
            except Exception as e:
                logger.debug(f"Could not introspect TA-Lib function {func_name}: {e}")

    except Exception as e:
        logger.debug(f"Error discovering TA-Lib metadata: {e}")

    _talib_metadata_cache = metadata
    logger.debug(f"Auto-discovered metadata for {len(metadata)} TA-Lib functions")
    return metadata


def get_pandas_ta_metadata() -> dict:
    """Auto-discover pandas-ta indicator signatures.

    Dynamically introspects pandas-ta at import time.
    Results are cached for efficiency.

    Returns:
        Dictionary mapping indicator names to their metadata:
        {
            "rsi": {"params": ["length", "scalar", ...], "defaults": {"length": 14, ...}},
            "sma": {"params": ["length", ...], "defaults": {"length": 10}},
        }
    """
    global _pandas_ta_metadata_cache
    if _pandas_ta_metadata_cache is not None:
        return _pandas_ta_metadata_cache

    try:
        import pandas_ta as pta
    except ImportError:
        logger.debug("pandas-ta not available for metadata discovery")
        return {}

    metadata = {}

    # pandas-ta organizes indicators by category
    categories = ["candles", "cycles", "momentum", "overlap",
                  "performance", "statistics", "trend", "volatility", "volume"]

    for category in categories:
        cat_module = getattr(pta, category, None)
        if cat_module is None:
            continue

        for name in dir(cat_module):
            if name.startswith("_"):
                continue

            func = getattr(cat_module, name, None)
            if not callable(func):
                continue

            try:
                sig = inspect.signature(func)
                params = []
                defaults = {}

                for pname, param in sig.parameters.items():
                    # Skip common non-indicator params
                    if pname in ('open_', 'high', 'low', 'close', 'volume',
                                 'kwargs', 'args', 'talib', 'offset', 'append'):
                        continue

                    params.append(pname)
                    if param.default is not inspect.Parameter.empty:
                        defaults[pname] = param.default

                metadata[name.lower()] = {
                    "params": params,
                    "defaults": defaults,
                    "category": category,
                }
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not introspect pandas-ta {name}: {e}")

    _pandas_ta_metadata_cache = metadata
    logger.debug(f"Auto-discovered metadata for {len(metadata)} pandas-ta indicators")
    return metadata


def get_talipp_metadata() -> dict:
    """Auto-discover talipp indicator class signatures.

    Dynamically introspects talipp at import time.
    Results are cached for efficiency.

    Returns:
        Dictionary mapping indicator names to their metadata:
        {
            "sma": {"params": ["period"], "defaults": {}},
            "ema": {"params": ["period"], "defaults": {}},
            "rsi": {"params": ["period"], "defaults": {"period": 14}},
        }
    """
    global _talipp_metadata_cache
    if _talipp_metadata_cache is not None:
        return _talipp_metadata_cache

    try:
        from talipp import indicators as talipp_ind
    except ImportError:
        logger.debug("talipp not available for metadata discovery")
        return {}

    metadata = {}

    for name in dir(talipp_ind):
        if name.startswith("_"):
            continue

        cls = getattr(talipp_ind, name, None)
        if not isinstance(cls, type):
            continue

        try:
            # Get __init__ signature
            sig = inspect.signature(cls.__init__)
            params = []
            defaults = {}

            for pname, param in sig.parameters.items():
                # Skip self and common params
                if pname in ('self', 'input_values', 'input_indicator'):
                    continue

                params.append(pname)
                if param.default is not inspect.Parameter.empty:
                    defaults[pname] = param.default

            # Determine input type (close-only vs OHLCV)
            input_type = "close"
            if name.upper() in ("ATR", "ADX", "STOCH", "CCI", "WILLIAMS", "AROON", "OBV", "VWAP"):
                input_type = "ohlcv"

            metadata[name.lower()] = {
                "params": params,
                "defaults": defaults,
                "input_type": input_type,
                "class": name,
            }
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not introspect talipp {name}: {e}")

    _talipp_metadata_cache = metadata
    logger.debug(f"Auto-discovered metadata for {len(metadata)} talipp indicators")
    return metadata


def get_all_external_metadata() -> dict:
    """Get metadata for all external libraries.

    Returns:
        Dictionary with metadata for each available library:
        {
            "talib": {...},
            "pandas_ta": {...},
            "talipp": {...},
        }
    """
    return {
        "talib": get_talib_metadata(),
        "pandas_ta": get_pandas_ta_metadata(),
        "talipp": get_talipp_metadata(),
    }
