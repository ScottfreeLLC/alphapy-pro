"""Unified multi-source indicator system.

This module provides access to indicators from multiple sources:
- AlphaPy transforms.py (76 functions) with talipp streaming
- TA-Lib (200+ indicators)
- pandas-ta (130+ indicators)
- VectorBT (backtesting indicators)

All operations work with Polars DataFrames.

Architecture:
    The indicator system has three layers:
    1. Registry: Unified indicator lookup with source prefix parsing
    2. Engine: Multi-source routing with automatic streaming fallback
    3. External adapters: TA-Lib, pandas-ta, VectorBT

Source Prefixes:
    - ap: AlphaPy transforms.py (default when no prefix)
    - ta: TA-Lib
    - pta: pandas-ta
    - vbt: VectorBT

Usage:
    # Multi-source indicators with prefixes
    from alphapy.indicators import IndicatorEngine, add_indicators

    engine = IndicatorEngine()
    df = engine.compute(df, [
        "ap:rsi_14",           # AlphaPy transforms.py
        "ta:MACD_12_26_9",     # TA-Lib
        "pta:supertrend_10_3", # pandas-ta
        "vbt:rsi_14",          # VectorBT
        "sma_20",              # No prefix -> defaults to ap:
    ])

    # Or use the convenience function
    df = add_indicators(df, ["rsi_14", "ta:ATR_14", "ma_close_20"])

    # List all available indicators
    from alphapy.indicators import list_all_indicators
    indicators = list_all_indicators()
    # {"alphapy": [...], "talib": [...], "pandas_ta": [...], "vectorbt": [...]}
"""

# Engine exports (primary API)
from .engine import (
    IndicatorEngine,
    add_indicators,
    get_indicator_config,
    list_available_indicators,
    TALIPP_INDICATORS,
    INDICATOR_ALIASES,
)

# Registry exports (source prefix parsing and discovery)
from .registry import (
    IndicatorRegistry,
    get_registry,
    get_indicator,
    parse_indicator,
    discover_all,
    # Multi-source support
    SOURCE_PREFIXES,
    parse_source_prefix,
    parse_indicator_string,
    list_all_indicators,
    list_alphapy_indicators,
    # Alias loading
    ensure_aliases_loaded,
    # Dynamic metadata discovery
    get_transforms_metadata,
    get_talib_metadata,
    get_pandas_ta_metadata,
    get_talipp_metadata,
    get_all_external_metadata,
)
from .spec import (
    IndicatorSpec,
    IndicatorSource,
    IndicatorCategory,
    IndicatorParam,
)
from .external import (
    PandasTAAdapter,
    TALibAdapter,
    VectorBTAdapter,
    CustomModuleAdapter,
    get_pandas_ta_adapter,
    get_talib_adapter,
    get_vectorbt_adapter,
    compute_external,
    ExternalAdapterError,
)


__all__ = [
    # Engine exports (primary API)
    "IndicatorEngine",
    "add_indicators",
    "get_indicator_config",
    "list_available_indicators",
    "TALIPP_INDICATORS",
    "INDICATOR_ALIASES",
    # Registry exports (includes parse_indicator_string)
    "parse_indicator_string",
    # Registry exports
    "IndicatorRegistry",
    "IndicatorSpec",
    "IndicatorSource",
    "IndicatorCategory",
    "IndicatorParam",
    "get_registry",
    "get_indicator",
    "parse_indicator",
    "discover_all",
    # Multi-source support
    "SOURCE_PREFIXES",
    "parse_source_prefix",
    "list_all_indicators",
    "list_alphapy_indicators",
    # Alias loading
    "ensure_aliases_loaded",
    # Dynamic metadata discovery
    "get_transforms_metadata",
    "get_talib_metadata",
    "get_pandas_ta_metadata",
    "get_talipp_metadata",
    "get_all_external_metadata",
    # External library adapters
    "PandasTAAdapter",
    "TALibAdapter",
    "VectorBTAdapter",
    "CustomModuleAdapter",
    "get_pandas_ta_adapter",
    "get_talib_adapter",
    "get_vectorbt_adapter",
    "compute_external",
    "ExternalAdapterError",
]
