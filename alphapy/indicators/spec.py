"""Indicator specification dataclass for the unified registry system.

This module defines the IndicatorSpec dataclass that describes technical
indicators in a backend-agnostic way.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union

import polars as pl


class IndicatorSource(Enum):
    """Source of indicator implementation."""

    TRANSFORMS = "transforms"  # Built-in alphapy.transforms module
    TALIPP = "talipp"  # talipp streaming library
    PANDAS_TA = "pandas_ta"  # pandas-ta library
    TALIB = "talib"  # TA-Lib C library
    CUSTOM = "custom"  # User-defined module


class IndicatorCategory(Enum):
    """Category of technical indicator."""

    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OVERLAP = "overlap"  # Moving averages, etc.
    CALENDAR = "calendar"  # Date-based features
    STATISTICAL = "statistical"  # Statistical transforms
    OTHER = "other"


@dataclass
class IndicatorParam:
    """Parameter specification for an indicator."""

    name: str
    param_type: type
    default: Any
    description: str = ""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    def validate(self, value: Any) -> bool:
        """Validate a parameter value."""
        if not isinstance(value, self.param_type):
            try:
                value = self.param_type(value)
            except (ValueError, TypeError):
                return False

        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False

        return True


@dataclass
class IndicatorSpec:
    """Specification for a technical indicator.

    This dataclass describes an indicator's metadata, parameters,
    and how to compute it. The registry uses these specs to provide
    a unified interface regardless of the underlying implementation.

    Attributes:
        name: Canonical name of the indicator (e.g., "rsi", "adx", "ma")
        category: Category classification (trend, momentum, etc.)
        source: Where the implementation comes from
        description: Human-readable description
        params: List of parameter specifications
        aliases: Alternative names for this indicator
        outputs: Names of columns produced (for multi-output indicators)
        requires_ohlcv: Whether OHLCV data is required (vs just close)
        supports_streaming: Whether O(1) incremental updates are supported
        compute_func: Reference to the computation function
        streaming_class: Reference to talipp streaming class (if applicable)
    """

    name: str
    category: IndicatorCategory
    source: IndicatorSource
    description: str = ""
    params: list[IndicatorParam] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    requires_ohlcv: bool = False
    supports_streaming: bool = False
    compute_func: Optional[Callable] = None
    streaming_class: Optional[type] = None
    external_module: Optional[str] = None  # e.g., "pandas_ta.momentum"
    external_func: Optional[str] = None  # e.g., "rsi"

    def __post_init__(self):
        """Set default outputs to indicator name if not specified."""
        if not self.outputs:
            self.outputs = [self.name]

    @property
    def full_name(self) -> str:
        """Get fully qualified name with source."""
        return f"{self.source.value}:{self.name}"

    def get_default_params(self) -> dict[str, Any]:
        """Get dictionary of parameter names to default values."""
        return {p.name: p.default for p in self.params}

    def validate_params(self, **kwargs) -> dict[str, Any]:
        """Validate and fill in default parameters.

        Args:
            **kwargs: Parameter values to validate

        Returns:
            Dictionary of validated parameters with defaults filled in

        Raises:
            ValueError: If a parameter is invalid
        """
        result = self.get_default_params()

        # Known parameter mapping (common aliases)
        param_aliases = {
            "period": "p",
            "column": "c",
            "window": "p",
        }

        for key, value in kwargs.items():
            # Check for alias
            actual_key = param_aliases.get(key, key)

            # Find the matching param spec
            param_spec = None
            for p in self.params:
                if p.name == actual_key or p.name == key:
                    param_spec = p
                    actual_key = p.name
                    break

            if param_spec is None:
                # Unknown parameter - pass through for flexibility
                result[key] = value
            elif not param_spec.validate(value):
                raise ValueError(
                    f"Invalid value {value} for parameter {key} "
                    f"(expected {param_spec.param_type.__name__})"
                )
            else:
                result[actual_key] = param_spec.param_type(value)

        return result

    def compute(
        self,
        df: pl.DataFrame,
        **params
    ) -> Union[pl.Series, pl.DataFrame]:
        """Compute this indicator on a DataFrame.

        Args:
            df: Polars DataFrame with required columns
            **params: Indicator parameters

        Returns:
            Series or DataFrame with computed indicator values

        Raises:
            NotImplementedError: If no compute function is registered
        """
        if self.compute_func is None:
            raise NotImplementedError(
                f"No compute function registered for {self.name}"
            )

        validated_params = self.validate_params(**params)
        return self.compute_func(df, **validated_params)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "category": self.category.value,
            "source": self.source.value,
            "description": self.description,
            "params": [
                {
                    "name": p.name,
                    "type": p.param_type.__name__,
                    "default": p.default,
                    "description": p.description,
                }
                for p in self.params
            ],
            "aliases": self.aliases,
            "outputs": self.outputs,
            "requires_ohlcv": self.requires_ohlcv,
            "supports_streaming": self.supports_streaming,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndicatorSpec":
        """Create from dictionary."""
        params = [
            IndicatorParam(
                name=p["name"],
                param_type={"int": int, "float": float, "str": str, "bool": bool}.get(
                    p["type"], str
                ),
                default=p["default"],
                description=p.get("description", ""),
            )
            for p in data.get("params", [])
        ]

        return cls(
            name=data["name"],
            category=IndicatorCategory(data.get("category", "other")),
            source=IndicatorSource(data.get("source", "custom")),
            description=data.get("description", ""),
            params=params,
            aliases=data.get("aliases", []),
            outputs=data.get("outputs", [data["name"]]),
            requires_ohlcv=data.get("requires_ohlcv", False),
            supports_streaming=data.get("supports_streaming", False),
        )


# Common parameter definitions for reuse
PERIOD_PARAM = IndicatorParam(
    name="p",
    param_type=int,
    default=14,
    description="Lookback period",
    min_value=1,
)

COLUMN_PARAM = IndicatorParam(
    name="c",
    param_type=str,
    default="close",
    description="Column name to use",
)

STD_DEV_PARAM = IndicatorParam(
    name="sd",
    param_type=float,
    default=2.0,
    description="Number of standard deviations",
    min_value=0.1,
)
