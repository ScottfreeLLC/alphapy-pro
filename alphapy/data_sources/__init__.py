"""Data source abstractions for unified market data access.

This module provides a unified interface for fetching market data from
various providers (Polygon, Yahoo, Alpaca, etc.) with support for both
Pandas and Polars DataFrames.
"""

from .base import DataSource, DataSourceConfig
from .alpaca import AlpacaDataSource
from .polygon import PolygonDataSource

__all__ = [
    "DataSource",
    "DataSourceConfig",
    "AlpacaDataSource",
    "PolygonDataSource",
]
