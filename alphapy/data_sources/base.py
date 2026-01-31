"""Abstract base class for data sources."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    # Rate limiting
    requests_per_minute: int = 100

    # Additional provider-specific settings
    extra: dict = field(default_factory=dict)


class DataSource(ABC):
    """Abstract base class for market data sources.

    Provides a unified interface for:
    - Historical bar data (OHLCV)
    - Real-time quotes
    - Market snapshots

    All data is returned as Polars DataFrames.
    """

    def __init__(self, config: Optional[DataSourceConfig] = None):
        """Initialize data source.

        Args:
            config: Data source configuration. Uses defaults if not provided.
        """
        self.config = config or DataSourceConfig()

    @property
    def name(self) -> str:
        """Return the name of this data source."""
        return self.__class__.__name__.replace("DataSource", "").lower()

    @abstractmethod
    def get_bars(
        self,
        symbols: Union[str, list[str]],
        timeframe: str = "1Day",
        lookback: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch historical OHLCV bars.

        Args:
            symbols: Single symbol or list of symbols
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 30Min, 1Hour, 1Day)
            lookback: Number of bars to fetch (used if start_date not provided)
            start_date: Start date for data
            end_date: End date for data (defaults to now)

        Returns:
            Dictionary mapping symbol to Polars DataFrame with columns:
            - datetime (index)
            - open, high, low, close, volume
            - Optional: vwap, trades
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> dict:
        """Get the latest quote for a symbol.

        Args:
            symbol: Stock or crypto symbol

        Returns:
            Dictionary with:
            - symbol: str
            - price: float (last trade price)
            - bid: float (optional)
            - ask: float (optional)
            - size: int (last trade size)
            - timestamp: datetime
        """
        pass

    @abstractmethod
    def get_snapshot(
        self, symbols: Union[str, list[str]]
    ) -> dict[str, dict]:
        """Get current market snapshot for symbols.

        Args:
            symbols: Single symbol or list of symbols

        Returns:
            Dictionary mapping symbol to snapshot data:
            - last_price: float
            - open: float
            - high: float
            - low: float
            - volume: int
            - prev_close: float
            - change: float (optional)
            - change_percent: float (optional)
        """
        pass

    def is_crypto(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency."""
        return "/" in symbol

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to provider-specific format."""
        return symbol.upper()

    def _ensure_list(self, symbols: Union[str, list[str]]) -> list[str]:
        """Ensure symbols is a list."""
        if isinstance(symbols, str):
            return [symbols]
        return list(symbols)
