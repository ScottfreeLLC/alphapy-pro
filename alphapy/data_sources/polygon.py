"""Polygon.io data source implementation.

This module consolidates Polygon API access for both AlphaPy batch processing
and the trading agent's real-time needs. All data returned as Polars DataFrames.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Union

import polars as pl
from polygon import RESTClient

from .base import DataSource, DataSourceConfig

logger = logging.getLogger(__name__)


class PolygonDataSource(DataSource):
    """Polygon.io market data source.

    Supports both stocks and crypto via the Polygon API.
    Returns all data as Polars DataFrames.
    """

    # Unified timeframe mappings
    TIMEFRAME_MAP = {
        # Standard formats
        "1Min": ("minute", 1),
        "5Min": ("minute", 5),
        "15Min": ("minute", 15),
        "30Min": ("minute", 30),
        "1Hour": ("hour", 1),
        "1Day": ("day", 1),
        "1Week": ("week", 1),
        "1Month": ("month", 1),
        # Pandas offset aliases (for AlphaPy compatibility)
        "1min": ("minute", 1),
        "5min": ("minute", 5),
        "15min": ("minute", 15),
        "30min": ("minute", 30),
        "1T": ("minute", 1),
        "5T": ("minute", 5),
        "15T": ("minute", 15),
        "30T": ("minute", 30),
        "1H": ("hour", 1),
        "1D": ("day", 1),
        "D": ("day", 1),
        "W": ("week", 1),
        "M": ("month", 1),
    }

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize Polygon data source.

        Args:
            config: Data source configuration
            api_key: Polygon API key (overrides config and env var)
        """
        super().__init__(config)

        # Resolve API key: param > config > env
        self.api_key = (
            api_key
            or (self.config.extra.get("api_key") if self.config else None)
            or os.environ.get("POLYGON_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._client = RESTClient(api_key=self.api_key)

    @property
    def client(self) -> RESTClient:
        """Access the underlying Polygon REST client."""
        return self._client

    def normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to Polygon format.

        Args:
            symbol: Standard symbol (e.g., "AAPL", "BTC/USD")

        Returns:
            Polygon-formatted symbol (e.g., "AAPL", "X:BTCUSD")
        """
        symbol = symbol.upper()
        if "/" in symbol:
            base, quote = symbol.split("/")
            return f"X:{base}{quote}"
        return symbol

    def get_bars(
        self,
        symbols: Union[str, list[str]],
        timeframe: str = "1Day",
        lookback: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch historical OHLCV bars from Polygon.

        Args:
            symbols: Single symbol or list of symbols
            timeframe: Bar timeframe (see TIMEFRAME_MAP for options)
            lookback: Number of bars to fetch
            start_date: Start date (calculated from lookback if not provided)
            end_date: End date (defaults to now)

        Returns:
            Dictionary mapping symbol to Polars DataFrame with OHLCV data
        """
        symbols = self._ensure_list(symbols)

        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid options: {list(self.TIMEFRAME_MAP.keys())}"
            )

        timespan, multiplier = self.TIMEFRAME_MAP[timeframe]
        end_date = end_date or datetime.now()

        # Calculate start date from lookback if not provided
        if start_date is None:
            start_date = self._calculate_start_date(
                end_date, lookback, timespan, multiplier
            )

        results = {}
        for symbol in symbols:
            try:
                df = self._fetch_bars_single(
                    symbol=symbol,
                    timespan=timespan,
                    multiplier=multiplier,
                    start_date=start_date,
                    end_date=end_date,
                    limit=lookback,
                )
                if not df.is_empty():
                    results[symbol] = df
                    logger.debug(f"Fetched {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")

        return results

    def _calculate_start_date(
        self,
        end_date: datetime,
        lookback: int,
        timespan: str,
        multiplier: int,
    ) -> datetime:
        """Calculate start date based on lookback period."""
        if timespan == "minute":
            trading_minutes_per_day = 390
            bars_per_day = trading_minutes_per_day / multiplier
            days_needed = int(lookback / bars_per_day) + 5
        elif timespan == "hour":
            bars_per_day = 6.5 / multiplier
            days_needed = int(lookback / bars_per_day) + 5
        elif timespan == "day":
            days_needed = int(lookback * 1.5)
        elif timespan == "week":
            days_needed = lookback * 7 + 10
        elif timespan == "month":
            days_needed = lookback * 31 + 30
        else:
            days_needed = lookback + 10

        return end_date - timedelta(days=days_needed)

    def _fetch_bars_single(
        self,
        symbol: str,
        timespan: str,
        multiplier: int,
        start_date: datetime,
        end_date: datetime,
        limit: int,
    ) -> pl.DataFrame:
        """Fetch bars for a single symbol."""
        polygon_symbol = self.normalize_symbol(symbol)

        aggs = []
        for a in self._client.list_aggs(
            ticker=polygon_symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
            limit=50000,
        ):
            aggs.append({
                "timestamp": a.timestamp,
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume,
                "vwap": getattr(a, "vwap", None),
                "trades": getattr(a, "transactions", None),
            })

        if not aggs:
            return pl.DataFrame()

        # Create Polars DataFrame directly
        df = pl.DataFrame(aggs)

        # Convert timestamp to datetime
        df = df.with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("datetime")
        ).drop("timestamp")

        # Reorder columns
        cols = ["datetime", "open", "high", "low", "close", "volume"]
        optional = ["vwap", "trades"]
        cols.extend([c for c in optional if c in df.columns and df[c].null_count() < len(df)])
        df = df.select([c for c in cols if c in df.columns])

        # Return last N bars
        return df.tail(limit)

    def get_quote(self, symbol: str) -> dict:
        """Get the latest quote for a symbol."""
        polygon_symbol = self.normalize_symbol(symbol)

        try:
            trades = list(self._client.list_trades(
                polygon_symbol,
                limit=1,
            ))
            if trades:
                return {
                    "symbol": symbol,
                    "price": trades[0].price,
                    "size": trades[0].size,
                    "timestamp": datetime.fromtimestamp(
                        trades[0].sip_timestamp / 1e9
                    ),
                }
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")

        return {"symbol": symbol, "price": None, "error": "No data"}

    def get_snapshot(
        self, symbols: Union[str, list[str]]
    ) -> dict[str, dict]:
        """Get current market snapshot for symbols."""
        symbols = self._ensure_list(symbols)
        results = {}

        for symbol in symbols:
            try:
                if self.is_crypto(symbol):
                    snapshot = self._get_crypto_snapshot(symbol)
                else:
                    snapshot = self._get_stock_snapshot(symbol)

                if snapshot:
                    results[symbol] = snapshot
            except Exception as e:
                logger.warning(f"Error getting snapshot for {symbol}: {e}")

        return results

    def _get_stock_snapshot(self, symbol: str) -> Optional[dict]:
        """Get snapshot for a stock symbol."""
        polygon_symbol = self.normalize_symbol(symbol)

        try:
            snapshot = self._client.get_snapshot_ticker("stocks", polygon_symbol)
            if snapshot and snapshot.day:
                prev_close = getattr(snapshot.prev_day, "close", None)
                last_price = getattr(snapshot.day, "close", None)

                return {
                    "last_price": last_price,
                    "open": getattr(snapshot.day, "open", None),
                    "high": getattr(snapshot.day, "high", None),
                    "low": getattr(snapshot.day, "low", None),
                    "volume": getattr(snapshot.day, "volume", None),
                    "prev_close": prev_close,
                    "change": (
                        last_price - prev_close
                        if last_price and prev_close
                        else None
                    ),
                    "change_percent": (
                        (last_price - prev_close) / prev_close * 100
                        if last_price and prev_close
                        else None
                    ),
                }
        except Exception as e:
            logger.debug(f"Stock snapshot error for {symbol}: {e}")

        return None

    def _get_crypto_snapshot(self, symbol: str) -> Optional[dict]:
        """Get snapshot for a crypto symbol."""
        polygon_symbol = self.normalize_symbol(symbol)

        try:
            snapshot = self._client.get_snapshot_crypto(polygon_symbol)
            if snapshot and snapshot.day:
                prev_close = getattr(snapshot.prev_day, "close", None)
                last_price = getattr(snapshot.day, "close", None)

                return {
                    "last_price": last_price,
                    "open": getattr(snapshot.day, "open", None),
                    "high": getattr(snapshot.day, "high", None),
                    "low": getattr(snapshot.day, "low", None),
                    "volume": getattr(snapshot.day, "volume", None),
                    "prev_close": prev_close,
                    "change": (
                        last_price - prev_close
                        if last_price and prev_close
                        else None
                    ),
                    "change_percent": (
                        (last_price - prev_close) / prev_close * 100
                        if last_price and prev_close
                        else None
                    ),
                }
        except Exception as e:
            logger.debug(f"Crypto snapshot error for {symbol}: {e}")

        return None
