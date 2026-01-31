"""Alpaca Markets data source implementation.

This module provides Alpaca API access for both stocks and crypto data.
All data returned as Polars DataFrames.

Note: Crypto data requires no authentication.
      Stock data requires API keys from https://alpaca.markets/
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Union

import polars as pl

from .base import DataSource, DataSourceConfig

logger = logging.getLogger(__name__)


class AlpacaDataSource(DataSource):
    """Alpaca Markets data source for stocks and crypto.

    Supports both stocks and crypto via the Alpaca API.
    Returns all data as Polars DataFrames.

    Crypto data is free and requires no authentication.
    Stock data requires API keys.
    """

    # Timeframe mappings to Alpaca TimeFrame objects
    # Built lazily to avoid import at module level
    _timeframe_map = None

    @classmethod
    def _get_timeframe_map(cls):
        """Get timeframe mapping, building lazily on first access."""
        if cls._timeframe_map is None:
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            cls._timeframe_map = {
                # Standard formats
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "30Min": TimeFrame(30, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
                "1Week": TimeFrame.Week,
                "1Month": TimeFrame.Month,
                # Pandas offset aliases (for AlphaPy compatibility)
                "1min": TimeFrame.Minute,
                "5min": TimeFrame(5, TimeFrameUnit.Minute),
                "15min": TimeFrame(15, TimeFrameUnit.Minute),
                "30min": TimeFrame(30, TimeFrameUnit.Minute),
                "1T": TimeFrame.Minute,
                "5T": TimeFrame(5, TimeFrameUnit.Minute),
                "15T": TimeFrame(15, TimeFrameUnit.Minute),
                "30T": TimeFrame(30, TimeFrameUnit.Minute),
                "1H": TimeFrame.Hour,
                "1h": TimeFrame.Hour,
                "1D": TimeFrame.Day,
                "1d": TimeFrame.Day,
                "D": TimeFrame.Day,
                "d": TimeFrame.Day,
                "W": TimeFrame.Week,
                "M": TimeFrame.Month,
            }
        return cls._timeframe_map

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """Initialize Alpaca data source.

        Args:
            config: Data source configuration
            api_key: Alpaca API key (optional for crypto data)
            api_secret: Alpaca API secret (optional for crypto data)
        """
        super().__init__(config)

        # Resolve API credentials: param > config > env
        self.api_key = (
            api_key
            or (self.config.extra.get("api_key") if self.config else None)
            or os.environ.get("ALPACA_API_KEY")
        )
        self.api_secret = (
            api_secret
            or (self.config.extra.get("api_secret") if self.config else None)
            or os.environ.get("ALPACA_API_SECRET")
        )

        # Import Alpaca clients
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

        # Crypto client - no auth required
        self._crypto_client = CryptoHistoricalDataClient()

        # Stock client - requires auth
        if self.api_key and self.api_secret:
            self._stock_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
        else:
            self._stock_client = None
            logger.info(
                "Alpaca stock client not initialized (no API keys). "
                "Crypto data will still work."
            )

    def is_crypto(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency.

        Alpaca crypto symbols use slash notation: BTC/USD, ETH/USD
        """
        return "/" in symbol

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to Alpaca format.

        Args:
            symbol: Standard symbol

        Returns:
            Alpaca-formatted symbol (already correct format for Alpaca)
        """
        return symbol.upper()

    def get_bars(
        self,
        symbols: Union[str, list[str]],
        timeframe: str = "1Day",
        lookback: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch historical OHLCV bars from Alpaca.

        Args:
            symbols: Single symbol or list of symbols
            timeframe: Bar timeframe (see _get_timeframe_map for options)
            lookback: Number of bars to fetch
            start_date: Start date (calculated from lookback if not provided)
            end_date: End date (defaults to now)

        Returns:
            Dictionary mapping symbol to Polars DataFrame with OHLCV data
        """
        symbols = self._ensure_list(symbols)
        timeframe_map = self._get_timeframe_map()

        if timeframe not in timeframe_map:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid options: {list(timeframe_map.keys())}"
            )

        alpaca_timeframe = timeframe_map[timeframe]
        end_date = end_date or datetime.now()

        # Calculate start date from lookback if not provided
        if start_date is None:
            start_date = self._calculate_start_date(end_date, lookback, timeframe)

        # Separate crypto and stock symbols
        crypto_symbols = [s for s in symbols if self.is_crypto(s)]
        stock_symbols = [s for s in symbols if not self.is_crypto(s)]

        results = {}

        # Fetch crypto data
        if crypto_symbols:
            crypto_results = self._fetch_crypto_bars(
                crypto_symbols, alpaca_timeframe, start_date, end_date, lookback
            )
            results.update(crypto_results)

        # Fetch stock data
        if stock_symbols:
            if self._stock_client is None:
                logger.error(
                    "Cannot fetch stock data without API keys. "
                    "Set ALPACA_API_KEY and ALPACA_API_SECRET."
                )
            else:
                stock_results = self._fetch_stock_bars(
                    stock_symbols, alpaca_timeframe, start_date, end_date, lookback
                )
                results.update(stock_results)

        return results

    def _calculate_start_date(
        self,
        end_date: datetime,
        lookback: int,
        timeframe: str,
    ) -> datetime:
        """Calculate start date based on lookback period."""
        timeframe_lower = timeframe.lower()

        if "min" in timeframe_lower or "t" in timeframe_lower:
            # Intraday - estimate trading minutes
            trading_minutes_per_day = 390
            multiplier = int("".join(filter(str.isdigit, timeframe)) or 1)
            bars_per_day = trading_minutes_per_day / multiplier
            days_needed = int(lookback / bars_per_day) + 5
        elif "h" in timeframe_lower:
            # Hourly
            bars_per_day = 6.5
            days_needed = int(lookback / bars_per_day) + 5
        elif "d" in timeframe_lower:
            # Daily
            days_needed = int(lookback * 1.5)
        elif "w" in timeframe_lower:
            # Weekly
            days_needed = lookback * 7 + 10
        elif "m" in timeframe_lower:
            # Monthly
            days_needed = lookback * 31 + 30
        else:
            days_needed = lookback + 10

        return end_date - timedelta(days=days_needed)

    def _fetch_crypto_bars(
        self,
        symbols: list[str],
        timeframe,
        start_date: datetime,
        end_date: datetime,
        lookback: int,
    ) -> dict[str, pl.DataFrame]:
        """Fetch bars for crypto symbols."""
        from alpaca.data.requests import CryptoBarsRequest

        results = {}

        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )

            bars = self._crypto_client.get_crypto_bars(request)

            # Convert to Polars DataFrames
            for symbol in symbols:
                try:
                    symbol_bars = bars[symbol]
                    if symbol_bars:
                        df = self._bars_to_polars(symbol_bars, symbol)
                        if not df.is_empty():
                            results[symbol] = df.tail(lookback)
                            logger.debug(f"Fetched {len(df)} crypto bars for {symbol}")
                except KeyError:
                    logger.warning(f"No crypto data returned for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching crypto bars: {e}")

        return results

    def _fetch_stock_bars(
        self,
        symbols: list[str],
        timeframe,
        start_date: datetime,
        end_date: datetime,
        lookback: int,
    ) -> dict[str, pl.DataFrame]:
        """Fetch bars for stock symbols."""
        from alpaca.data.requests import StockBarsRequest

        results = {}

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )

            bars = self._stock_client.get_stock_bars(request)

            # Convert to Polars DataFrames
            for symbol in symbols:
                try:
                    symbol_bars = bars[symbol]
                    if symbol_bars:
                        df = self._bars_to_polars(symbol_bars, symbol)
                        if not df.is_empty():
                            results[symbol] = df.tail(lookback)
                            logger.debug(f"Fetched {len(df)} stock bars for {symbol}")
                except KeyError:
                    logger.warning(f"No stock data returned for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching stock bars: {e}")

        return results

    def _bars_to_polars(self, bars, symbol: str) -> pl.DataFrame:
        """Convert Alpaca bars to Polars DataFrame.

        Args:
            bars: Alpaca bar data (list of Bar objects)
            symbol: Symbol for logging

        Returns:
            Polars DataFrame with OHLCV columns
        """
        if not bars:
            return pl.DataFrame()

        # Extract bar data
        data = []
        for bar in bars:
            data.append({
                "datetime": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": getattr(bar, "vwap", None),
                "trade_count": getattr(bar, "trade_count", None),
            })

        # Create Polars DataFrame
        df = pl.DataFrame(data)

        # Ensure datetime is proper type (remove timezone for consistency)
        if "datetime" in df.columns:
            df = df.with_columns(
                pl.col("datetime").dt.replace_time_zone(None)
            )

        # Reorder columns
        cols = ["datetime", "open", "high", "low", "close", "volume"]
        optional = ["vwap", "trade_count"]
        cols.extend([c for c in optional if c in df.columns and df[c].null_count() < len(df)])
        df = df.select([c for c in cols if c in df.columns])

        return df

    def get_quote(self, symbol: str) -> dict:
        """Get the latest quote for a symbol."""
        try:
            if self.is_crypto(symbol):
                from alpaca.data.requests import CryptoLatestQuoteRequest
                request = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
                quotes = self._crypto_client.get_crypto_latest_quote(request)
            else:
                if self._stock_client is None:
                    return {"symbol": symbol, "price": None, "error": "No stock API keys"}
                from alpaca.data.requests import StockLatestQuoteRequest
                request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
                quotes = self._stock_client.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    "symbol": symbol,
                    "price": (quote.ask_price + quote.bid_price) / 2 if quote.ask_price and quote.bid_price else None,
                    "bid": quote.bid_price,
                    "ask": quote.ask_price,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "timestamp": quote.timestamp,
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

    def _get_crypto_snapshot(self, symbol: str) -> Optional[dict]:
        """Get snapshot for a crypto symbol."""
        try:
            from alpaca.data.requests import CryptoSnapshotRequest
            request = CryptoSnapshotRequest(symbol_or_symbols=[symbol])
            snapshots = self._crypto_client.get_crypto_snapshot(request)

            if symbol in snapshots:
                snapshot = snapshots[symbol]
                daily_bar = snapshot.daily_bar
                prev_daily_bar = snapshot.previous_daily_bar

                last_price = daily_bar.close if daily_bar else None
                prev_close = prev_daily_bar.close if prev_daily_bar else None

                return {
                    "last_price": last_price,
                    "open": daily_bar.open if daily_bar else None,
                    "high": daily_bar.high if daily_bar else None,
                    "low": daily_bar.low if daily_bar else None,
                    "volume": daily_bar.volume if daily_bar else None,
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

    def _get_stock_snapshot(self, symbol: str) -> Optional[dict]:
        """Get snapshot for a stock symbol."""
        if self._stock_client is None:
            logger.warning("Cannot get stock snapshot without API keys")
            return None

        try:
            from alpaca.data.requests import StockSnapshotRequest
            request = StockSnapshotRequest(symbol_or_symbols=[symbol])
            snapshots = self._stock_client.get_stock_snapshot(request)

            if symbol in snapshots:
                snapshot = snapshots[symbol]
                daily_bar = snapshot.daily_bar
                prev_daily_bar = snapshot.previous_daily_bar

                last_price = daily_bar.close if daily_bar else None
                prev_close = prev_daily_bar.close if prev_daily_bar else None

                return {
                    "last_price": last_price,
                    "open": daily_bar.open if daily_bar else None,
                    "high": daily_bar.high if daily_bar else None,
                    "low": daily_bar.low if daily_bar else None,
                    "volume": daily_bar.volume if daily_bar else None,
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
