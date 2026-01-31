"""Tests for alphapy.data_sources.polygon module."""

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from alphapy.data_sources.base import DataSourceConfig
from alphapy.data_sources.polygon import PolygonDataSource


class TestPolygonDataSourceInit:
    """Tests for PolygonDataSource initialization."""

    def test_init_with_api_key_param(self):
        """Test initialization with API key parameter."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            ds = PolygonDataSource(api_key="test-key")

            assert ds.api_key == "test-key"
            mock.assert_called_once_with(api_key="test-key")

    def test_init_with_env_var(self, mock_polygon_env):
        """Test initialization with environment variable."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            ds = PolygonDataSource()

            assert ds.api_key == "test-polygon-key"
            mock.assert_called_once_with(api_key="test-polygon-key")

    def test_init_with_config_extra(self):
        """Test initialization with API key in config.extra."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            config = DataSourceConfig(extra={"api_key": "config-key"})
            ds = PolygonDataSource(config=config)

            assert ds.api_key == "config-key"
            mock.assert_called_once_with(api_key="config-key")

    def test_init_priority_param_over_env(self, mock_polygon_env):
        """Test that param API key takes priority over env."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            ds = PolygonDataSource(api_key="param-key")

            assert ds.api_key == "param-key"
            mock.assert_called_once_with(api_key="param-key")

    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing POLYGON_API_KEY
            os.environ.pop("POLYGON_API_KEY", None)

            with pytest.raises(ValueError, match="Polygon API key required"):
                PolygonDataSource()

    def test_client_property(self, mock_polygon_env):
        """Test client property returns RESTClient."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client

            ds = PolygonDataSource()

            assert ds.client is mock_client


class TestPolygonNormalizeSymbol:
    """Tests for symbol normalization."""

    def test_stock_symbol_uppercase(self, mock_polygon_env):
        """Test stock symbol is uppercased."""
        with patch("alphapy.data_sources.polygon.RESTClient"):
            ds = PolygonDataSource()

            assert ds.normalize_symbol("aapl") == "AAPL"
            assert ds.normalize_symbol("TSLA") == "TSLA"

    def test_crypto_symbol_format(self, mock_polygon_env):
        """Test crypto symbol converts to Polygon format."""
        with patch("alphapy.data_sources.polygon.RESTClient"):
            ds = PolygonDataSource()

            assert ds.normalize_symbol("BTC/USD") == "X:BTCUSD"
            assert ds.normalize_symbol("eth/usdt") == "X:ETHUSDT"


class TestPolygonTimeframes:
    """Tests for timeframe mapping."""

    def test_all_timeframes_mapped(self, mock_polygon_env):
        """Test all expected timeframes are in the map."""
        with patch("alphapy.data_sources.polygon.RESTClient"):
            ds = PolygonDataSource()

            # Standard formats
            assert "1Min" in ds.TIMEFRAME_MAP
            assert "5Min" in ds.TIMEFRAME_MAP
            assert "15Min" in ds.TIMEFRAME_MAP
            assert "30Min" in ds.TIMEFRAME_MAP
            assert "1Hour" in ds.TIMEFRAME_MAP
            assert "1Day" in ds.TIMEFRAME_MAP

            # Pandas aliases
            assert "1T" in ds.TIMEFRAME_MAP
            assert "5T" in ds.TIMEFRAME_MAP
            assert "D" in ds.TIMEFRAME_MAP

    @pytest.mark.parametrize("timeframe,expected", [
        ("1Min", ("minute", 1)),
        ("5Min", ("minute", 5)),
        ("1Hour", ("hour", 1)),
        ("1Day", ("day", 1)),
        ("1D", ("day", 1)),
        ("D", ("day", 1)),
    ])
    def test_timeframe_values(self, mock_polygon_env, timeframe, expected):
        """Test specific timeframe mappings."""
        with patch("alphapy.data_sources.polygon.RESTClient"):
            ds = PolygonDataSource()

            assert ds.TIMEFRAME_MAP[timeframe] == expected


class TestPolygonGetBars:
    """Tests for get_bars method."""

    def test_get_bars_single_symbol(self, mock_polygon_client, mock_polygon_env):
        """Test fetching bars for a single symbol."""
        ds = PolygonDataSource()

        result = ds.get_bars("AAPL", timeframe="1Day", lookback=100)

        assert "AAPL" in result
        assert isinstance(result["AAPL"], pl.DataFrame)
        assert len(result["AAPL"]) == 100

    def test_get_bars_multiple_symbols(self, mock_polygon_client, mock_polygon_env):
        """Test fetching bars for multiple symbols."""
        ds = PolygonDataSource()

        result = ds.get_bars(["AAPL", "TSLA"], timeframe="1Day", lookback=100)

        assert "AAPL" in result
        assert "TSLA" in result

    def test_get_bars_invalid_timeframe(self, mock_polygon_client, mock_polygon_env):
        """Test error on invalid timeframe."""
        ds = PolygonDataSource()

        with pytest.raises(ValueError, match="Invalid timeframe"):
            ds.get_bars("AAPL", timeframe="invalid")

    def test_get_bars_dataframe_columns(self, mock_polygon_client, mock_polygon_env):
        """Test DataFrame has expected columns."""
        ds = PolygonDataSource()

        result = ds.get_bars("AAPL", timeframe="1Day", lookback=10)
        df = result["AAPL"]

        assert "datetime" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_get_bars_with_date_range(self, mock_polygon_client, mock_polygon_env):
        """Test fetching bars with explicit date range."""
        ds = PolygonDataSource()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        result = ds.get_bars(
            "AAPL",
            timeframe="1Day",
            start_date=start_date,
            end_date=end_date
        )

        assert "AAPL" in result
        mock_polygon_client.list_aggs.assert_called()

    def test_get_bars_empty_response(self, mock_polygon_env):
        """Test handling of empty API response."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()
            client.list_aggs.return_value = []
            mock.return_value = client

            ds = PolygonDataSource()
            result = ds.get_bars("INVALID", timeframe="1Day", lookback=10)

            # Should not include the symbol if no data
            assert "INVALID" not in result

    def test_get_bars_api_error(self, mock_polygon_env):
        """Test handling of API error."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()
            client.list_aggs.side_effect = Exception("API Error")
            mock.return_value = client

            ds = PolygonDataSource()
            result = ds.get_bars("AAPL", timeframe="1Day", lookback=10)

            # Should return empty dict on error
            assert "AAPL" not in result


class TestPolygonGetQuote:
    """Tests for get_quote method."""

    def test_get_quote_success(self, mock_polygon_env):
        """Test successful quote retrieval."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()

            mock_trade = MagicMock()
            mock_trade.price = 150.25
            mock_trade.size = 100
            mock_trade.sip_timestamp = int(datetime.now().timestamp() * 1e9)

            client.list_trades.return_value = [mock_trade]
            mock.return_value = client

            ds = PolygonDataSource()
            quote = ds.get_quote("AAPL")

            assert quote["symbol"] == "AAPL"
            assert quote["price"] == 150.25
            assert quote["size"] == 100
            assert "timestamp" in quote

    def test_get_quote_no_trades(self, mock_polygon_env):
        """Test quote when no trades available."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()
            client.list_trades.return_value = []
            mock.return_value = client

            ds = PolygonDataSource()
            quote = ds.get_quote("AAPL")

            assert quote["symbol"] == "AAPL"
            assert quote["price"] is None
            assert "error" in quote

    def test_get_quote_api_error(self, mock_polygon_env):
        """Test quote on API error."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()
            client.list_trades.side_effect = Exception("API Error")
            mock.return_value = client

            ds = PolygonDataSource()
            quote = ds.get_quote("AAPL")

            assert quote["symbol"] == "AAPL"
            assert quote["price"] is None


class TestPolygonGetSnapshot:
    """Tests for get_snapshot method."""

    def test_get_stock_snapshot(self, mock_polygon_env):
        """Test stock snapshot retrieval."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()

            mock_snapshot = MagicMock()
            mock_snapshot.day = MagicMock()
            mock_snapshot.day.open = 148.0
            mock_snapshot.day.high = 152.0
            mock_snapshot.day.low = 147.5
            mock_snapshot.day.close = 151.0
            mock_snapshot.day.volume = 50000000
            mock_snapshot.prev_day = MagicMock()
            mock_snapshot.prev_day.close = 149.0

            client.get_snapshot_ticker.return_value = mock_snapshot
            mock.return_value = client

            ds = PolygonDataSource()
            result = ds.get_snapshot("AAPL")

            assert "AAPL" in result
            snap = result["AAPL"]
            assert snap["open"] == 148.0
            assert snap["high"] == 152.0
            assert snap["low"] == 147.5
            assert snap["last_price"] == 151.0
            assert snap["volume"] == 50000000
            assert snap["prev_close"] == 149.0
            assert snap["change"] == 2.0  # 151 - 149

    def test_get_crypto_snapshot(self, mock_polygon_env):
        """Test crypto snapshot retrieval."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()

            mock_snapshot = MagicMock()
            mock_snapshot.day = MagicMock()
            mock_snapshot.day.open = 40000.0
            mock_snapshot.day.high = 41000.0
            mock_snapshot.day.low = 39500.0
            mock_snapshot.day.close = 40500.0
            mock_snapshot.day.volume = 1000000
            mock_snapshot.prev_day = MagicMock()
            mock_snapshot.prev_day.close = 40000.0

            client.get_snapshot_crypto.return_value = mock_snapshot
            mock.return_value = client

            ds = PolygonDataSource()
            result = ds.get_snapshot("BTC/USD")

            assert "BTC/USD" in result
            snap = result["BTC/USD"]
            assert snap["last_price"] == 40500.0
            assert snap["change_percent"] == 1.25  # (40500-40000)/40000*100

    def test_get_snapshot_multiple_symbols(self, mock_polygon_env):
        """Test snapshot for multiple symbols."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()

            mock_snapshot = MagicMock()
            mock_snapshot.day = MagicMock()
            mock_snapshot.day.open = 150.0
            mock_snapshot.day.high = 152.0
            mock_snapshot.day.low = 149.0
            mock_snapshot.day.close = 151.0
            mock_snapshot.day.volume = 1000000
            mock_snapshot.prev_day = MagicMock()
            mock_snapshot.prev_day.close = 150.0

            client.get_snapshot_ticker.return_value = mock_snapshot
            mock.return_value = client

            ds = PolygonDataSource()
            result = ds.get_snapshot(["AAPL", "TSLA"])

            assert "AAPL" in result
            assert "TSLA" in result

    def test_get_snapshot_null_day(self, mock_polygon_env):
        """Test snapshot when day data is null."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()

            mock_snapshot = MagicMock()
            mock_snapshot.day = None

            client.get_snapshot_ticker.return_value = mock_snapshot
            mock.return_value = client

            ds = PolygonDataSource()
            result = ds.get_snapshot("AAPL")

            assert "AAPL" not in result

    def test_get_snapshot_api_error(self, mock_polygon_env):
        """Test snapshot on API error."""
        with patch("alphapy.data_sources.polygon.RESTClient") as mock:
            client = MagicMock()
            client.get_snapshot_ticker.side_effect = Exception("API Error")
            mock.return_value = client

            ds = PolygonDataSource()
            result = ds.get_snapshot("AAPL")

            assert "AAPL" not in result


class TestPolygonCalculateStartDate:
    """Tests for start date calculation."""

    def test_minute_timeframe_start_date(self, mock_polygon_env):
        """Test start date calculation for minute timeframe."""
        with patch("alphapy.data_sources.polygon.RESTClient"):
            ds = PolygonDataSource()
            end_date = datetime(2024, 1, 15)

            start = ds._calculate_start_date(
                end_date=end_date,
                lookback=100,
                timespan="minute",
                multiplier=5
            )

            # Should be at least a day before end_date
            assert start < end_date
            # 100 bars at 5-min = ~8 hours, so should be within 10 days
            assert (end_date - start).days <= 10

    def test_day_timeframe_start_date(self, mock_polygon_env):
        """Test start date calculation for day timeframe."""
        with patch("alphapy.data_sources.polygon.RESTClient"):
            ds = PolygonDataSource()
            end_date = datetime(2024, 1, 15)

            start = ds._calculate_start_date(
                end_date=end_date,
                lookback=100,
                timespan="day",
                multiplier=1
            )

            # 100 days * 1.5 = 150 days lookback
            expected_days = int(100 * 1.5)
            assert (end_date - start).days == expected_days

    def test_week_timeframe_start_date(self, mock_polygon_env):
        """Test start date calculation for week timeframe."""
        with patch("alphapy.data_sources.polygon.RESTClient"):
            ds = PolygonDataSource()
            end_date = datetime(2024, 1, 15)

            start = ds._calculate_start_date(
                end_date=end_date,
                lookback=52,
                timespan="week",
                multiplier=1
            )

            # 52 weeks * 7 + 10 = 374 days
            expected_days = 52 * 7 + 10
            assert (end_date - start).days == expected_days
