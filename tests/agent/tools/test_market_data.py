"""Tests for agent.tools.market_data module."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from agent.tools.market_data import MarketDataTool


class TestMarketDataToolInit:
    """Tests for MarketDataTool initialization."""

    def test_tool_metadata(self, mock_polygon_env):
        """Test tool name and description."""
        with patch("agent.tools.market_data.PolygonDataSource"):
            tool = MarketDataTool()

            assert tool.name == "get_market_data"
            assert "market data" in tool.description.lower()

    def test_input_schema(self, mock_polygon_env):
        """Test input schema has required fields."""
        with patch("agent.tools.market_data.PolygonDataSource"):
            tool = MarketDataTool()

            assert "symbols" in tool.input_schema["properties"]
            assert "timeframe" in tool.input_schema["properties"]
            assert "lookback_bars" in tool.input_schema["properties"]


class TestMarketDataToolExecute:
    """Tests for execute method."""

    @pytest.fixture
    def mock_tool(self, mock_polygon_env, sample_ohlcv_df):
        """Create a mock market data tool."""
        with patch("agent.tools.market_data.PolygonDataSource") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_bars.return_value = {
                "AAPL": sample_ohlcv_df,
            }
            mock_cls.return_value = mock_client

            tool = MarketDataTool()
            tool._client = mock_client
            yield tool

    @pytest.mark.asyncio
    async def test_execute_single_symbol(self, mock_tool, sample_ohlcv_df):
        """Test fetching data for a single symbol."""
        result = await mock_tool.execute(
            symbols=["AAPL"],
            timeframe="5Min",
            lookback_bars=100,
        )

        data = json.loads(result)
        assert "AAPL" in data
        assert data["AAPL"]["count"] == len(sample_ohlcv_df)
        assert data["AAPL"]["latest_close"] is not None

    @pytest.mark.asyncio
    async def test_execute_multiple_symbols(self, mock_tool, sample_ohlcv_df):
        """Test fetching data for multiple symbols."""
        mock_tool._client.get_bars.return_value = {
            "AAPL": sample_ohlcv_df,
            "TSLA": sample_ohlcv_df,
        }

        result = await mock_tool.execute(
            symbols=["AAPL", "TSLA"],
            timeframe="5Min",
            lookback_bars=100,
        )

        data = json.loads(result)
        assert "AAPL" in data
        assert "TSLA" in data
        assert data["_summary"]["symbols_requested"] == 2

    @pytest.mark.asyncio
    async def test_execute_with_different_timeframes(self, mock_tool, sample_ohlcv_df):
        """Test different timeframe values."""
        for timeframe in ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"]:
            await mock_tool.execute(
                symbols=["AAPL"],
                timeframe=timeframe,
                lookback_bars=100,
            )

            mock_tool._client.get_bars.assert_called_with(
                symbols=["AAPL"],
                timeframe=timeframe,
                lookback=100,
            )

    @pytest.mark.asyncio
    async def test_execute_returns_summary(self, mock_tool, sample_ohlcv_df):
        """Test that response includes summary."""
        result = await mock_tool.execute(
            symbols=["AAPL"],
            timeframe="5Min",
            lookback_bars=100,
        )

        data = json.loads(result)
        assert "_summary" in data
        assert data["_summary"]["symbols_requested"] == 1
        assert data["_summary"]["timeframe"] == "5Min"
        assert data["_summary"]["lookback_bars"] == 100

    @pytest.mark.asyncio
    async def test_execute_handles_empty_data(self, mock_tool):
        """Test handling of empty data response."""
        mock_tool._client.get_bars.return_value = {
            "AAPL": pl.DataFrame(),  # Empty DataFrame
        }

        result = await mock_tool.execute(
            symbols=["AAPL"],
            timeframe="5Min",
            lookback_bars=100,
        )

        data = json.loads(result)
        assert data["AAPL"]["count"] == 0
        assert data["AAPL"]["bars"] == []
        assert data["AAPL"]["latest_close"] is None

    @pytest.mark.asyncio
    async def test_execute_crypto_symbol(self, mock_tool, sample_ohlcv_df):
        """Test fetching crypto symbol data."""
        mock_tool._client.get_bars.return_value = {
            "BTC/USD": sample_ohlcv_df,
        }

        result = await mock_tool.execute(
            symbols=["BTC/USD"],
            timeframe="5Min",
            lookback_bars=100,
        )

        data = json.loads(result)
        assert "BTC/USD" in data

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, mock_tool):
        """Test error handling when API fails."""
        mock_tool._client.get_bars.side_effect = Exception("API Error")

        result = await mock_tool.execute(
            symbols=["AAPL"],
            timeframe="5Min",
            lookback_bars=100,
        )

        data = json.loads(result)
        assert "error" in data
        assert "API Error" in data["error"]
        assert data["symbols"] == ["AAPL"]

    @pytest.mark.asyncio
    async def test_execute_default_values(self, mock_tool, sample_ohlcv_df):
        """Test default parameter values."""
        await mock_tool.execute(symbols=["AAPL"])

        mock_tool._client.get_bars.assert_called_with(
            symbols=["AAPL"],
            timeframe="5Min",  # default
            lookback=100,  # default
        )

    @pytest.mark.asyncio
    async def test_execute_returns_latest_time(self, mock_tool, sample_ohlcv_df):
        """Test that latest_time is included in response."""
        result = await mock_tool.execute(
            symbols=["AAPL"],
            timeframe="5Min",
            lookback_bars=100,
        )

        data = json.loads(result)
        assert "latest_time" in data["AAPL"]
        assert data["AAPL"]["latest_time"] is not None
