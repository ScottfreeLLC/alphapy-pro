"""Tests for agent.tools.order_execution module.

CRITICAL: All tests must verify that mocks prevent real Alpaca API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.order_execution import OrderExecutionTool


class TestOrderExecutionToolInit:
    """Tests for OrderExecutionTool initialization."""

    def test_tool_metadata(self, mock_alpaca_env):
        """Test tool name and description."""
        with patch("agent.tools.order_execution.AlpacaClient"):
            tool = OrderExecutionTool()

            assert tool.name == "execute_order"
            assert "trading order" in tool.description.lower()

    def test_input_schema(self, mock_alpaca_env):
        """Test input schema has required fields."""
        with patch("agent.tools.order_execution.AlpacaClient"):
            tool = OrderExecutionTool()

            assert "symbol" in tool.input_schema["properties"]
            assert "side" in tool.input_schema["properties"]
            assert "quantity" in tool.input_schema["properties"]
            assert "order_type" in tool.input_schema["properties"]


class TestOrderExecutionToolExecute:
    """Tests for execute method - CRITICAL: Must mock API calls."""

    @pytest.fixture
    def mock_tool(self, mock_alpaca_env):
        """Create a mock order execution tool."""
        with patch("agent.tools.order_execution.AlpacaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.submit_order.return_value = {
                "id": "test-order-123",
                "client_order_id": "client-123",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "order_type": "market",
                "status": "accepted",
            }
            mock_cls.return_value = mock_client

            tool = OrderExecutionTool()
            tool._client = mock_client
            yield tool

    @pytest.mark.asyncio
    async def test_execute_market_order(self, mock_tool):
        """Test executing a market order."""
        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=10,
            order_type="market",
        )

        data = json.loads(result)
        assert data["success"] is True
        assert "order" in data
        assert data["order"]["symbol"] == "AAPL"

        # CRITICAL: Verify mock was called, not real API
        mock_tool._client.submit_order.assert_called_once_with(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market",
            limit_price=None,
            stop_price=None,
            time_in_force="day",
        )

    @pytest.mark.asyncio
    async def test_execute_limit_order(self, mock_tool):
        """Test executing a limit order."""
        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=10,
            order_type="limit",
            limit_price=145.50,
        )

        data = json.loads(result)
        assert data["success"] is True

        mock_tool._client.submit_order.assert_called_once_with(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="limit",
            limit_price=145.50,
            stop_price=None,
            time_in_force="day",
        )

    @pytest.mark.asyncio
    async def test_execute_stop_order(self, mock_tool):
        """Test executing a stop order."""
        result = await mock_tool.execute(
            symbol="AAPL",
            side="sell",
            quantity=10,
            order_type="stop",
            stop_price=140.0,
        )

        data = json.loads(result)
        assert data["success"] is True

        mock_tool._client.submit_order.assert_called_with(
            symbol="AAPL",
            side="sell",
            qty=10,
            order_type="stop",
            limit_price=None,
            stop_price=140.0,
            time_in_force="day",
        )

    @pytest.mark.asyncio
    async def test_execute_stop_limit_order(self, mock_tool):
        """Test executing a stop limit order."""
        result = await mock_tool.execute(
            symbol="AAPL",
            side="sell",
            quantity=10,
            order_type="stop_limit",
            limit_price=139.0,
            stop_price=140.0,
        )

        data = json.loads(result)
        assert data["success"] is True

        mock_tool._client.submit_order.assert_called_with(
            symbol="AAPL",
            side="sell",
            qty=10,
            order_type="stop_limit",
            limit_price=139.0,
            stop_price=140.0,
            time_in_force="day",
        )

    @pytest.mark.asyncio
    async def test_execute_with_time_in_force(self, mock_tool):
        """Test executing order with different time in force."""
        await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=10,
            order_type="market",
            time_in_force="gtc",
        )

        mock_tool._client.submit_order.assert_called_with(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market",
            limit_price=None,
            stop_price=None,
            time_in_force="gtc",
        )

    @pytest.mark.asyncio
    async def test_execute_crypto_symbol(self, mock_tool):
        """Test executing order for crypto symbol."""
        result = await mock_tool.execute(
            symbol="BTC/USD",
            side="buy",
            quantity=0.5,
            order_type="market",
        )

        data = json.loads(result)
        assert data["success"] is True

        mock_tool._client.submit_order.assert_called_with(
            symbol="BTC/USD",
            side="buy",
            qty=0.5,
            order_type="market",
            limit_price=None,
            stop_price=None,
            time_in_force="day",
        )

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, mock_tool):
        """Test error handling when API fails."""
        mock_tool._client.submit_order.side_effect = Exception("Insufficient funds")

        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=10,
            order_type="market",
        )

        data = json.loads(result)
        assert data["success"] is False
        assert "error" in data
        assert "Insufficient funds" in data["error"]
        assert data["symbol"] == "AAPL"


class TestOrderExecutionToolCancel:
    """Tests for cancel method."""

    @pytest.fixture
    def mock_tool(self, mock_alpaca_env):
        """Create a mock order execution tool."""
        with patch("agent.tools.order_execution.AlpacaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.cancel_order.return_value = {
                "success": True,
                "order_id": "test-order-123",
            }
            mock_cls.return_value = mock_client

            tool = OrderExecutionTool()
            tool._client = mock_client
            yield tool

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, mock_tool):
        """Test canceling an order."""
        result = await mock_tool.cancel("test-order-123")

        data = json.loads(result)
        assert data["success"] is True

        mock_tool._client.cancel_order.assert_called_once_with("test-order-123")

    @pytest.mark.asyncio
    async def test_cancel_order_error(self, mock_tool):
        """Test error handling when cancel fails."""
        mock_tool._client.cancel_order.side_effect = Exception("Order not found")

        result = await mock_tool.cancel("invalid-order-id")

        data = json.loads(result)
        assert data["success"] is False
        assert "Order not found" in data["error"]


class TestOrderExecutionToolClosePosition:
    """Tests for close_position method."""

    @pytest.fixture
    def mock_tool(self, mock_alpaca_env):
        """Create a mock order execution tool."""
        with patch("agent.tools.order_execution.AlpacaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.close_position.return_value = {
                "id": "close-order-123",
                "symbol": "AAPL",
                "side": "sell",
                "qty": 100,
                "status": "accepted",
            }
            mock_cls.return_value = mock_client

            tool = OrderExecutionTool()
            tool._client = mock_client
            yield tool

    @pytest.mark.asyncio
    async def test_close_position_success(self, mock_tool):
        """Test closing a position."""
        result = await mock_tool.close_position("AAPL")

        data = json.loads(result)
        assert data["success"] is True
        assert "Position closed" in data["message"]

        mock_tool._client.close_position.assert_called_once_with("AAPL")

    @pytest.mark.asyncio
    async def test_close_position_error(self, mock_tool):
        """Test error handling when close fails."""
        mock_tool._client.close_position.side_effect = Exception("No position exists")

        result = await mock_tool.close_position("INVALID")

        data = json.loads(result)
        assert data["success"] is False
        assert "No position exists" in data["error"]


class TestMockingVerification:
    """CRITICAL: Verify mocks are working correctly."""

    def test_no_real_api_key_used(self, mock_alpaca_env):
        """Verify tests use mock environment, not real credentials."""
        import os

        # The mock_alpaca_env fixture sets test credentials
        assert os.environ.get("ALPACA_API_KEY") == "test-api-key"
        assert os.environ.get("ALPACA_API_SECRET") == "test-api-secret"
        assert os.environ.get("ALPACA_PAPER") == "true"

    @pytest.mark.asyncio
    async def test_mock_prevents_real_api_call(self, mock_alpaca_env):
        """Verify mock client is used, not real Alpaca."""
        with patch("agent.tools.order_execution.AlpacaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.submit_order.return_value = {"id": "mock-123"}
            mock_cls.return_value = mock_client

            tool = OrderExecutionTool()

            await tool.execute(
                symbol="AAPL",
                side="buy",
                quantity=10,
                order_type="market",
            )

            # Verify the mock was instantiated
            mock_cls.assert_called_once()

            # Verify submit_order was called on mock, not real client
            tool._client.submit_order.assert_called_once()
