"""Tests for agent.utils.alpaca_client module."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if alpaca-py is not installed
pytest.importorskip("alpaca")

from agent.utils.alpaca_client import AlpacaClient


class TestAlpacaClientInit:
    """Tests for AlpacaClient initialization."""

    def test_init_with_env_vars(self, mock_alpaca_env):
        """Test initialization with environment variables."""
        with patch("alpaca.trading.client.TradingClient") as mock:
            mock.return_value = MagicMock()

            client = AlpacaClient()

            assert client.api_key == "test-api-key"
            assert client.api_secret == "test-api-secret"
            assert client.paper is True

    def test_init_with_params(self):
        """Test initialization with explicit parameters."""
        with patch("alpaca.trading.client.TradingClient") as mock:
            mock.return_value = MagicMock()

            client = AlpacaClient(
                api_key="param-key",
                api_secret="param-secret",
                paper=False,
            )

            assert client.api_key == "param-key"
            assert client.api_secret == "param-secret"
            assert client.paper is False

    def test_init_missing_credentials(self):
        """Test initialization fails without credentials."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ALPACA_API_KEY", None)
            os.environ.pop("ALPACA_API_SECRET", None)

            with pytest.raises(ValueError, match="Alpaca credentials required"):
                AlpacaClient()


class TestAlpacaClientGetAccount:
    """Tests for get_account method."""

    @pytest.fixture
    def mock_client(self, mock_alpaca_env, mock_alpaca_account):
        """Create a mock Alpaca client."""
        with patch("alpaca.trading.client.TradingClient") as mock:
            trading_client = MagicMock()
            trading_client.get_account.return_value = mock_alpaca_account
            mock.return_value = trading_client

            client = AlpacaClient()
            yield client

    def test_get_account_returns_dict(self, mock_client):
        """Test get_account returns dictionary."""
        result = mock_client.get_account()

        assert isinstance(result, dict)
        assert "equity" in result
        assert "buying_power" in result
        assert "cash" in result

    def test_get_account_values(self, mock_client):
        """Test get_account returns correct values."""
        result = mock_client.get_account()

        assert result["equity"] == 100000.0
        assert result["buying_power"] == 80000.0
        assert result["cash"] == 50000.0


class TestAlpacaClientGetPositions:
    """Tests for get_positions method."""

    @pytest.fixture
    def mock_client(self, mock_alpaca_env, mock_alpaca_position):
        """Create a mock Alpaca client with position."""
        with patch("alpaca.trading.client.TradingClient") as mock:
            trading_client = MagicMock()
            trading_client.get_all_positions.return_value = [mock_alpaca_position]
            mock.return_value = trading_client

            client = AlpacaClient()
            yield client

    def test_get_positions_returns_list(self, mock_client):
        """Test get_positions returns list."""
        result = mock_client.get_positions()

        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_positions_dict_structure(self, mock_client):
        """Test position dictionary has correct structure."""
        result = mock_client.get_positions()
        pos = result[0]

        assert "symbol" in pos
        assert "qty" in pos
        assert "side" in pos
        assert "market_value" in pos
        assert "unrealized_pl" in pos


class TestAlpacaClientSubmitOrder:
    """Tests for submit_order method."""

    @pytest.fixture
    def mock_client(self, mock_alpaca_env, mock_alpaca_order):
        """Create a mock Alpaca client for orders."""
        with patch("alpaca.trading.client.TradingClient") as mock:
            trading_client = MagicMock()
            trading_client.submit_order.return_value = mock_alpaca_order
            mock.return_value = trading_client

            client = AlpacaClient()
            yield client

    def test_submit_market_order(self, mock_client):
        """Test submitting a market order."""
        result = mock_client.submit_order(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market",
        )

        assert result["symbol"] == "AAPL"
        assert result["id"] == "test-order-123"

    def test_submit_limit_order_requires_price(self, mock_client):
        """Test limit order requires limit_price."""
        with pytest.raises(ValueError, match="limit_price required"):
            mock_client.submit_order(
                symbol="AAPL",
                side="buy",
                qty=10,
                order_type="limit",
            )

    def test_submit_stop_order_requires_price(self, mock_client):
        """Test stop order requires stop_price."""
        with pytest.raises(ValueError, match="stop_price required"):
            mock_client.submit_order(
                symbol="AAPL",
                side="sell",
                qty=10,
                order_type="stop",
            )

    def test_submit_stop_limit_requires_both_prices(self, mock_client):
        """Test stop_limit order requires both prices."""
        with pytest.raises(ValueError, match="Both limit_price and stop_price"):
            mock_client.submit_order(
                symbol="AAPL",
                side="sell",
                qty=10,
                order_type="stop_limit",
                limit_price=150.0,
            )

    def test_invalid_order_type(self, mock_client):
        """Test invalid order type raises error."""
        with pytest.raises(ValueError, match="Invalid order type"):
            mock_client.submit_order(
                symbol="AAPL",
                side="buy",
                qty=10,
                order_type="invalid",
            )


class TestAlpacaClientCancelOrder:
    """Tests for cancel_order method."""

    @pytest.fixture
    def mock_client(self, mock_alpaca_env):
        """Create a mock Alpaca client for cancellation."""
        with patch("alpaca.trading.client.TradingClient") as mock:
            trading_client = MagicMock()
            trading_client.cancel_order_by_id.return_value = None
            mock.return_value = trading_client

            client = AlpacaClient()
            yield client

    def test_cancel_order_success(self, mock_client):
        """Test successful order cancellation."""
        result = mock_client.cancel_order("test-order-123")

        assert result["status"] == "cancelled"
        assert result["order_id"] == "test-order-123"

    def test_cancel_order_error(self, mock_client):
        """Test order cancellation error handling."""
        mock_client.client.cancel_order_by_id.side_effect = Exception("Not found")

        result = mock_client.cancel_order("invalid-id")

        assert result["status"] == "error"
        assert "error" in result


class TestAlpacaClientClosePosition:
    """Tests for close_position method."""

    @pytest.fixture
    def mock_client(self, mock_alpaca_env, mock_alpaca_order):
        """Create a mock Alpaca client for position closing."""
        with patch("alpaca.trading.client.TradingClient") as mock:
            trading_client = MagicMock()
            mock_alpaca_order.symbol = "AAPL"
            trading_client.close_position.return_value = mock_alpaca_order
            mock.return_value = trading_client

            client = AlpacaClient()
            yield client

    def test_close_position_success(self, mock_client):
        """Test successful position close."""
        result = mock_client.close_position("AAPL")

        assert result["symbol"] == "AAPL"
        assert "id" in result

    def test_close_position_error(self, mock_client):
        """Test position close error handling."""
        mock_client.client.close_position.side_effect = Exception("No position")

        result = mock_client.close_position("INVALID")

        assert result["status"] == "error"
        assert "error" in result
