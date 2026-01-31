"""Tests for agent.tools.portfolio_state module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.portfolio_state import PortfolioStateTool


class TestPortfolioStateToolInit:
    """Tests for PortfolioStateTool initialization."""

    def test_tool_metadata(self, mock_alpaca_env):
        """Test tool name and description."""
        with patch("agent.tools.portfolio_state.AlpacaClient"):
            with patch("agent.tools.portfolio_state.LivePortfolio"):
                tool = PortfolioStateTool()

                assert tool.name == "get_portfolio_state"
                assert "portfolio state" in tool.description.lower()

    def test_input_schema(self, mock_alpaca_env):
        """Test input schema has expected fields."""
        with patch("agent.tools.portfolio_state.AlpacaClient"):
            with patch("agent.tools.portfolio_state.LivePortfolio"):
                tool = PortfolioStateTool()

                assert "include_orders" in tool.input_schema["properties"]
                assert "include_history" in tool.input_schema["properties"]


class TestPortfolioStateToolExecute:
    """Tests for execute method."""

    @pytest.fixture
    def mock_tool(self, mock_alpaca_env):
        """Create a mock portfolio state tool."""
        with patch("agent.tools.portfolio_state.AlpacaClient") as mock_client_cls:
            with patch("agent.tools.portfolio_state.LivePortfolio") as mock_portfolio_cls:
                mock_client = MagicMock()
                mock_client.get_account.return_value = {
                    "equity": 100000.0,
                    "last_equity": 99500.0,
                    "trading_blocked": False,
                    "account_blocked": False,
                    "pattern_day_trader": False,
                    "daytrade_count": 0,
                }
                mock_client.get_orders.return_value = []
                mock_client_cls.return_value = mock_client

                mock_portfolio = MagicMock()
                mock_portfolio.positions = {}
                mock_portfolio.long_market_value = 0.0
                mock_portfolio.short_market_value = 0.0
                mock_portfolio.unrealized_pl = 0.0
                mock_portfolio.cash = 50000.0
                mock_portfolio.equity = 100000.0
                mock_portfolio.summary.return_value = {
                    "equity": 100000.0,
                    "cash": 50000.0,
                }
                mock_portfolio.exposure.return_value = {
                    "long_exposure": 0.5,
                    "short_exposure": 0.0,
                    "net_exposure": 0.5,
                    "gross_exposure": 0.5,
                }
                mock_portfolio_cls.return_value = mock_portfolio

                tool = PortfolioStateTool()
                tool._client = mock_client
                tool._portfolio = mock_portfolio
                yield tool

    @pytest.mark.asyncio
    async def test_execute_basic(self, mock_tool):
        """Test basic execute returns portfolio state."""
        result = await mock_tool.execute()

        data = json.loads(result)
        assert "account" in data
        assert "positions" in data
        assert "daily_pnl" in data
        assert "risk_metrics" in data
        assert "status" in data

    @pytest.mark.asyncio
    async def test_execute_calculates_daily_pnl(self, mock_tool):
        """Test daily P&L calculation."""
        result = await mock_tool.execute()

        data = json.loads(result)
        # equity: 100000, last_equity: 99500
        # daily_pl = 500, daily_pl_pct = 0.5025%
        assert data["daily_pnl"]["value"] == 500.0
        assert abs(data["daily_pnl"]["percent"] - 0.50) < 0.01

    @pytest.mark.asyncio
    async def test_execute_with_positions(self, mock_tool):
        """Test execute with positions."""
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = 100
        mock_position.side = "long"
        mock_position.market_value = 15000.0
        mock_position.unrealized_pl = 500.0
        mock_position.unrealized_plpc = 0.0345
        mock_position.current_price = 150.0
        mock_position.avg_entry_price = 145.0

        mock_tool._portfolio.positions = {"AAPL": mock_position}
        mock_tool._portfolio.long_market_value = 15000.0

        result = await mock_tool.execute()

        data = json.loads(result)
        assert data["positions"]["count"] == 1
        assert len(data["positions"]["details"]) == 1
        assert data["positions"]["details"][0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_execute_include_orders(self, mock_tool):
        """Test execute with include_orders=True."""
        mock_tool._client.get_orders.return_value = [
            {"id": "order-1", "symbol": "AAPL", "status": "open"},
        ]

        result = await mock_tool.execute(include_orders=True)

        data = json.loads(result)
        assert "open_orders" in data
        assert data["open_orders"]["count"] == 1

    @pytest.mark.asyncio
    async def test_execute_include_history(self, mock_tool):
        """Test execute with include_history=True."""
        mock_tool._client.get_orders.return_value = [
            {"id": "order-1", "symbol": "AAPL", "status": "filled"},
            {"id": "order-2", "symbol": "TSLA", "status": "filled"},
        ]

        result = await mock_tool.execute(include_history=True)

        data = json.loads(result)
        assert "todays_fills" in data
        assert data["todays_fills"]["count"] == 2

    @pytest.mark.asyncio
    async def test_execute_risk_metrics(self, mock_tool):
        """Test risk metrics are included."""
        result = await mock_tool.execute()

        data = json.loads(result)
        assert "risk_metrics" in data
        assert "long_exposure" in data["risk_metrics"]
        assert "short_exposure" in data["risk_metrics"]
        assert "net_exposure" in data["risk_metrics"]
        assert "gross_exposure" in data["risk_metrics"]

    @pytest.mark.asyncio
    async def test_execute_status_info(self, mock_tool):
        """Test status information is included."""
        result = await mock_tool.execute()

        data = json.loads(result)
        assert "status" in data
        assert data["status"]["trading_blocked"] is False
        assert data["status"]["account_blocked"] is False

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, mock_tool):
        """Test error handling."""
        mock_tool._portfolio.sync.side_effect = Exception("API Error")

        result = await mock_tool.execute()

        data = json.loads(result)
        assert "error" in data
        assert "API Error" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_syncs_portfolio(self, mock_tool):
        """Test that execute syncs portfolio."""
        await mock_tool.execute()

        mock_tool._portfolio.sync.assert_called_once()
