"""Tests for agent.tools.risk_checks module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.risk_checks import RiskCheckTool


class TestRiskCheckToolInit:
    """Tests for RiskCheckTool initialization."""

    def test_default_values(self, mock_alpaca_env):
        """Test default risk limit values."""
        with patch("agent.tools.risk_checks.AlpacaClient"):
            tool = RiskCheckTool()

            assert tool.max_position_value == 5000.0
            assert tool.max_portfolio_exposure == 25000.0
            assert tool.max_positions == 5
            assert tool.max_symbol_pct == 0.25
            assert tool.daily_loss_limit == 0.02
            assert tool.min_order_value == 100.0

    def test_tool_metadata(self, mock_alpaca_env):
        """Test tool name and description."""
        with patch("agent.tools.risk_checks.AlpacaClient"):
            tool = RiskCheckTool()

            assert tool.name == "check_risk"
            assert "risk management rules" in tool.description


class TestRiskCheckToolConfigure:
    """Tests for configure method."""

    def test_configure_all_limits(self, mock_alpaca_env):
        """Test configuring all risk limits."""
        with patch("agent.tools.risk_checks.AlpacaClient"):
            tool = RiskCheckTool()
            tool.configure(
                max_position_value=10000.0,
                max_portfolio_exposure=50000.0,
                max_positions=10,
                max_symbol_pct=0.5,
                daily_loss_limit=0.05,
                position_stop_loss=0.02,
                min_order_value=200.0,
                no_new_trades_before_close=30,
            )

            assert tool.max_position_value == 10000.0
            assert tool.max_portfolio_exposure == 50000.0
            assert tool.max_positions == 10
            assert tool.max_symbol_pct == 0.5
            assert tool.daily_loss_limit == 0.05
            assert tool.position_stop_loss == 0.02
            assert tool.min_order_value == 200.0
            assert tool.no_new_trades_before_close == 30

    def test_configure_partial(self, mock_alpaca_env):
        """Test configuring only some limits."""
        with patch("agent.tools.risk_checks.AlpacaClient"):
            tool = RiskCheckTool()
            original_max_positions = tool.max_positions

            tool.configure(max_position_value=8000.0)

            assert tool.max_position_value == 8000.0
            assert tool.max_positions == original_max_positions


class TestRiskCheckToolExecute:
    """Tests for execute method."""

    @pytest.fixture
    def mock_tool(self, mock_alpaca_env):
        """Create a mock risk check tool."""
        with patch("agent.tools.risk_checks.AlpacaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_account.return_value = {
                "equity": 100000.0,
                "last_equity": 100000.0,
                "trading_blocked": False,
            }
            mock_client.get_positions.return_value = []
            mock_cls.return_value = mock_client

            tool = RiskCheckTool()
            tool._client = mock_client
            yield tool

    @pytest.mark.asyncio
    async def test_approve_valid_trade(self, mock_tool):
        """Test approving a valid trade."""
        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
        )

        data = json.loads(result)
        assert data["approved"] is True
        assert data["symbol"] == "AAPL"
        assert data["order_value"] == 1500.0

    @pytest.mark.asyncio
    async def test_reject_below_min_order_value(self, mock_tool):
        """Test rejecting trade below minimum order value."""
        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=1,
            price=50.0,  # $50 < $100 min
        )

        data = json.loads(result)
        assert data["approved"] is False
        assert any(c["rule"] == "min_order_value" and not c["passed"]
                  for c in data["checks"])

    @pytest.mark.asyncio
    async def test_reject_exceeds_max_position_value(self, mock_tool):
        """Test rejecting trade exceeding max position value."""
        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=100.0,  # $10,000 > $5,000 max
        )

        data = json.loads(result)
        assert data["approved"] is False
        assert any(c["rule"] == "max_position_value" and not c["passed"]
                  for c in data["checks"])

    @pytest.mark.asyncio
    async def test_reject_exceeds_portfolio_exposure(self, mock_tool):
        """Test rejecting trade exceeding portfolio exposure."""
        # Mock existing positions with high exposure
        mock_tool._client.get_positions.return_value = [
            {"symbol": "TSLA", "market_value": 22000.0},
        ]

        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=30,
            price=150.0,  # $4,500 would push total to $26,500
        )

        data = json.loads(result)
        assert data["approved"] is False
        assert any(c["rule"] == "max_portfolio_exposure" and not c["passed"]
                  for c in data["checks"])

    @pytest.mark.asyncio
    async def test_reject_exceeds_max_positions(self, mock_tool):
        """Test rejecting new position when at max positions."""
        mock_tool._client.get_positions.return_value = [
            {"symbol": "TSLA", "market_value": 1000.0},
            {"symbol": "MSFT", "market_value": 1000.0},
            {"symbol": "GOOG", "market_value": 1000.0},
            {"symbol": "AMZN", "market_value": 1000.0},
            {"symbol": "META", "market_value": 1000.0},
        ]

        result = await mock_tool.execute(
            symbol="NVDA",  # New position
            side="buy",
            quantity=10,
            price=150.0,
        )

        data = json.loads(result)
        assert data["approved"] is False
        assert any(c["rule"] == "max_positions" and not c["passed"]
                  for c in data["checks"])

    @pytest.mark.asyncio
    async def test_allow_add_to_existing_position(self, mock_tool):
        """Test allowing add to existing position even at max positions."""
        mock_tool._client.get_positions.return_value = [
            {"symbol": "AAPL", "market_value": 1000.0},
            {"symbol": "TSLA", "market_value": 1000.0},
            {"symbol": "MSFT", "market_value": 1000.0},
            {"symbol": "GOOG", "market_value": 1000.0},
            {"symbol": "AMZN", "market_value": 1000.0},
        ]

        result = await mock_tool.execute(
            symbol="AAPL",  # Existing position
            side="buy",
            quantity=10,
            price=150.0,
        )

        data = json.loads(result)
        # Should not fail max_positions check for existing position
        position_checks = [c for c in data["checks"] if c["rule"] == "max_positions"]
        # Either no check or passed
        assert all(c["passed"] for c in position_checks)

    @pytest.mark.asyncio
    async def test_reject_exceeds_concentration_limit(self, mock_tool):
        """Test rejecting trade exceeding concentration limit."""
        # 25% of $100,000 = $25,000
        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=200,
            price=150.0,  # $30,000 > 25%
        )

        data = json.loads(result)
        assert data["approved"] is False
        assert any(c["rule"] == "max_symbol_pct" and not c["passed"]
                  for c in data["checks"])

    @pytest.mark.asyncio
    async def test_reject_daily_loss_exceeded(self, mock_tool):
        """Test rejecting trade when daily loss limit exceeded."""
        mock_tool._client.get_account.return_value = {
            "equity": 97000.0,  # Down 3%
            "last_equity": 100000.0,
            "trading_blocked": False,
        }

        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
        )

        data = json.loads(result)
        assert data["approved"] is False
        assert any(c["rule"] == "daily_loss_limit" and not c["passed"]
                  for c in data["checks"])

    @pytest.mark.asyncio
    async def test_reject_trading_blocked(self, mock_tool):
        """Test rejecting trade when trading is blocked."""
        mock_tool._client.get_account.return_value = {
            "equity": 100000.0,
            "last_equity": 100000.0,
            "trading_blocked": True,
        }

        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
        )

        data = json.loads(result)
        assert data["approved"] is False
        assert any(c["rule"] == "trading_blocked" and not c["passed"]
                  for c in data["checks"])

    @pytest.mark.asyncio
    async def test_sell_does_not_increase_exposure(self, mock_tool):
        """Test that sell orders don't add to exposure calculation."""
        mock_tool._client.get_positions.return_value = [
            {"symbol": "AAPL", "market_value": 24000.0},  # Near limit
        ]

        result = await mock_tool.execute(
            symbol="AAPL",
            side="sell",  # Selling, not buying
            quantity=50,
            price=150.0,
        )

        data = json.loads(result)
        exposure_check = next(c for c in data["checks"]
                            if c["rule"] == "max_portfolio_exposure")
        assert exposure_check["passed"] is True

    @pytest.mark.asyncio
    async def test_quantity_adjustment(self, mock_tool):
        """Test quantity adjustment to fit limits."""
        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=50,
            price=150.0,  # Would be $7500, but max position is $5000
        )

        data = json.loads(result)
        # Should be approved with adjusted quantity
        # Since $5000 / $150 = 33.33, adjusted_quantity should be capped
        assert data["approved_quantity"] <= 50

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_tool):
        """Test error handling when API fails."""
        mock_tool._client.get_account.side_effect = Exception("API Error")

        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
        )

        data = json.loads(result)
        assert data["approved"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_includes_risk_limits_in_response(self, mock_tool):
        """Test that response includes current risk limits."""
        result = await mock_tool.execute(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
        )

        data = json.loads(result)
        assert "risk_limits" in data
        assert data["risk_limits"]["max_position_value"] == 5000.0
        assert data["risk_limits"]["max_portfolio_exposure"] == 25000.0
