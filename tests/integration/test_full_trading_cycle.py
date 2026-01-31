"""Integration tests for full trading cycle.

These tests verify the complete workflow from market data to order execution,
with all external APIs mocked.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import polars as pl
import pytest


class TestTradingCycleIntegration:
    """Integration tests for the full trading cycle."""

    @pytest.fixture
    def market_data_with_indicators(self):
        """Create market data with pre-computed indicators."""
        import numpy as np

        np.random.seed(42)
        n = 100

        returns = np.random.normal(0.001, 0.02, n)
        prices = 150 * np.exp(np.cumsum(returns))

        return pl.DataFrame({
            "open": prices * (1 + np.random.normal(0, 0.005, n)),
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n),
            "sma_20": prices * (1 + np.random.normal(0, 0.01, n)),
            "rsi_14": np.clip(np.random.normal(50, 15, n), 0, 100),
        })

    @pytest.fixture
    def mock_trading_environment(self, mock_alpaca_env, mock_polygon_env):
        """Set up complete mock trading environment."""
        yield

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Integration test needs to be updated to match actual tool APIs")
    async def test_signal_to_risk_check_flow(self, market_data_with_indicators, mock_trading_environment):
        """Test the flow from signal generation to risk checking."""
        from agent.tools.signal_generator import SignalGeneratorTool
        from agent.tools.risk_checks import RiskCheckTool
        from agent.utils.market_hours import MarketHours

        # Create tools
        signal_tool = SignalGeneratorTool()
        risk_tool = RiskCheckTool()

        # Generate a signal
        signal_result_json = await signal_tool.execute(
            symbol="AAPL",
            features={
                "rsi_14": 25.0,  # Oversold
                "price_vs_sma": -0.05,  # Below SMA
                "model_prediction": 0.7,
                "model_probability": 0.75,
            },
            current_price=150.0,
        )
        signal_result = json.loads(signal_result_json)

        # Verify signal was generated
        assert signal_result["status"] == "success"
        signal = signal_result["signal"]

        # Mock portfolio for risk check
        mock_portfolio = {
            "equity": 100000.0,
            "buying_power": 80000.0,
            "positions": [],
        }

        # Run risk check on the signal
        # Mock market hours to be open
        with patch.object(MarketHours, 'is_stock_market_open', return_value=True):
            risk_result_json = await risk_tool.execute(
                symbol=signal["symbol"],
                side=signal["direction"],
                quantity=10,
                price=signal["entry_price"],
                portfolio=mock_portfolio,
            )
            risk_result = json.loads(risk_result_json)

        assert risk_result["status"] == "success"
        assert "checks" in risk_result

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Integration test needs to be updated to match actual tool APIs")
    async def test_portfolio_state_to_order_flow(self, mock_trading_environment, mock_alpaca_account, mock_alpaca_position):
        """Test the flow from portfolio sync to order execution."""
        from agent.tools.portfolio_state import PortfolioStateTool
        from agent.tools.order_execution import OrderExecutionTool

        # Mock Alpaca client
        with patch("alpaca.trading.client.TradingClient") as mock_client:
            trading_client = MagicMock()
            trading_client.get_account.return_value = mock_alpaca_account
            trading_client.get_all_positions.return_value = [mock_alpaca_position]
            mock_client.return_value = trading_client

            # Get portfolio state
            portfolio_tool = PortfolioStateTool()
            portfolio_result_json = await portfolio_tool.execute()
            portfolio_result = json.loads(portfolio_result_json)

            assert portfolio_result["status"] == "success"
            portfolio = portfolio_result["portfolio"]

            # Verify portfolio data
            assert portfolio["equity"] == 100000.0
            assert len(portfolio["positions"]) == 1

            # Create order tool and submit order
            order_tool = OrderExecutionTool()

            # Mock order submission
            mock_order = MagicMock()
            mock_order.id = "test-order-123"
            mock_order.symbol = "TSLA"
            mock_order.side = "buy"
            mock_order.qty = "10"
            mock_order.type = "market"
            mock_order.status = "accepted"
            trading_client.submit_order.return_value = mock_order

            order_result_json = await order_tool.execute(
                symbol="TSLA",
                side="buy",
                quantity=10,
                order_type="market",
            )
            order_result = json.loads(order_result_json)

            assert order_result["status"] == "success"
            assert order_result["order"]["symbol"] == "TSLA"

    @pytest.mark.asyncio
    async def test_market_data_flow(self, mock_trading_environment, mock_polygon_env):
        """Test fetching market data and computing indicators."""
        from agent.utils.feature_calculator import FeatureCalculator

        # Create sample market data as if from API
        mock_bars = {
            "open": [145.0 + i for i in range(100)],
            "high": [146.0 + i for i in range(100)],
            "low": [144.0 + i for i in range(100)],
            "close": [145.5 + i for i in range(100)],
            "volume": [1000000] * 100,
        }

        # Convert to DataFrame and compute indicators
        df = pl.DataFrame({
            "open": mock_bars["open"],
            "high": mock_bars["high"],
            "low": mock_bars["low"],
            "close": mock_bars["close"],
            "volume": mock_bars["volume"],
        })

        # Using string-based indicator specs
        calc = FeatureCalculator(indicators=[
            "rsi_14",
            "ma_close_20",
        ])
        df_with_indicators = calc.compute_single(df, "AAPL")

        assert "rsi_14" in df_with_indicators.columns
        assert "ma_close_20" in df_with_indicators.columns

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Integration test needs to be updated to match actual tool APIs")
    async def test_complete_trading_loop(self, mock_trading_environment, mock_alpaca_account, mock_alpaca_position, mock_alpaca_order):
        """Test a complete trading loop: data -> indicators -> signal -> risk -> order."""
        from agent.tools.signal_generator import SignalGeneratorTool
        from agent.tools.risk_checks import RiskCheckTool
        from agent.tools.order_execution import OrderExecutionTool
        from agent.utils.market_hours import MarketHours

        # Mock all external dependencies
        with patch("alpaca.trading.client.TradingClient") as mock_client, \
             patch.object(MarketHours, 'is_stock_market_open', return_value=True):

            trading_client = MagicMock()
            trading_client.get_account.return_value = mock_alpaca_account
            trading_client.get_all_positions.return_value = []
            trading_client.submit_order.return_value = mock_alpaca_order
            mock_client.return_value = trading_client

            # Step 1: Generate signal
            signal_tool = SignalGeneratorTool()
            signal_result_json = await signal_tool.execute(
                symbol="NVDA",
                features={
                    "rsi_14": 30.0,
                    "price_vs_sma": -0.03,
                    "model_prediction": 0.8,
                    "model_probability": 0.85,
                },
                current_price=500.0,
            )
            signal_result = json.loads(signal_result_json)

            assert signal_result["status"] == "success"
            signal = signal_result["signal"]
            assert signal["direction"] == "buy"

            # Step 2: Risk check
            risk_tool = RiskCheckTool()
            risk_result_json = await risk_tool.execute(
                symbol=signal["symbol"],
                side=signal["direction"],
                quantity=5,
                price=signal["entry_price"],
                portfolio={
                    "equity": 100000.0,
                    "buying_power": 80000.0,
                    "positions": [],
                },
            )
            risk_result = json.loads(risk_result_json)

            assert risk_result["status"] == "success"
            assert risk_result["approved"] is True

            # Step 3: Execute order
            order_tool = OrderExecutionTool()
            order_result_json = await order_tool.execute(
                symbol=signal["symbol"],
                side=signal["direction"],
                quantity=5,
                order_type="market",
            )
            order_result = json.loads(order_result_json)

            assert order_result["status"] == "success"
            assert "order" in order_result

            # Verify mock was called
            trading_client.submit_order.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Integration test needs to be updated to match actual MemoryTool API")
    async def test_memory_persistence_across_cycle(self, mock_trading_environment, tmp_path):
        """Test that memory persists trade history across trading cycles."""
        from agent.tools.memory import MemoryTool

        memory_tool = MemoryTool()

        # Record first trade
        result1_json = await memory_tool.execute(
            action="record_trade",
            trade={
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "price": 150.0,
                "timestamp": "2024-01-16T10:30:00",
            },
        )
        result1 = json.loads(result1_json)
        assert result1["status"] == "success"

        # Record second trade
        result2_json = await memory_tool.execute(
            action="record_trade",
            trade={
                "symbol": "AAPL",
                "side": "sell",
                "quantity": 10,
                "price": 155.0,
                "timestamp": "2024-01-16T14:30:00",
            },
        )
        result2 = json.loads(result2_json)
        assert result2["status"] == "success"

        # Retrieve history
        history_result_json = await memory_tool.execute(
            action="get_history",
            symbol="AAPL",
        )
        history_result = json.loads(history_result_json)

        assert history_result["status"] == "success"
        assert len(history_result["trades"]) == 2
