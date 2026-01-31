"""Tests for alphapy.portfolio.live module."""

from datetime import datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from alphapy.portfolio.live import LivePosition, LivePortfolio


class TestLivePosition:
    """Tests for LivePosition dataclass."""

    def test_create_position(self):
        """Test creating a position directly."""
        pos = LivePosition(
            symbol="AAPL",
            qty=100,
            side="long",
            avg_entry_price=145.0,
            current_price=150.0,
            market_value=15000.0,
            cost_basis=14500.0,
            unrealized_pl=500.0,
            unrealized_plpc=0.0345,
        )

        assert pos.symbol == "AAPL"
        assert pos.qty == 100
        assert pos.side == "long"
        assert pos.avg_entry_price == 145.0
        assert pos.current_price == 150.0
        assert pos.market_value == 15000.0
        assert pos.unrealized_pl == 500.0

    def test_from_alpaca_dict(self):
        """Test creating position from Alpaca dictionary."""
        alpaca_dict = {
            "symbol": "TSLA",
            "qty": 50,
            "side": "long",
            "avg_entry_price": 200.0,
            "current_price": 210.0,
            "market_value": 10500.0,
            "cost_basis": 10000.0,
            "unrealized_pl": 500.0,
            "unrealized_plpc": 0.05,
            "change_today": 0.02,
        }

        pos = LivePosition.from_alpaca(alpaca_dict)

        assert pos.symbol == "TSLA"
        assert pos.qty == 50
        assert pos.side == "long"
        assert pos.change_today == 0.02

    def test_from_alpaca_missing_change_today(self):
        """Test from_alpaca when change_today is missing."""
        alpaca_dict = {
            "symbol": "AAPL",
            "qty": 100,
            "side": "long",
            "avg_entry_price": 145.0,
            "current_price": 150.0,
            "market_value": 15000.0,
            "cost_basis": 14500.0,
            "unrealized_pl": 500.0,
            "unrealized_plpc": 0.0345,
        }

        pos = LivePosition.from_alpaca(alpaca_dict)

        assert pos.change_today == 0.0

    def test_to_dict(self):
        """Test converting position to dictionary."""
        pos = LivePosition(
            symbol="AAPL",
            qty=100,
            side="long",
            avg_entry_price=145.0,
            current_price=150.0,
            market_value=15000.0,
            cost_basis=14500.0,
            unrealized_pl=500.0,
            unrealized_plpc=0.0345,
            change_today=0.01,
        )

        d = pos.to_dict()

        assert d["symbol"] == "AAPL"
        assert d["qty"] == 100
        assert d["unrealized_pl"] == 500.0
        assert "last_updated" in d

    def test_last_updated_default(self):
        """Test that last_updated is set by default."""
        pos = LivePosition(
            symbol="AAPL",
            qty=100,
            side="long",
            avg_entry_price=145.0,
            current_price=150.0,
            market_value=15000.0,
            cost_basis=14500.0,
            unrealized_pl=500.0,
            unrealized_plpc=0.0345,
        )

        assert pos.last_updated is not None
        assert isinstance(pos.last_updated, datetime)


class TestLivePortfolio:
    """Tests for LivePortfolio class."""

    def test_init_without_broker(self):
        """Test initialization without broker."""
        portfolio = LivePortfolio()

        assert portfolio._broker is None
        assert portfolio.positions == {}
        assert portfolio.cash == 0.0
        assert portfolio.equity == 0.0
        assert portfolio.unrealized_pl == 0.0

    def test_init_with_broker(self):
        """Test initialization with broker."""
        mock_broker = MagicMock()
        portfolio = LivePortfolio(broker=mock_broker)

        assert portfolio._broker is mock_broker

    def test_set_broker(self):
        """Test setting broker after init."""
        portfolio = LivePortfolio()
        mock_broker = MagicMock()

        portfolio.set_broker(mock_broker)

        assert portfolio._broker is mock_broker

    def test_sync_without_broker_raises(self):
        """Test sync raises error without broker."""
        portfolio = LivePortfolio()

        with pytest.raises(RuntimeError, match="Broker not set"):
            portfolio.sync()

    def test_sync_with_broker(self):
        """Test syncing portfolio state."""
        mock_broker = MagicMock()
        mock_broker.get_account.return_value = {
            "cash": 50000.0,
            "equity": 100000.0,
            "buying_power": 80000.0,
            "portfolio_value": 100000.0,
            "long_market_value": 45000.0,
            "short_market_value": 0.0,
        }
        mock_broker.get_positions.return_value = [
            {
                "symbol": "AAPL",
                "qty": 100,
                "side": "long",
                "avg_entry_price": 145.0,
                "current_price": 150.0,
                "market_value": 15000.0,
                "cost_basis": 14500.0,
                "unrealized_pl": 500.0,
                "unrealized_plpc": 0.0345,
            }
        ]

        portfolio = LivePortfolio(broker=mock_broker)
        portfolio.sync()

        assert portfolio.cash == 50000.0
        assert portfolio.equity == 100000.0
        assert portfolio.buying_power == 80000.0
        assert len(portfolio.positions) == 1
        assert "AAPL" in portfolio.positions

    def test_sync_tracks_starting_equity(self):
        """Test that starting equity is tracked on first sync."""
        mock_broker = MagicMock()
        mock_broker.get_account.return_value = {
            "cash": 50000.0,
            "equity": 100000.0,
            "buying_power": 80000.0,
            "portfolio_value": 100000.0,
            "long_market_value": 45000.0,
            "short_market_value": 0.0,
        }
        mock_broker.get_positions.return_value = []

        portfolio = LivePortfolio(broker=mock_broker)
        portfolio.sync()

        assert portfolio._starting_equity == 100000.0

    def test_get_position_exists(self):
        """Test getting an existing position."""
        portfolio = LivePortfolio()
        portfolio.positions["AAPL"] = LivePosition(
            symbol="AAPL", qty=100, side="long",
            avg_entry_price=145.0, current_price=150.0,
            market_value=15000.0, cost_basis=14500.0,
            unrealized_pl=500.0, unrealized_plpc=0.0345,
        )

        pos = portfolio.get_position("AAPL")

        assert pos is not None
        assert pos.symbol == "AAPL"

    def test_get_position_case_insensitive(self):
        """Test getting position is case insensitive."""
        portfolio = LivePortfolio()
        portfolio.positions["AAPL"] = LivePosition(
            symbol="AAPL", qty=100, side="long",
            avg_entry_price=145.0, current_price=150.0,
            market_value=15000.0, cost_basis=14500.0,
            unrealized_pl=500.0, unrealized_plpc=0.0345,
        )

        pos = portfolio.get_position("aapl")

        assert pos is not None
        assert pos.symbol == "AAPL"

    def test_get_position_not_exists(self):
        """Test getting a non-existent position."""
        portfolio = LivePortfolio()

        pos = portfolio.get_position("AAPL")

        assert pos is None

    def test_has_position_true(self):
        """Test has_position returns True."""
        portfolio = LivePortfolio()
        portfolio.positions["AAPL"] = MagicMock()

        assert portfolio.has_position("AAPL") is True
        assert portfolio.has_position("aapl") is True

    def test_has_position_false(self):
        """Test has_position returns False."""
        portfolio = LivePortfolio()

        assert portfolio.has_position("AAPL") is False

    def test_get_quantity_with_position(self):
        """Test get_quantity with existing position."""
        portfolio = LivePortfolio()
        portfolio.positions["AAPL"] = LivePosition(
            symbol="AAPL", qty=100, side="long",
            avg_entry_price=145.0, current_price=150.0,
            market_value=15000.0, cost_basis=14500.0,
            unrealized_pl=500.0, unrealized_plpc=0.0345,
        )

        assert portfolio.get_quantity("AAPL") == 100

    def test_get_quantity_no_position(self):
        """Test get_quantity with no position."""
        portfolio = LivePortfolio()

        assert portfolio.get_quantity("AAPL") == 0.0

    def test_positions_df_empty(self):
        """Test positions_df with no positions."""
        portfolio = LivePortfolio()

        df = portfolio.positions_df()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "symbol" in df.columns

    def test_positions_df_with_positions(self):
        """Test positions_df with positions."""
        portfolio = LivePortfolio()
        portfolio.positions["AAPL"] = LivePosition(
            symbol="AAPL", qty=100, side="long",
            avg_entry_price=145.0, current_price=150.0,
            market_value=15000.0, cost_basis=14500.0,
            unrealized_pl=500.0, unrealized_plpc=0.0345,
        )

        df = portfolio.positions_df()

        assert len(df) == 1
        assert df["symbol"][0] == "AAPL"
        assert "last_updated" not in df.columns

    def test_history_df_empty(self):
        """Test history_df with no history."""
        portfolio = LivePortfolio()

        df = portfolio.history_df()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "equity" in df.columns

    def test_total_return_no_starting_equity(self):
        """Test total_return when starting equity not set."""
        portfolio = LivePortfolio()

        assert portfolio.total_return() == 0.0

    def test_total_return_calculation(self):
        """Test total_return calculation."""
        portfolio = LivePortfolio()
        portfolio._starting_equity = 100000.0
        portfolio.equity = 110000.0

        assert portfolio.total_return() == 0.1  # 10%

    def test_total_return_pct(self):
        """Test total_return_pct calculation."""
        portfolio = LivePortfolio()
        portfolio._starting_equity = 100000.0
        portfolio.equity = 110000.0

        assert portfolio.total_return_pct() == 10.0

    def test_summary(self):
        """Test summary method."""
        portfolio = LivePortfolio()
        portfolio.equity = 100000.0
        portfolio.cash = 50000.0
        portfolio.buying_power = 80000.0
        portfolio.portfolio_value = 100000.0
        portfolio.unrealized_pl = 500.0
        portfolio.realized_pl = 200.0

        summary = portfolio.summary()

        assert summary["equity"] == 100000.0
        assert summary["cash"] == 50000.0
        assert summary["unrealized_pl"] == 500.0
        assert "num_positions" in summary

    def test_can_buy_sufficient_power(self):
        """Test can_buy with sufficient buying power."""
        portfolio = LivePortfolio()
        portfolio.buying_power = 10000.0

        assert portfolio.can_buy("AAPL", qty=50, price=150.0) is True

    def test_can_buy_insufficient_power(self):
        """Test can_buy with insufficient buying power."""
        portfolio = LivePortfolio()
        portfolio.buying_power = 5000.0

        assert portfolio.can_buy("AAPL", qty=50, price=150.0) is False

    def test_position_size_calculation(self):
        """Test position size calculation."""
        portfolio = LivePortfolio()
        portfolio.equity = 100000.0
        portfolio.buying_power = 50000.0

        # 2% risk, 5% stop loss at $100 price
        # Risk amount = 100000 * 0.02 = 2000
        # Risk per share = 100 * 0.05 = 5
        # Shares = 2000 / 5 = 400
        size = portfolio.position_size(price=100.0, risk_pct=0.02, stop_loss_pct=0.05)

        assert size == 400

    def test_position_size_limited_by_buying_power(self):
        """Test position size limited by buying power."""
        portfolio = LivePortfolio()
        portfolio.equity = 100000.0
        portfolio.buying_power = 1000.0  # Very low

        size = portfolio.position_size(price=100.0, risk_pct=0.02, stop_loss_pct=0.05)

        # Max from buying power: 1000 / 100 = 10
        assert size == 10

    def test_position_size_invalid_price(self):
        """Test position size with invalid price."""
        portfolio = LivePortfolio()

        assert portfolio.position_size(price=0.0) == 0
        assert portfolio.position_size(price=-100.0) == 0

    def test_position_size_invalid_stop_loss(self):
        """Test position size with invalid stop loss."""
        portfolio = LivePortfolio()

        assert portfolio.position_size(price=100.0, stop_loss_pct=0.0) == 0

    def test_record_realized_pl(self):
        """Test recording realized P&L."""
        portfolio = LivePortfolio()
        portfolio.realized_pl = 0.0

        portfolio.record_realized_pl(500.0)
        assert portfolio.realized_pl == 500.0

        portfolio.record_realized_pl(-200.0)
        assert portfolio.realized_pl == 300.0

    def test_exposure_calculation(self):
        """Test exposure calculation."""
        portfolio = LivePortfolio()
        portfolio.equity = 100000.0
        portfolio.long_market_value = 60000.0
        portfolio.short_market_value = 10000.0

        exposure = portfolio.exposure()

        assert exposure["long_exposure"] == 0.6
        assert exposure["short_exposure"] == 0.1
        assert exposure["net_exposure"] == 0.5
        assert exposure["gross_exposure"] == 0.7

    def test_exposure_zero_equity(self):
        """Test exposure with zero equity."""
        portfolio = LivePortfolio()
        portfolio.equity = 0.0

        exposure = portfolio.exposure()

        assert exposure["long_exposure"] == 0
        assert exposure["short_exposure"] == 0

    def test_repr(self):
        """Test string representation."""
        portfolio = LivePortfolio()
        portfolio.equity = 100000.0
        portfolio._starting_equity = 100000.0

        repr_str = repr(portfolio)

        assert "LivePortfolio" in repr_str
        assert "100000" in repr_str
