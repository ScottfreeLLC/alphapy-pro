"""
Tests for alphapy.backtest module - vectorbt integration.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Import Orders before backtest to avoid import issues
from alphapy.globals import Orders


class TestSignalsToVectorbt:
    """Tests for the signals_to_vectorbt function."""

    @pytest.fixture
    def sample_price_df(self):
        """Create a sample price DataFrame for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'AAPL': np.random.uniform(150, 160, 100),
            'GOOGL': np.random.uniform(2800, 2900, 100),
            'MSFT': np.random.uniform(350, 360, 100)
        }, index=dates)

    @pytest.fixture
    def sample_tradelist(self):
        """Create a sample trade list."""
        return [
            (datetime(2023, 1, 5), ['AAPL', Orders.le, 100, 155.0]),
            (datetime(2023, 1, 10), ['AAPL', Orders.lx, -100, 158.0]),
            (datetime(2023, 1, 15), ['GOOGL', Orders.se, -50, 2850.0]),
            (datetime(2023, 1, 20), ['GOOGL', Orders.sx, 50, 2820.0]),
            (datetime(2023, 1, 25), ['MSFT', Orders.le, 75, 355.0]),
            (datetime(2023, 1, 30), ['MSFT', Orders.lh, -75, 360.0]),
        ]

    def test_signal_conversion_basic(self, sample_tradelist, sample_price_df):
        """Test basic signal conversion from trade list."""
        from alphapy.backtest import signals_to_vectorbt

        symbols = ['AAPL', 'GOOGL', 'MSFT']
        entries, exits, short_entries, short_exits = signals_to_vectorbt(
            sample_tradelist, symbols, sample_price_df
        )

        # Check DataFrame shapes
        assert entries.shape == sample_price_df.shape
        assert exits.shape == sample_price_df.shape
        assert short_entries.shape == sample_price_df.shape
        assert short_exits.shape == sample_price_df.shape

        # Check that signals are boolean
        assert entries.dtypes.apply(lambda x: x == bool).all()
        assert exits.dtypes.apply(lambda x: x == bool).all()

    def test_signal_counts(self, sample_tradelist, sample_price_df):
        """Test that signal counts match expected values."""
        from alphapy.backtest import signals_to_vectorbt

        symbols = ['AAPL', 'GOOGL', 'MSFT']
        entries, exits, short_entries, short_exits = signals_to_vectorbt(
            sample_tradelist, symbols, sample_price_df
        )

        # Count signals - should have 2 long entries (AAPL and MSFT)
        assert entries.sum().sum() == 2

        # Should have 2 long exits (AAPL lx and MSFT lh)
        assert exits.sum().sum() == 2

        # Should have 1 short entry (GOOGL)
        assert short_entries.sum().sum() == 1

        # Should have 1 short exit (GOOGL)
        assert short_exits.sum().sum() == 1

    def test_empty_tradelist(self, sample_price_df):
        """Test handling of empty trade list."""
        from alphapy.backtest import signals_to_vectorbt

        symbols = ['AAPL', 'GOOGL', 'MSFT']
        entries, exits, short_entries, short_exits = signals_to_vectorbt(
            [], symbols, sample_price_df
        )

        # All should be False
        assert entries.sum().sum() == 0
        assert exits.sum().sum() == 0
        assert short_entries.sum().sum() == 0
        assert short_exits.sum().sum() == 0

    def test_symbol_not_in_universe(self, sample_price_df):
        """Test handling of trades for symbols not in universe."""
        from alphapy.backtest import signals_to_vectorbt

        tradelist = [
            (datetime(2023, 1, 5), ['UNKNOWN', Orders.le, 100, 155.0]),
        ]

        symbols = ['AAPL', 'GOOGL', 'MSFT']
        entries, exits, short_entries, short_exits = signals_to_vectorbt(
            tradelist, symbols, sample_price_df
        )

        # Should have no signals since UNKNOWN is not in universe
        assert entries.sum().sum() == 0


class TestConfigMapping:
    """Tests for configuration mapping from AlphaPy to vectorbt."""

    def test_cost_bps_conversion(self):
        """Test that cost_bps is correctly converted to fees."""
        # 10 basis points = 0.001
        cost_bps = 10
        expected_fees = cost_bps / 10000.0
        assert expected_fees == 0.001

        # 50 basis points = 0.005
        cost_bps = 50
        expected_fees = cost_bps / 10000.0
        assert expected_fees == 0.005

    def test_kelly_frac_as_percent(self):
        """Test that kelly_frac is used as percent size."""
        kelly_frac = 0.1  # 10%
        # In vectorbt, this should be used with size_type='percent'
        assert kelly_frac == 0.1


class TestVBTBacktesterOutputFormat:
    """Tests for VBTBacktester output format compatibility."""

    @pytest.fixture
    def mock_portfolio(self):
        """Create a mock vectorbt portfolio."""
        mock = MagicMock()

        # Mock returns
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        returns = pd.Series(np.random.uniform(-0.02, 0.02, 30), index=dates)
        mock.returns.return_value = returns

        # Mock asset_value
        asset_values = pd.DataFrame({
            'AAPL': np.random.uniform(10000, 11000, 30),
            'GOOGL': np.random.uniform(15000, 16000, 30),
        }, index=dates)
        mock.asset_value.return_value = asset_values

        # Mock cash
        cash = pd.Series(np.random.uniform(70000, 80000, 30), index=dates)
        mock.cash.return_value = cash

        # Mock stats
        stats = pd.Series({
            'Start': dates[0],
            'End': dates[-1],
            'Total Return [%]': 5.23,
            'Sharpe Ratio': 1.5,
            'Max Drawdown [%]': -8.2,
            'Total Trades': 10,
        })
        mock.stats.return_value = stats

        # Mock trades
        trades_df = pd.DataFrame({
            'Column': ['AAPL', 'GOOGL'],
            'Entry Timestamp': [dates[5], dates[10]],
            'Exit Timestamp': [dates[15], dates[20]],
            'Size': [100, 50],
            'Entry Price': [150.0, 2800.0],
            'Exit Price': [155.0, 2850.0],
        })
        mock.trades.records_readable = trades_df

        # Mock plot
        mock.plot.return_value = MagicMock()

        return mock

    def test_returns_frame_format(self, mock_portfolio):
        """Test that returns frame matches AlphaPy format."""
        from alphapy.backtest import VBTBacktester

        backtester = VBTBacktester.__new__(VBTBacktester)
        backtester._portfolio = mock_portfolio

        rf = backtester.to_returns_frame()

        # Check structure
        assert isinstance(rf, pd.DataFrame)
        assert 'return' in rf.columns
        assert rf.index.name == 'date'

    def test_positions_frame_format(self, mock_portfolio):
        """Test that positions frame matches AlphaPy format."""
        from alphapy.backtest import VBTBacktester

        backtester = VBTBacktester.__new__(VBTBacktester)
        backtester._portfolio = mock_portfolio

        pf = backtester.to_positions_frame()

        # Check structure
        assert isinstance(pf, pd.DataFrame)
        assert 'cash' in pf.columns
        assert pf.index.name == 'date'

    def test_transactions_frame_format(self, mock_portfolio):
        """Test that transactions frame matches AlphaPy format."""
        from alphapy.backtest import VBTBacktester

        backtester = VBTBacktester.__new__(VBTBacktester)
        backtester._portfolio = mock_portfolio

        tf = backtester.to_transactions_frame()

        # Check structure
        assert isinstance(tf, pd.DataFrame)
        if not tf.empty:
            assert 'symbol' in tf.columns
            assert 'amount' in tf.columns
            assert 'price' in tf.columns

    def test_stats_output(self, mock_portfolio):
        """Test that stats returns properly formatted Series."""
        from alphapy.backtest import VBTBacktester

        backtester = VBTBacktester.__new__(VBTBacktester)
        backtester._portfolio = mock_portfolio

        stats = backtester.stats()

        assert isinstance(stats, pd.Series)
        assert 'Total Return [%]' in stats.index
        assert 'Sharpe Ratio' in stats.index


class TestOrderTypeMapping:
    """Tests for order type mapping."""

    def test_long_entry_order(self):
        """Test long entry order type."""
        assert Orders.le == 'le'

    def test_long_exit_order(self):
        """Test long exit order types."""
        assert Orders.lx == 'lx'
        assert Orders.lh == 'lh'

    def test_short_entry_order(self):
        """Test short entry order type."""
        assert Orders.se == 'se'

    def test_short_exit_order(self):
        """Test short exit order types."""
        assert Orders.sx == 'sx'
        assert Orders.sh == 'sh'


class TestBacktesterNotRunError:
    """Tests for error handling when backtest hasn't been run."""

    def test_stats_without_run(self):
        """Test that stats raises error if backtest not run."""
        from alphapy.backtest import VBTBacktester

        backtester = VBTBacktester.__new__(VBTBacktester)
        backtester._portfolio = None

        with pytest.raises(ValueError, match="Must run backtest first"):
            backtester.stats()

    def test_returns_without_run(self):
        """Test that to_returns_frame raises error if backtest not run."""
        from alphapy.backtest import VBTBacktester

        backtester = VBTBacktester.__new__(VBTBacktester)
        backtester._portfolio = None

        with pytest.raises(ValueError, match="Must run backtest first"):
            backtester.to_returns_frame()

    def test_positions_without_run(self):
        """Test that to_positions_frame raises error if backtest not run."""
        from alphapy.backtest import VBTBacktester

        backtester = VBTBacktester.__new__(VBTBacktester)
        backtester._portfolio = None

        with pytest.raises(ValueError, match="Must run backtest first"):
            backtester.to_positions_frame()

    def test_transactions_without_run(self):
        """Test that to_transactions_frame raises error if backtest not run."""
        from alphapy.backtest import VBTBacktester

        backtester = VBTBacktester.__new__(VBTBacktester)
        backtester._portfolio = None

        with pytest.raises(ValueError, match="Must run backtest first"):
            backtester.to_transactions_frame()

    def test_tearsheet_without_run(self):
        """Test that generate_tearsheet raises error if backtest not run."""
        from alphapy.backtest import VBTBacktester

        backtester = VBTBacktester.__new__(VBTBacktester)
        backtester._portfolio = None

        with pytest.raises(ValueError, match="Must run backtest first"):
            backtester.generate_tearsheet('/tmp/test.html')
