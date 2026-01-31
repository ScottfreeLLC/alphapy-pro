"""
Pytest configuration and shared fixtures for AlphaPy tests.
"""
import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import polars as pl
import pytest


# ============================================================================
# Async Support
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data_dir():
    """Path to sample data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_config_dir():
    """Path to sample config directory."""
    return Path(__file__).parent / "config"


@pytest.fixture
def memory_temp_dir(tmp_path):
    """Create temporary directory for memory tool tests."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV Polars DataFrame for testing."""
    n_rows = 100
    base_price = 150.0
    dates = [datetime.now() - timedelta(minutes=5 * i) for i in range(n_rows)][::-1]

    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, n_rows)
    prices = base_price * np.cumprod(1 + returns)

    return pl.DataFrame({
        "datetime": dates,
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_rows)),
        "high": prices * (1 + np.random.uniform(0, 0.02, n_rows)),
        "low": prices * (1 - np.random.uniform(0, 0.02, n_rows)),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n_rows),
    })


@pytest.fixture
def sample_ohlcv_pandas_df(sample_ohlcv_df):
    """Convert Polars OHLCV to Pandas for model tests."""
    return sample_ohlcv_df.to_pandas()


@pytest.fixture
def sample_bars_dict(sample_ohlcv_df):
    """Create bar data dict for market data tool tests."""
    return {
        "AAPL": {
            "bars": sample_ohlcv_df.to_dicts(),
            "count": len(sample_ohlcv_df),
            "latest_close": float(sample_ohlcv_df["close"][-1]),
            "latest_time": str(sample_ohlcv_df["datetime"][-1]),
        },
        "_summary": {
            "symbols_requested": 1,
            "symbols_returned": 1,
            "timeframe": "5Min",
            "lookback_bars": 100,
        }
    }


@pytest.fixture
def empty_ohlcv_df():
    """Create an empty OHLCV DataFrame for edge case testing."""
    return pl.DataFrame({
        "datetime": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }).cast({
        "datetime": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
    })


@pytest.fixture
def small_ohlcv_df():
    """Create a small OHLCV DataFrame (5 rows) for testing indicator warm-up."""
    return pl.DataFrame({
        "datetime": [datetime.now() - timedelta(minutes=5 * i) for i in range(5)][::-1],
        "open": [100.0, 101.0, 102.0, 101.5, 103.0],
        "high": [101.5, 102.5, 103.0, 102.5, 104.0],
        "low": [99.5, 100.5, 101.0, 100.5, 102.0],
        "close": [101.0, 102.0, 101.5, 102.5, 103.5],
        "volume": [100000, 120000, 110000, 130000, 140000],
    })


# ============================================================================
# Mock External Clients
# ============================================================================

@pytest.fixture
def mock_polygon_agg():
    """Create a mock Polygon aggregate bar."""
    mock_agg = MagicMock()
    mock_agg.timestamp = int(datetime.now().timestamp() * 1000)
    mock_agg.open = 150.0
    mock_agg.high = 152.0
    mock_agg.low = 149.0
    mock_agg.close = 151.0
    mock_agg.volume = 500000
    mock_agg.vwap = 150.5
    mock_agg.transactions = 1000
    return mock_agg


@pytest.fixture
def mock_polygon_client(mock_polygon_agg):
    """Mock Polygon REST client."""
    with patch("alphapy.data_sources.polygon.RESTClient") as mock:
        client = MagicMock()

        # Mock list_aggs - return 100 bars
        client.list_aggs.return_value = [mock_polygon_agg] * 100

        # Mock list_trades
        client.list_trades.return_value = []

        # Mock get_snapshot_ticker
        client.get_snapshot_ticker.return_value = None

        mock.return_value = client
        yield client


@pytest.fixture
def mock_alpaca_account():
    """Create a mock Alpaca account."""
    account = MagicMock()
    account.equity = 100000.0
    account.buying_power = 80000.0
    account.cash = 50000.0
    account.portfolio_value = 100000.0
    account.last_equity = 99500.0
    account.long_market_value = 45000.0
    account.short_market_value = 0.0
    account.initial_margin = 0.0
    account.maintenance_margin = 0.0
    account.daytrade_count = 0
    account.pattern_day_trader = False
    account.trading_blocked = False
    account.account_blocked = False
    return account


@pytest.fixture
def mock_alpaca_position():
    """Create a mock Alpaca position."""
    position = MagicMock()
    position.symbol = "AAPL"
    position.qty = 100
    position.side.value = "long"
    position.market_value = 15000.0
    position.cost_basis = 14500.0
    position.unrealized_pl = 500.0
    position.unrealized_plpc = 0.0345
    position.current_price = 150.0
    position.avg_entry_price = 145.0
    position.change_today = 0.02
    return position


@pytest.fixture
def mock_alpaca_order():
    """Create a mock Alpaca order."""
    order = MagicMock()
    order.id = "test-order-123"
    order.client_order_id = "client-123"
    order.symbol = "AAPL"
    order.side.value = "buy"
    order.qty = 10
    order.filled_qty = 0
    order.type.value = "market"
    order.status.value = "accepted"
    order.limit_price = None
    order.stop_price = None
    order.filled_avg_price = None
    order.created_at = datetime.now()
    return order


@pytest.fixture
def mock_alpaca_client(mock_alpaca_account, mock_alpaca_order):
    """Mock Alpaca Trading Client."""
    with patch("agent.utils.alpaca_client.TradingClient") as mock:
        client = MagicMock()

        # Mock account
        client.get_account.return_value = mock_alpaca_account

        # Mock positions (empty by default)
        client.get_all_positions.return_value = []
        client.get_open_position.side_effect = Exception("Position not found")

        # Mock order submission
        client.submit_order.return_value = mock_alpaca_order

        # Mock cancel order
        client.cancel_order_by_id.return_value = None

        # Mock get orders
        client.get_orders.return_value = []

        # Mock close position
        client.close_position.return_value = mock_alpaca_order

        # Mock close all positions
        client.close_all_positions.return_value = []

        mock.return_value = client
        yield client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic Claude API client."""
    with patch("anthropic.Anthropic") as mock:
        client = MagicMock()

        # Mock messages.create response
        response = MagicMock()
        response.content = [MagicMock(type="text", text="Test response from Claude")]
        response.stop_reason = "end_turn"
        client.messages.create.return_value = response

        mock.return_value = client
        yield client


# ============================================================================
# Mock Model Fixtures
# ============================================================================

@pytest.fixture
def mock_xgb_model(tmp_path):
    """Create a mock XGBoost model for testing."""
    from sklearn.ensemble import GradientBoostingClassifier
    import joblib

    # Create a simple model
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)

    # Create run directory structure
    model_dir = tmp_path / "model"
    config_dir = tmp_path / "config"
    model_dir.mkdir()
    config_dir.mkdir()

    # Save model
    joblib.dump(model, model_dir / "xgb_predictor.pkl")

    # Save feature map
    feature_names = [f"feature_{i}" for i in range(10)]
    feature_map = {name: i for i, name in enumerate(feature_names)}
    joblib.dump(feature_map, model_dir / "feature_map.pkl")

    # Save config
    config = {"target": "target", "algorithms": ["xgb"]}
    import yaml
    with open(config_dir / "model.yml", "w") as f:
        yaml.dump(config, f)

    return {
        "run_dir": tmp_path,
        "model": model,
        "feature_names": feature_names,
    }


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_alpaca_env():
    """Set mock Alpaca environment variables."""
    with patch.dict(os.environ, {
        "ALPACA_API_KEY": "test-api-key",
        "ALPACA_API_SECRET": "test-api-secret",
        "ALPACA_PAPER": "true",
    }):
        yield


@pytest.fixture
def mock_polygon_env():
    """Set mock Polygon environment variable."""
    with patch.dict(os.environ, {
        "POLYGON_API_KEY": "test-polygon-key",
    }):
        yield


@pytest.fixture
def mock_anthropic_env():
    """Set mock Anthropic environment variable."""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-anthropic-key",
    }):
        yield


@pytest.fixture
def mock_all_env(mock_alpaca_env, mock_polygon_env, mock_anthropic_env):
    """Set all mock environment variables."""
    yield


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_alphapy_config(temp_dir):
    """Create a mock AlphaPy configuration for testing."""
    config = {
        'directory': temp_dir,
        'file_extension': 'csv',
        'separator': ',',
        'target': 'target',
        'algorithms': ['rf', 'xgb'],
        'cv_folds': 3,
        'lag_period': 1,
        'leaders': 1,
        'predict_mode': False,
        'predict_history': False,
        'score_validation': False,
        'split': 0.4,
        'test_size': 0.2,
        'validation_size': 0.2,
    }
    return config


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "temperature": 0.7,
        "symbols_stocks": ["AAPL", "TSLA"],
        "symbols_crypto": ["BTC/USD"],
        "timeframe": "5Min",
        "bar_lookback": 100,
        "run_dir": "projects/test/runs/latest",
        "algo": "xgb",
        "prob_min": 0.55,
        "max_position_value": 5000.0,
        "max_portfolio_exposure": 25000.0,
        "max_positions": 5,
        "daily_loss_limit": 0.02,
        "loop_interval_seconds": 300,
    }


@pytest.fixture
def sample_risk_config():
    """Sample risk configuration."""
    return {
        "max_position_value": 5000.0,
        "max_portfolio_exposure": 25000.0,
        "max_positions": 5,
        "max_symbol_pct": 0.25,
        "daily_loss_limit": 0.02,
        "position_stop_loss": 0.01,
        "min_order_value": 100.0,
    }


# ============================================================================
# Indicator Test Fixtures
# ============================================================================

@pytest.fixture
def indicator_test_data():
    """Create test data specifically for indicator validation."""
    # Use deterministic data for indicator tests
    n = 200
    np.random.seed(42)

    # Trending data with noise
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 2, n)
    prices = trend + noise

    # Add some volatility spikes
    prices[50:60] += np.random.normal(0, 5, 10)
    prices[150:160] -= np.random.normal(0, 5, 10)

    return pl.DataFrame({
        "datetime": [datetime.now() - timedelta(minutes=5 * i) for i in range(n)][::-1],
        "open": prices * (1 + np.random.uniform(-0.002, 0.002, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n),
    })


# ============================================================================
# Portfolio Test Fixtures
# ============================================================================

@pytest.fixture
def sample_positions_dict():
    """Sample positions dictionary for portfolio tests."""
    return [
        {
            "symbol": "AAPL",
            "qty": 100.0,
            "side": "long",
            "avg_entry_price": 145.0,
            "current_price": 150.0,
            "market_value": 15000.0,
            "cost_basis": 14500.0,
            "unrealized_pl": 500.0,
            "unrealized_plpc": 0.0345,
            "change_today": 0.02,
        },
        {
            "symbol": "TSLA",
            "qty": 50.0,
            "side": "long",
            "avg_entry_price": 200.0,
            "current_price": 210.0,
            "market_value": 10500.0,
            "cost_basis": 10000.0,
            "unrealized_pl": 500.0,
            "unrealized_plpc": 0.05,
            "change_today": -0.01,
        },
    ]


# ============================================================================
# Trading Test Fixtures
# ============================================================================

@pytest.fixture
def sample_signals():
    """Sample trading signals for testing."""
    return {
        "AAPL": {
            "prediction": 1,
            "probability": 0.72,
            "signal": "long",
            "latest_close": 150.0,
            "latest_time": str(datetime.now()),
        },
        "TSLA": {
            "prediction": 0,
            "probability": 0.45,
            "signal": "none",
            "latest_close": 210.0,
            "latest_time": str(datetime.now()),
        },
    }


@pytest.fixture
def sample_order_params():
    """Sample order parameters for testing."""
    return {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10,
        "order_type": "market",
        "time_in_force": "day",
    }
