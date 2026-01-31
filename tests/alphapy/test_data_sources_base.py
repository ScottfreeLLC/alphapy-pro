"""Tests for alphapy.data_sources.base module."""

import pytest
from alphapy.data_sources.base import DataSource, DataSourceConfig


class TestDataSourceConfig:
    """Tests for DataSourceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataSourceConfig()

        assert config.api_key is None
        assert config.base_url is None
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 300
        assert config.requests_per_minute == 100
        assert config.extra == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DataSourceConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            timeout=60,
            max_retries=5,
            cache_enabled=False,
            cache_ttl_seconds=600,
            requests_per_minute=50,
            extra={"custom": "value"},
        )

        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.cache_enabled is False
        assert config.cache_ttl_seconds == 600
        assert config.requests_per_minute == 50
        assert config.extra == {"custom": "value"}

    def test_extra_is_mutable_default(self):
        """Test that each config gets its own extra dict."""
        config1 = DataSourceConfig()
        config2 = DataSourceConfig()

        config1.extra["key1"] = "value1"

        # Verify separate instances
        assert "key1" not in config2.extra


class TestDataSourceBase:
    """Tests for DataSource abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that DataSource cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataSource()

    def test_concrete_implementation(self):
        """Test a minimal concrete implementation."""

        class ConcreteDataSource(DataSource):
            def get_bars(self, symbols, timeframe="1Day", lookback=100,
                        start_date=None, end_date=None):
                return {}

            def get_quote(self, symbol):
                return {"symbol": symbol, "price": 100.0}

            def get_snapshot(self, symbols):
                return {}

        # Should be instantiable
        ds = ConcreteDataSource()
        assert ds is not None

    def test_name_property(self):
        """Test the name property extracts class name."""

        class TestDataSource(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        ds = TestDataSource()
        assert ds.name == "test"

    def test_name_property_without_datasource_suffix(self):
        """Test name property when class doesn't end in DataSource."""

        class MyProvider(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        ds = MyProvider()
        assert ds.name == "myprovider"

    def test_config_default(self):
        """Test that default config is created if not provided."""

        class TestDS(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        ds = TestDS()
        assert ds.config is not None
        assert isinstance(ds.config, DataSourceConfig)
        assert ds.config.timeout == 30  # default value

    def test_config_custom(self):
        """Test that custom config is used."""

        class TestDS(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        config = DataSourceConfig(timeout=120)
        ds = TestDS(config=config)

        assert ds.config.timeout == 120

    def test_is_crypto_with_slash(self):
        """Test is_crypto returns True for crypto symbols."""

        class TestDS(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        ds = TestDS()
        assert ds.is_crypto("BTC/USD") is True
        assert ds.is_crypto("ETH/USDT") is True

    def test_is_crypto_without_slash(self):
        """Test is_crypto returns False for stock symbols."""

        class TestDS(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        ds = TestDS()
        assert ds.is_crypto("AAPL") is False
        assert ds.is_crypto("TSLA") is False

    def test_normalize_symbol_uppercase(self):
        """Test normalize_symbol converts to uppercase."""

        class TestDS(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        ds = TestDS()
        assert ds.normalize_symbol("aapl") == "AAPL"
        assert ds.normalize_symbol("tsla") == "TSLA"
        assert ds.normalize_symbol("MSFT") == "MSFT"

    def test_ensure_list_with_string(self):
        """Test _ensure_list converts string to list."""

        class TestDS(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        ds = TestDS()
        result = ds._ensure_list("AAPL")

        assert isinstance(result, list)
        assert result == ["AAPL"]

    def test_ensure_list_with_list(self):
        """Test _ensure_list keeps list as list."""

        class TestDS(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        ds = TestDS()
        result = ds._ensure_list(["AAPL", "TSLA"])

        assert isinstance(result, list)
        assert result == ["AAPL", "TSLA"]

    def test_ensure_list_with_tuple(self):
        """Test _ensure_list converts tuple to list."""

        class TestDS(DataSource):
            def get_bars(self, symbols, **kwargs):
                return {}

            def get_quote(self, symbol):
                return {}

            def get_snapshot(self, symbols):
                return {}

        ds = TestDS()
        result = ds._ensure_list(("AAPL", "TSLA"))

        assert isinstance(result, list)
        assert result == ["AAPL", "TSLA"]
