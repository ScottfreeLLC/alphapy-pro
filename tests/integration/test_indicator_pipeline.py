"""Integration tests for the indicator pipeline.

These tests verify the full flow from raw data to computed indicators.
Uses string-based API for all indicator definitions.
"""

import polars as pl
import pytest

from alphapy.indicators import IndicatorEngine, add_indicators


class TestIndicatorPipelineIntegration:
    """Integration tests for indicator computation pipeline."""

    @pytest.fixture
    def realistic_market_data(self):
        """Create realistic market data with proper OHLCV relationships."""
        import numpy as np

        np.random.seed(42)
        n = 252  # One trading year

        # Generate realistic price series using geometric random walk
        returns = np.random.normal(0.0005, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
        open_prices = prices * (1 + np.random.normal(0, 0.005, n))

        # Ensure OHLC relationships are valid
        high = np.maximum(high, np.maximum(open_prices, prices))
        low = np.minimum(low, np.minimum(open_prices, prices))

        # Generate volume
        volume = np.random.randint(100000, 10000000, n)

        return pl.DataFrame({
            "open": open_prices,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        })

    def test_full_indicator_suite_computation(self, realistic_market_data):
        """Test computing a full suite of indicators end-to-end."""
        engine = IndicatorEngine()

        # All indicators in one call - auto-dependency resolution handles atr_14
        indicators = [
            # Trend indicators (ma, ema from transforms.py)
            "ma_close_20",
            "ema_close_12",
            "ema_close_26",

            # Momentum indicators
            "rsi_14",

            # Volatility - atr_14 auto-resolves truerange dependency
            "atr_14",
        ]
        result = engine.compute(realistic_market_data, indicators)

        # Verify all indicators were added
        assert "ma_close_20" in result.columns
        assert "ema_close_12" in result.columns
        assert "ema_close_26" in result.columns
        assert "rsi_14" in result.columns
        assert "truerange" in result.columns  # Auto-computed dependency
        assert "atr_14" in result.columns

        # Verify data integrity
        assert len(result) == len(realistic_market_data)

        # Verify indicator values make sense
        import numpy as np
        rsi_values = [v for v in result["rsi_14"].to_list()
                      if v is not None and not (isinstance(v, float) and np.isnan(v))]
        assert all(0 <= v <= 100 for v in rsi_values), "RSI should be between 0-100"

    def test_string_shorthand_pipeline(self, realistic_market_data):
        """Test using string shorthand for indicator definitions."""
        # All indicators in one call - auto-dependency resolution handles atr_14
        result = add_indicators(
            realistic_market_data,
            ["ma_close_20", "ema_close_12", "rsi_14", "atr_14"]
        )

        assert "ma_close_20" in result.columns
        assert "ema_close_12" in result.columns
        assert "rsi_14" in result.columns
        assert "truerange" in result.columns  # Auto-computed dependency
        assert "atr_14" in result.columns

    def test_indicator_values_converge(self, realistic_market_data):
        """Test that indicator values stabilize after warmup period."""
        import numpy as np

        engine = IndicatorEngine()
        result = engine.compute(realistic_market_data, ["ma_close_20", "rsi_14"])

        # After warmup, MA should have non-null values
        ma_values = result["ma_close_20"].to_list()
        null_count = sum(1 for v in ma_values if v is None or (isinstance(v, float) and np.isnan(v)))

        # MA-20 should have at most 19 null/NaN values
        assert null_count <= 19, f"Expected <= 19 null values, got {null_count}"

        # Non-null values should be within reasonable range of close prices
        close_prices = result["close"].to_list()
        for i, ma in enumerate(ma_values):
            if ma is not None and not (isinstance(ma, float) and np.isnan(ma)):
                assert abs(ma - close_prices[i]) / close_prices[i] < 0.2

    def test_multi_symbol_pipeline(self, realistic_market_data):
        """Test computing indicators for multiple symbols."""
        from alphapy.indicators import IndicatorEngine

        # Create data for multiple symbols
        symbols_data = {
            "AAPL": realistic_market_data,
            "GOOGL": realistic_market_data.clone(),
            "MSFT": realistic_market_data.clone(),
        }

        indicators = ["ma_close_20", "rsi_14"]

        engine = IndicatorEngine()

        results = {}
        for symbol, df in symbols_data.items():
            results[symbol] = engine.compute(df, indicators)

        # Verify all symbols processed
        assert len(results) == 3
        for symbol, result in results.items():
            assert "ma_close_20" in result.columns
            assert "rsi_14" in result.columns

    def test_indicator_chaining(self, realistic_market_data):
        """Test computing indicators in sequence (chaining)."""
        engine = IndicatorEngine()

        # First pass: trend indicators
        result1 = engine.compute(realistic_market_data, ["ma_close_20", "ema_close_20"])

        # Second pass: momentum indicators
        result2 = engine.compute(result1, ["rsi_14"])

        # All indicators should be present
        assert "ma_close_20" in result2.columns
        assert "ema_close_20" in result2.columns
        assert "rsi_14" in result2.columns
