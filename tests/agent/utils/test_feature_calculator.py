"""Tests for agent.utils.feature_calculator module."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from agent.utils.feature_calculator import FeatureCalculator


class TestFeatureCalculatorInit:
    """Tests for FeatureCalculator initialization."""

    def test_default_indicators(self):
        """Test default indicators are set."""
        calc = FeatureCalculator()

        assert calc.indicators is not None
        assert len(calc.indicators) > 0

    def test_custom_indicators(self):
        """Test custom indicators can be provided."""
        custom = ["rsi_14"]
        calc = FeatureCalculator(indicators=custom)

        assert len(calc.indicators) == 1
        assert calc.indicators[0] == "rsi_14"


class TestFeatureCalculatorComputeSingle:
    """Tests for compute_single method."""

    @pytest.fixture
    def calculator(self):
        """Create a feature calculator."""
        return FeatureCalculator()

    def test_compute_single_adds_columns(self, calculator, sample_ohlcv_df):
        """Test that compute_single adds indicator columns."""
        result = calculator.compute_single(sample_ohlcv_df, "AAPL")

        # Should have more columns than original
        assert len(result.columns) > len(sample_ohlcv_df.columns)

    def test_compute_single_preserves_original(self, calculator, sample_ohlcv_df):
        """Test that original columns are preserved."""
        result = calculator.compute_single(sample_ohlcv_df, "AAPL")

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_compute_single_empty_df(self, calculator, empty_ohlcv_df):
        """Test handling of empty DataFrame."""
        result = calculator.compute_single(empty_ohlcv_df, "AAPL")

        assert result.is_empty()


class TestFeatureCalculatorComputeParallel:
    """Tests for compute_parallel method."""

    @pytest.fixture
    def calculator(self):
        """Create a feature calculator."""
        return FeatureCalculator()

    def test_compute_parallel_multiple_symbols(self, calculator, sample_ohlcv_df):
        """Test computing for multiple symbols."""
        frames = {
            "AAPL": sample_ohlcv_df,
            "TSLA": sample_ohlcv_df.clone(),
        }

        result = calculator.compute_parallel(frames)

        assert "AAPL" in result
        assert "TSLA" in result

    def test_compute_parallel_handles_error(self, calculator, sample_ohlcv_df):
        """Test error handling in parallel computation."""
        with patch.object(calculator, 'compute_single', side_effect=[
            sample_ohlcv_df,  # First call succeeds
            Exception("Error"),  # Second call fails
        ]):
            frames = {
                "AAPL": sample_ohlcv_df,
                "TSLA": sample_ohlcv_df.clone(),
            }

            result = calculator.compute_parallel(frames)

            # Both should be in result (TSLA with original df due to error)
            assert "AAPL" in result
            assert "TSLA" in result


class TestFeatureCalculatorHelpers:
    """Tests for helper methods."""

    @pytest.fixture
    def calculator(self):
        """Create a feature calculator."""
        return FeatureCalculator()

    def test_get_latest_features(self, calculator, sample_ohlcv_df):
        """Test getting latest features."""
        # Add some indicator columns
        df_with_indicators = sample_ohlcv_df.with_columns([
            pl.lit(50.0).alias("rsi_14"),
            pl.lit(150.0).alias("sma_20"),
        ])

        result = calculator.get_latest_features(df_with_indicators, n_rows=5)

        assert len(result) == 5
        assert "rsi_14" in result.columns
        assert "sma_20" in result.columns
        # OHLCV should be excluded
        assert "close" not in result.columns

    def test_validate_features(self, calculator, sample_ohlcv_df):
        """Test feature validation."""
        # Use minimal indicators for testing
        calculator.indicators = ["rsi_14"]

        # Add expected column
        df = sample_ohlcv_df.with_columns([
            pl.lit(50.0).alias("rsi_14"),
        ])

        result = calculator.validate_features(df)

        assert "valid" in result
        assert "present" in result
        assert "missing" in result
