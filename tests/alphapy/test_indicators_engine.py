"""
Tests for alphapy.indicators.engine module.

Tests all indicator types in both batch and streaming modes.
"""
import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta

from alphapy.indicators import (
    IndicatorEngine,
    add_indicators,
    list_available_indicators,
    TALIPP_INDICATORS,
)


class TestIndicatorNames:
    """Tests for indicator name configuration."""

    def test_all_indicator_types_defined(self):
        """Test that all expected indicator types are defined."""
        expected_types = [
            "sma", "ema", "dema", "tema", "kama", "zlema",  # Trend
            "rsi", "macd", "stoch", "cci", "williams", "roc", "adx", "aroon",  # Momentum
            "bollinger", "atr", "parabolic_sar",  # Volatility
            "obv", "vwap",  # Volume
        ]
        available = list_available_indicators()
        for ind_name in expected_types:
            assert ind_name in available, f"Missing indicator: {ind_name}"

    def test_indicator_config_structure(self):
        """Test indicator config has required keys."""
        for name, config in TALIPP_INDICATORS.items():
            assert "class" in config, f"{name} missing 'class'"
            assert "input" in config, f"{name} missing 'input'"
            assert "output" in config, f"{name} missing 'output'"
            assert config["input"] in ("close", "ohlcv"), f"{name} invalid input type"
            assert config["output"] in ("single", "multi"), f"{name} invalid output type"


class TestIndicatorEngine:
    """Tests for IndicatorEngine class."""

    def test_engine_initialization(self):
        """Test engine can be initialized."""
        engine = IndicatorEngine()
        assert engine is not None
        assert hasattr(engine, "_streaming_indicators")

    def test_compute_empty_indicators(self, sample_ohlcv_df):
        """Test compute with empty indicator list returns original df."""
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv_df, [])
        assert result.shape == sample_ohlcv_df.shape

    def test_compute_preserves_original_columns(self, sample_ohlcv_df):
        """Test that compute preserves original DataFrame columns."""
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv_df, ["ma_close_10"])

        for col in sample_ohlcv_df.columns:
            assert col in result.columns


class TestTrendIndicators:
    """Tests for trend indicators (MA, EMA) using transforms.py."""

    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    def test_ma_computation(self, engine, sample_ohlcv_df):
        """Test MA indicator computation using transforms.py ma()."""
        result = engine.compute(sample_ohlcv_df, ["ma_close_20"])

        assert "ma_close_20" in result.columns
        # MA should have NaN for first (period-1) values
        ma_values = result["ma_close_20"].to_list()
        assert ma_values[-1] is not None  # Last value should exist

    def test_ema_computation(self, engine, sample_ohlcv_df):
        """Test EMA indicator computation using transforms.py ema()."""
        result = engine.compute(sample_ohlcv_df, ["ema_close_12"])

        assert "ema_close_12" in result.columns
        ema_values = result["ema_close_12"].to_list()
        assert ema_values[-1] is not None

    @pytest.mark.parametrize("period", [5, 10, 20, 50])
    def test_ma_various_periods(self, engine, sample_ohlcv_df, period):
        """Test MA with various periods."""
        result = engine.compute(sample_ohlcv_df, [f"ma_close_{period}"])

        assert f"ma_close_{period}" in result.columns


class TestMomentumIndicators:
    """Tests for momentum indicators using transforms.py."""

    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    def test_rsi_computation(self, engine, sample_ohlcv_df):
        """Test RSI indicator computation using transforms.py rsi()."""
        result = engine.compute(sample_ohlcv_df, ["rsi_14"])

        assert "rsi_14" in result.columns
        # RSI should be between 0 and 100
        # Filter out None and NaN values
        rsi_values = [v for v in result["rsi_14"].to_list()
                      if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if rsi_values:
            assert all(0 <= v <= 100 for v in rsi_values)

    def test_macd_not_in_transforms(self, engine, sample_ohlcv_df):
        """Test that macd is not available in transforms.py.

        Note: transforms.py doesn't have a macd function. If you need MACD,
        use talipp (via ap:macd) or external sources (ta:MACD, pta:macd).
        """
        # macd doesn't exist in transforms.py
        result = engine.compute(sample_ohlcv_df, ["macd"])
        # No macd column should be added since function doesn't exist
        macd_cols = [c for c in result.columns if "macd" in c.lower()]
        assert len(macd_cols) == 0


class TestVolatilityIndicators:
    """Tests for volatility indicators using transforms.py."""

    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    def test_bbands_computation(self, engine, sample_ohlcv_df):
        """Test Bollinger Bands indicator computation using transforms.py bbands()."""
        # bbands(df, c='close', p=20, sd=2) - default params
        result = engine.compute(sample_ohlcv_df, ["bbands_close_20_2"])

        # bbands returns DataFrame with multiple columns
        bbands_cols = [c for c in result.columns if "bbands" in c.lower()]
        assert len(bbands_cols) >= 1

    def test_truerange_computation(self, engine, sample_ohlcv_df):
        """Test True Range indicator computation using transforms.py truerange()."""
        result = engine.compute(sample_ohlcv_df, ["truerange"])

        assert "truerange" in result.columns
        # True range should be positive
        tr_values = [v for v in result["truerange"].to_list()
                     if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if tr_values:
            assert all(v >= 0 for v in tr_values)


class TestMultipleIndicators:
    """Tests for computing multiple indicators at once."""

    def test_compute_multiple_indicators(self, sample_ohlcv_df):
        """Test computing multiple indicators in one call."""
        engine = IndicatorEngine()
        result = engine.compute(sample_ohlcv_df, [
            "ma_close_10",
            "rsi_14",
            "ema_close_20",
        ])

        assert "ma_close_10" in result.columns
        assert "rsi_14" in result.columns
        assert "ema_close_20" in result.columns

    def test_compute_all_single_column_indicators(self, indicator_test_data):
        """Test computing multiple single-column indicators."""
        engine = IndicatorEngine()
        indicators = [
            "ma_close_20",
            "ema_close_20",
            "rsi_14",
            "truerange",
        ]
        result = engine.compute(indicator_test_data, indicators)

        for ind in indicators:
            assert ind in result.columns


class TestAddIndicatorsFunction:
    """Tests for add_indicators convenience function."""

    def test_add_indicators_with_strings(self, sample_ohlcv_df):
        """Test add_indicators with indicator strings."""
        result = add_indicators(sample_ohlcv_df, ["ma_close_20", "rsi_7", "ema_close_10"])

        assert "ma_close_20" in result.columns
        assert "rsi_7" in result.columns
        assert "ema_close_10" in result.columns

    def test_add_indicators_multiple(self, sample_ohlcv_df):
        """Test add_indicators with multiple indicators."""
        result = add_indicators(sample_ohlcv_df, [
            "ma_close_10",
            "rsi_14",
            "ema_close_20",
        ])

        assert "ma_close_10" in result.columns
        assert "rsi_14" in result.columns
        assert "ema_close_20" in result.columns

    def test_add_indicators_unknown_function(self, sample_ohlcv_df):
        """Test add_indicators with unknown function logs error."""
        # Unknown functions should be logged as errors but not crash
        result = add_indicators(sample_ohlcv_df, ["nonexistent_indicator_20"])
        # Column should not be added
        assert "nonexistent_indicator_20" not in result.columns


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pl.DataFrame({
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }).cast({
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        })

        result = IndicatorEngine().compute(empty_df, ["ma_close_10"])

        assert "ma_close_10" in result.columns
        assert len(result) == 0

    def test_small_dataframe_less_than_period(self, small_ohlcv_df):
        """Test with DataFrame smaller than indicator period."""
        engine = IndicatorEngine()
        # 5 rows, but MA needs 20
        result = engine.compute(small_ohlcv_df, ["ma_close_20"])

        # Should still compute, just with all NaN
        assert "ma_close_20" in result.columns
        ma_values = result["ma_close_20"].to_list()
        # All values should be NaN since not enough data
        assert all(v is None or (isinstance(v, float) and np.isnan(v)) for v in ma_values)

    def test_dataframe_with_nan_values(self):
        """Test handling of NaN values in data."""
        df = pl.DataFrame({
            "datetime": [datetime.now() - timedelta(minutes=5 * i) for i in range(20)][::-1],
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.0 if i != 10 else None for i in range(20)],  # One NaN
            "volume": [1000] * 20,
        })

        engine = IndicatorEngine()
        # Should not raise error
        result = engine.compute(df, ["ma_close_5"])
        assert "ma_close_5" in result.columns


class TestIndicatorAccuracy:
    """Tests for indicator calculation accuracy."""

    def test_ma_accuracy(self):
        """Test MA calculation accuracy against known values."""
        # Create simple data where MA is easy to verify
        df = pl.DataFrame({
            "datetime": [datetime.now() - timedelta(minutes=5 * i) for i in range(10)][::-1],
            "open": [float(i) for i in range(1, 11)],
            "high": [float(i) for i in range(1, 11)],
            "low": [float(i) for i in range(1, 11)],
            "close": [float(i) for i in range(1, 11)],  # 1, 2, 3, ..., 10
            "volume": [1000] * 10,
        })

        engine = IndicatorEngine()
        result = engine.compute(df, ["ma_close_5"])

        # MA(5) at position 4 (0-indexed) should be (1+2+3+4+5)/5 = 3.0
        # MA(5) at position 9 should be (6+7+8+9+10)/5 = 8.0
        ma_values = result["ma_close_5"].to_list()

        # Last value: average of 6,7,8,9,10 = 8.0
        assert abs(ma_values[-1] - 8.0) < 0.01

    def test_rsi_bounds(self, indicator_test_data):
        """Test RSI is always between 0 and 100."""
        engine = IndicatorEngine()
        result = engine.compute(indicator_test_data, ["rsi_14"])

        # Filter out None and NaN values
        rsi_values = [v for v in result["rsi_14"].to_list()
                      if v is not None and not (isinstance(v, float) and np.isnan(v))]
        assert all(0 <= v <= 100 for v in rsi_values)


class TestMultiSourceIndicators:
    """Tests for multi-source indicator support (ap:, ta:, pta:, vbt:)."""

    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    def test_alphapy_prefix(self, engine, sample_ohlcv_df):
        """Test ap: prefix routes to AlphaPy and produces standard column name."""
        result = engine.compute(sample_ohlcv_df, ["ap:rsi_14"])
        # Column name doesn't include prefix - prefix is just for routing
        assert "rsi_14" in result.columns

    def test_default_source_is_alphapy(self, engine, sample_ohlcv_df):
        """Test that no prefix defaults to alphapy source."""
        # Both should work the same way and produce same column name
        result1 = engine.compute(sample_ohlcv_df, ["rsi_14"])
        result2 = engine.compute(sample_ohlcv_df, ["ap:rsi_14"])

        # Both should have computed RSI with same column name
        assert "rsi_14" in result1.columns
        assert "rsi_14" in result2.columns

    def test_compute_with_string_specs(self, engine, sample_ohlcv_df):
        """Test computing indicators using string specs."""
        result = engine.compute(sample_ohlcv_df, ["ma_close_20", "ema_close_10", "rsi_14"])

        assert "ma_close_20" in result.columns
        assert "ema_close_10" in result.columns
        assert "rsi_14" in result.columns

    def test_compute_with_prefixed_strings(self, engine, sample_ohlcv_df):
        """Test computing with explicit ap: prefix."""
        result = engine.compute(sample_ohlcv_df, [
            "ap:ma_close_10",
            "rsi_14",
            "ap:ema_close_20",
        ])

        assert "ma_close_10" in result.columns
        assert "rsi_14" in result.columns
        # Column name doesn't include prefix
        assert "ema_close_20" in result.columns


class TestSourcePrefixParsing:
    """Tests for source prefix parsing functions."""

    def test_parse_source_prefix_alphapy(self):
        """Test parsing ap: prefix."""
        from alphapy.indicators import parse_source_prefix

        source, rest = parse_source_prefix("ap:rsi_14")
        assert source == "alphapy"
        assert rest == "rsi_14"

    def test_parse_source_prefix_talib(self):
        """Test parsing ta: prefix."""
        from alphapy.indicators import parse_source_prefix

        source, rest = parse_source_prefix("ta:RSI_14")
        assert source == "talib"
        assert rest == "RSI_14"

    def test_parse_source_prefix_pandas_ta(self):
        """Test parsing pta: prefix."""
        from alphapy.indicators import parse_source_prefix

        source, rest = parse_source_prefix("pta:supertrend_10_3")
        assert source == "pandas_ta"
        assert rest == "supertrend_10_3"

    def test_parse_source_prefix_vectorbt(self):
        """Test parsing vbt: prefix."""
        from alphapy.indicators import parse_source_prefix

        source, rest = parse_source_prefix("vbt:rsi_14")
        assert source == "vectorbt"
        assert rest == "rsi_14"

    def test_parse_source_prefix_default(self):
        """Test no prefix defaults to alphapy."""
        from alphapy.indicators import parse_source_prefix

        source, rest = parse_source_prefix("rsi_14")
        assert source == "alphapy"
        assert rest == "rsi_14"


class TestListAllIndicators:
    """Tests for list_all_indicators function."""

    def test_list_all_indicators_returns_dict(self):
        """Test list_all_indicators returns a dictionary."""
        from alphapy.indicators import list_all_indicators

        result = list_all_indicators()
        assert isinstance(result, dict)

    def test_list_all_indicators_has_alphapy(self):
        """Test alphapy indicators are always present."""
        from alphapy.indicators import list_all_indicators

        result = list_all_indicators()
        assert "alphapy" in result
        assert len(result["alphapy"]) > 0

    def test_list_all_indicators_keys(self):
        """Test list_all_indicators has expected source keys."""
        from alphapy.indicators import list_all_indicators

        result = list_all_indicators()
        expected_keys = {"alphapy", "talib", "pandas_ta", "vectorbt"}
        assert set(result.keys()) == expected_keys


class TestAlphaPyIndicators:
    """Tests for AlphaPy transforms.py indicator access."""

    def test_list_alphapy_indicators(self):
        """Test listing AlphaPy indicators."""
        from alphapy.indicators import list_alphapy_indicators

        result = list_alphapy_indicators()
        assert isinstance(result, list)
        # Should have 76 functions from transforms.py
        assert len(result) > 50

    def test_alphapy_indicators_include_rsi(self):
        """Test AlphaPy indicators include common indicators."""
        from alphapy.indicators import list_alphapy_indicators

        result = list_alphapy_indicators()
        assert "rsi" in result

    def test_alphapy_indicators_include_ma(self):
        """Test AlphaPy indicators include moving average functions."""
        from alphapy.indicators import list_alphapy_indicators

        result = list_alphapy_indicators()
        assert "ma" in result


class TestTransformsMetadata:
    """Tests for transforms.py metadata auto-detection system."""

    def test_get_transforms_metadata_returns_dict(self):
        """Test that metadata returns a dictionary."""
        from alphapy.indicators.registry import get_transforms_metadata

        result = get_transforms_metadata()
        assert isinstance(result, dict)
        assert len(result) > 50  # Should have 76+ functions

    def test_metadata_has_params_and_defaults(self):
        """Test metadata contains params and defaults."""
        from alphapy.indicators.registry import get_transforms_metadata

        result = get_transforms_metadata()

        # Check a function with no params
        assert "gap" in result
        assert result["gap"]["params"] == []
        assert result["gap"]["defaults"] == {}

        # Check a function with params
        assert "rsi" in result
        assert "p" in result["rsi"]["params"]
        assert result["rsi"]["defaults"]["p"] == 14

    def test_metadata_has_special_flags(self):
        """Test metadata includes special handling flags."""
        from alphapy.indicators.registry import get_transforms_metadata

        result = get_transforms_metadata()

        # diff should have length_change flag
        assert "diff" in result
        assert result["diff"].get("length_change") is True

        # c2max should have row_wise flag
        assert "c2max" in result
        assert result["c2max"].get("row_wise") is True

        # dateparts should have dataframe return flag
        assert "dateparts" in result
        assert result["dateparts"].get("returns") == "dataframe"


class TestParseTransformsString:
    """Tests for parse_transforms_string function.

    Note: parse_transforms_string now returns (name, args_list) where args_list
    is a simple list of positional arguments, not a dict. This matches the
    vfunc calling convention from variables.py.
    """

    def test_parse_no_params(self):
        """Test parsing function with no params."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("gap")
        assert name == "gap"
        assert args == []

    def test_parse_period_only(self):
        """Test parsing function with period param."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("rsi_14")
        assert name == "rsi"
        assert args == [14]

    def test_parse_column_and_period(self):
        """Test parsing function with column and period."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("ma_close_20")
        assert name == "ma"
        assert args == ["close", 20]

    def test_parse_column_and_period_different_column(self):
        """Test parsing with non-default column."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("ema_high_10")
        assert name == "ema"
        assert args == ["high", 10]

    def test_parse_with_float_param(self):
        """Test parsing with float parameter."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("bbands_close_20_2.5")
        assert name == "bbands"
        assert args == ["close", 20, 2.5]

    def test_parse_diff_with_n(self):
        """Test parsing diff with n parameter."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("diff_close_1")
        assert name == "diff"
        assert args == ["close", 1]


class TestSpecialFunctionChecks:
    """Tests for special function check helpers."""

    def test_is_row_wise_function(self):
        """Test row-wise function detection."""
        from alphapy.indicators.registry import is_row_wise_function

        assert is_row_wise_function("c2max") is True
        assert is_row_wise_function("c2min") is True
        assert is_row_wise_function("gtval0") is True
        assert is_row_wise_function("rsi") is False

    def test_is_length_changing_function(self):
        """Test length-changing function detection."""
        from alphapy.indicators.registry import is_length_changing_function

        assert is_length_changing_function("diff") is True
        assert is_length_changing_function("rsi") is False
        assert is_length_changing_function("ma") is False

    def test_returns_dataframe(self):
        """Test DataFrame return detection."""
        from alphapy.indicators.registry import returns_dataframe

        assert returns_dataframe("dateparts") is True
        assert returns_dataframe("bizday") is True
        assert returns_dataframe("runstest") is True
        assert returns_dataframe("rsi") is False


class TestTransformsCompute:
    """Tests for _compute_transforms with various edge cases."""

    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    def test_compute_diff_pads_with_nan(self, engine, sample_ohlcv_df):
        """Test that diff pads with NaN at beginning."""
        result = engine.compute(sample_ohlcv_df, ["ap:diff_close_1"])
        col_name = [c for c in result.columns if "diff" in c][0]

        # First value should be NaN (padding)
        assert np.isnan(result[col_name][0])
        # Length should match original
        assert len(result[col_name]) == len(sample_ohlcv_df)

    def test_compute_row_wise_logs_error(self, engine, sample_ohlcv_df):
        """Test that row-wise helpers log error and continue."""
        # c2max should log an error but not crash
        result = engine.compute(sample_ohlcv_df, ["ap:c2max_close_open"])
        # Check no c2max column was added
        assert not any("c2max" in c for c in result.columns)

    def test_compute_dateparts_multi_column(self, engine):
        """Test dateparts returns multiple columns."""
        # Create DataFrame with datetime column
        df = pl.DataFrame({
            "datetime": [datetime.now() - timedelta(days=i) for i in range(10)],
            "close": [100.0 + i for i in range(10)],
            "open": [99.0 + i for i in range(10)],
            "high": [101.0 + i for i in range(10)],
            "low": [98.0 + i for i in range(10)],
            "volume": [1000] * 10,
        })

        result = engine.compute(df, ["ap:dateparts_datetime"])
        new_cols = [c for c in result.columns if "dateparts" in c]

        # Should have multiple columns (year, month, day, dayofweek)
        assert len(new_cols) >= 4

    def test_compute_standard_transforms(self, engine, sample_ohlcv_df):
        """Test standard transforms.py functions work."""
        result = engine.compute(sample_ohlcv_df, ["ap:gap", "ap:rsi_14", "ap:ma_close_20"])

        assert "gap" in result.columns
        assert "rsi_14" in result.columns
        assert "ma_close_20" in result.columns


class TestVariablesYmlIntegration:
    """Tests for variable definitions matching variables.yml patterns.

    These tests verify that indicator strings work the same way as
    variables defined in variables.yml, following the vparse convention.
    """

    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    def test_rsi_function_call(self, engine, sample_ohlcv_df):
        """Test rsi_14 -> rsi(df, 14) as defined in variables.yml."""
        result = engine.compute(sample_ohlcv_df, ["rsi_14"])
        assert "rsi_14" in result.columns

    def test_ma_with_column_and_period(self, engine, sample_ohlcv_df):
        """Test ma_close_20 -> ma(df, 'close', 20) following smac alias pattern."""
        # smac in variables.yml is aliased to ma_close
        result = engine.compute(sample_ohlcv_df, ["ma_close_20"])
        assert "ma_close_20" in result.columns

    def test_ema_with_column_and_period(self, engine, sample_ohlcv_df):
        """Test ema_close_12 -> ema(df, 'close', 12)."""
        result = engine.compute(sample_ohlcv_df, ["ema_close_12"])
        assert "ema_close_12" in result.columns

    def test_truerange_no_params(self, engine, sample_ohlcv_df):
        """Test truerange -> truerange(df) as defined in variables.yml."""
        result = engine.compute(sample_ohlcv_df, ["truerange"])
        assert "truerange" in result.columns

    def test_gap_no_params(self, engine, sample_ohlcv_df):
        """Test gap -> gap(df) as defined in variables.yml."""
        result = engine.compute(sample_ohlcv_df, ["gap"])
        assert "gap" in result.columns

    def test_hlrange_no_params(self, engine, sample_ohlcv_df):
        """Test hlrange -> hlrange(df) as defined in variables.yml."""
        result = engine.compute(sample_ohlcv_df, ["hlrange"])
        assert "hlrange" in result.columns

    def test_higher_with_column(self, engine, sample_ohlcv_df):
        """Test higher_close -> higher(df, 'close') following hc alias pattern."""
        result = engine.compute(sample_ohlcv_df, ["higher_close"])
        assert "higher_close" in result.columns

    def test_lower_with_column(self, engine, sample_ohlcv_df):
        """Test lower_close -> lower(df, 'close') following lc alias pattern."""
        result = engine.compute(sample_ohlcv_df, ["lower_close"])
        assert "lower_close" in result.columns

    def test_highest_with_column_and_period(self, engine, sample_ohlcv_df):
        """Test highest_close_5 -> highest(df, 'close', 5) following cmax alias."""
        result = engine.compute(sample_ohlcv_df, ["highest_close_5"])
        assert "highest_close_5" in result.columns

    def test_lowest_with_column_and_period(self, engine, sample_ohlcv_df):
        """Test lowest_close_5 -> lowest(df, 'close', 5) following cmin alias."""
        result = engine.compute(sample_ohlcv_df, ["lowest_close_5"])
        assert "lowest_close_5" in result.columns

    def test_bbands_with_params(self, engine, sample_ohlcv_df):
        """Test bbands_close_20_2 -> bbands(df, 'close', 20, 2)."""
        result = engine.compute(sample_ohlcv_df, ["bbands_close_20_2"])
        # bbands returns multiple columns
        bbands_cols = [c for c in result.columns if "bbands" in c.lower()]
        assert len(bbands_cols) >= 1

    def test_adx_with_period(self, engine, sample_ohlcv_df):
        """Test adx_14 -> adx(df, 14) as defined in variables.yml."""
        result = engine.compute(sample_ohlcv_df, ["adx_14"])
        assert "adx_14" in result.columns

    def test_netreturn_with_column(self, engine, sample_ohlcv_df):
        """Test netreturn_close -> netreturn(df, 'close') following roi alias."""
        result = engine.compute(sample_ohlcv_df, ["netreturn_close"])
        assert "netreturn_close" in result.columns

    def test_atr_alias_expansion(self, engine, sample_ohlcv_df):
        """Test atr_14 alias expands to ma_truerange_14.

        In variables.yml: atr is aliased to ma_truerange
        So atr_14 should expand to ma_truerange_14 -> ma(df, 'truerange', 14)
        The engine auto-resolves the truerange dependency.
        """
        # Engine auto-resolves truerange dependency
        result = engine.compute(sample_ohlcv_df, ["atr_14"])

        assert "atr_14" in result.columns
        assert "truerange" in result.columns  # Auto-computed


class TestAliasExpansion:
    """Tests for alias expansion from variables.yml.

    These tests verify that aliases defined in config/variables.yml
    are properly expanded when computing indicators.
    """

    @pytest.fixture
    def engine(self):
        return IndicatorEngine()

    def test_parse_atr_alias(self):
        """Test that atr_14 is parsed to ma with truerange column."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("atr_14")
        # atr -> ma_truerange, so atr_14 -> ma('truerange', 14)
        assert name == "ma"
        assert args == ["truerange", 14]

    def test_parse_smac_alias(self):
        """Test that smac_20 is parsed to ma with close column."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("smac_20")
        # smac -> ma_close, so smac_20 -> ma('close', 20)
        assert name == "ma"
        assert args == ["close", 20]

    def test_parse_cmax_alias(self):
        """Test that cmax_5 is parsed to highest with close column."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("cmax_5")
        # cmax -> highest_close, so cmax_5 -> highest('close', 5)
        assert name == "highest"
        assert args == ["close", 5]

    def test_parse_cmin_alias(self):
        """Test that cmin_5 is parsed to lowest with close column."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("cmin_5")
        # cmin -> lowest_close, so cmin_5 -> lowest('close', 5)
        assert name == "lowest"
        assert args == ["close", 5]

    def test_parse_hc_alias(self):
        """Test that hc is parsed to higher with close column."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("hc")
        # hc -> higher_close
        assert name == "higher"
        assert args == ["close"]

    def test_parse_lc_alias(self):
        """Test that lc is parsed to lower with close column."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("lc")
        # lc -> lower_close
        assert name == "lower"
        assert args == ["close"]

    def test_parse_hh_alias(self):
        """Test that hh is parsed to higher with high column."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("hh")
        # hh -> higher_high
        assert name == "higher"
        assert args == ["high"]

    def test_parse_roi_alias(self):
        """Test that roi_5 is parsed to netreturn with close column."""
        from alphapy.indicators.registry import parse_transforms_string

        name, args = parse_transforms_string("roi_5")
        # roi -> netreturn_close, so roi_5 -> netreturn('close', 5)
        assert name == "netreturn"
        assert args == ["close", 5]

    def test_compute_smac_alias(self, engine, sample_ohlcv_df):
        """Test computing smac_20 works via alias expansion."""
        result = engine.compute(sample_ohlcv_df, ["smac_20"])
        assert "smac_20" in result.columns

    def test_compute_cmax_alias(self, engine, sample_ohlcv_df):
        """Test computing cmax_5 works via alias expansion."""
        result = engine.compute(sample_ohlcv_df, ["cmax_5"])
        assert "cmax_5" in result.columns

    def test_compute_hc_alias(self, engine, sample_ohlcv_df):
        """Test computing hc works via alias expansion."""
        result = engine.compute(sample_ohlcv_df, ["hc"])
        assert "hc" in result.columns

    def test_compute_roi_alias(self, engine, sample_ohlcv_df):
        """Test computing roi_5 works via alias expansion."""
        result = engine.compute(sample_ohlcv_df, ["roi_5"])
        assert "roi_5" in result.columns

    def test_compute_atr_with_auto_dependency(self, engine, sample_ohlcv_df):
        """Test that atr_14 auto-resolves truerange dependency.

        atr is aliased to ma_truerange, so atr_14 expands to ma(df, 'truerange', 14).
        The engine should automatically compute truerange if it doesn't exist.
        """
        # Don't compute truerange first - engine should auto-resolve
        result = engine.compute(sample_ohlcv_df, ["atr_14"])

        # Both atr_14 and the auto-computed truerange should exist
        assert "atr_14" in result.columns
        assert "truerange" in result.columns  # Auto-computed dependency
