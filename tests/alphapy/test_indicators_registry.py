"""
Tests for alphapy.indicators.registry module.

Tests the new unified indicator registry system that discovers and manages
technical indicators from multiple sources.
"""
import pytest
from pathlib import Path

from alphapy.indicators.spec import (
    IndicatorSpec,
    IndicatorParam,
    IndicatorSource,
    IndicatorCategory,
    PERIOD_PARAM,
    COLUMN_PARAM,
)
from alphapy.indicators.registry import (
    IndicatorRegistry,
    get_registry,
    get_indicator,
    parse_indicator,
    discover_all,
)


class TestIndicatorParam:
    """Tests for IndicatorParam dataclass."""

    def test_param_creation(self):
        """Test creating a parameter specification."""
        param = IndicatorParam(
            name="p",
            param_type=int,
            default=14,
            description="Lookback period",
        )
        assert param.name == "p"
        assert param.param_type == int
        assert param.default == 14

    def test_param_validation_int(self):
        """Test integer parameter validation."""
        param = IndicatorParam(name="p", param_type=int, default=14)
        assert param.validate(10) is True
        assert param.validate("10") is True  # Can convert
        assert param.validate("abc") is False  # Cannot convert

    def test_param_validation_float(self):
        """Test float parameter validation."""
        param = IndicatorParam(name="sd", param_type=float, default=2.0)
        assert param.validate(1.5) is True
        assert param.validate(2) is True  # int -> float ok
        assert param.validate("2.5") is True

    def test_param_validation_with_bounds(self):
        """Test parameter validation with min/max bounds."""
        param = IndicatorParam(
            name="p",
            param_type=int,
            default=14,
            min_value=1,
            max_value=100,
        )
        assert param.validate(50) is True
        assert param.validate(0) is False  # Below min
        assert param.validate(101) is False  # Above max

    def test_common_params(self):
        """Test pre-defined common parameters."""
        assert PERIOD_PARAM.name == "p"
        assert PERIOD_PARAM.default == 14
        assert COLUMN_PARAM.name == "c"
        assert COLUMN_PARAM.default == "close"


class TestIndicatorSpec:
    """Tests for IndicatorSpec dataclass."""

    def test_spec_creation(self):
        """Test creating an indicator specification."""
        spec = IndicatorSpec(
            name="rsi",
            category=IndicatorCategory.MOMENTUM,
            source=IndicatorSource.TRANSFORMS,
            description="Relative Strength Index",
            params=[PERIOD_PARAM],
        )
        assert spec.name == "rsi"
        assert spec.category == IndicatorCategory.MOMENTUM
        assert spec.source == IndicatorSource.TRANSFORMS

    def test_spec_default_outputs(self):
        """Test default outputs are set to indicator name."""
        spec = IndicatorSpec(
            name="sma",
            category=IndicatorCategory.OVERLAP,
            source=IndicatorSource.TRANSFORMS,
        )
        assert spec.outputs == ["sma"]

    def test_spec_full_name(self):
        """Test full_name property."""
        spec = IndicatorSpec(
            name="rsi",
            category=IndicatorCategory.MOMENTUM,
            source=IndicatorSource.TRANSFORMS,
        )
        assert spec.full_name == "transforms:rsi"

    def test_spec_get_default_params(self):
        """Test getting default parameter values."""
        spec = IndicatorSpec(
            name="ma",
            category=IndicatorCategory.OVERLAP,
            source=IndicatorSource.TRANSFORMS,
            params=[PERIOD_PARAM, COLUMN_PARAM],
        )
        defaults = spec.get_default_params()
        assert defaults["p"] == 14
        assert defaults["c"] == "close"

    def test_spec_validate_params(self):
        """Test parameter validation and filling defaults."""
        spec = IndicatorSpec(
            name="ma",
            category=IndicatorCategory.OVERLAP,
            source=IndicatorSource.TRANSFORMS,
            params=[PERIOD_PARAM, COLUMN_PARAM],
        )
        params = spec.validate_params(p=20)
        assert params["p"] == 20
        assert params["c"] == "close"  # Default filled in

    def test_spec_validate_params_alias(self):
        """Test parameter alias mapping (period -> p)."""
        spec = IndicatorSpec(
            name="ma",
            category=IndicatorCategory.OVERLAP,
            source=IndicatorSource.TRANSFORMS,
            params=[PERIOD_PARAM],
        )
        params = spec.validate_params(period=20)
        assert params["p"] == 20

    def test_spec_to_dict(self):
        """Test serialization to dictionary."""
        spec = IndicatorSpec(
            name="rsi",
            category=IndicatorCategory.MOMENTUM,
            source=IndicatorSource.TRANSFORMS,
            params=[PERIOD_PARAM],
        )
        d = spec.to_dict()
        assert d["name"] == "rsi"
        assert d["category"] == "momentum"
        assert d["source"] == "transforms"

    def test_spec_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "name": "rsi",
            "category": "momentum",
            "source": "transforms",
            "params": [{"name": "p", "type": "int", "default": 14}],
        }
        spec = IndicatorSpec.from_dict(d)
        assert spec.name == "rsi"
        assert spec.category == IndicatorCategory.MOMENTUM


class TestIndicatorRegistry:
    """Tests for IndicatorRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        IndicatorRegistry.reset()
        yield

    def test_singleton_pattern(self):
        """Test registry is a singleton."""
        r1 = IndicatorRegistry()
        r2 = IndicatorRegistry()
        assert r1 is r2

    def test_register_indicator(self):
        """Test registering an indicator."""
        registry = IndicatorRegistry()
        spec = IndicatorSpec(
            name="test_ind",
            category=IndicatorCategory.OTHER,
            source=IndicatorSource.CUSTOM,
        )
        registry.register(spec)
        assert registry.has("test_ind")

    def test_get_indicator(self):
        """Test getting an indicator by name."""
        registry = IndicatorRegistry()
        spec = IndicatorSpec(
            name="test_ind",
            category=IndicatorCategory.OTHER,
            source=IndicatorSource.CUSTOM,
        )
        registry.register(spec)
        result = registry.get("test_ind")
        assert result is spec

    def test_get_indicator_by_alias(self):
        """Test getting an indicator by alias."""
        registry = IndicatorRegistry()
        spec = IndicatorSpec(
            name="moving_average",
            category=IndicatorCategory.OVERLAP,
            source=IndicatorSource.TRANSFORMS,
            aliases=["ma", "sma"],
        )
        registry.register(spec)
        assert registry.get("ma") is spec
        assert registry.get("sma") is spec
        assert registry.get("moving_average") is spec

    def test_has_indicator(self):
        """Test checking indicator existence."""
        registry = IndicatorRegistry()
        spec = IndicatorSpec(
            name="test_ind",
            category=IndicatorCategory.OTHER,
            source=IndicatorSource.CUSTOM,
        )
        registry.register(spec)
        assert registry.has("test_ind") is True
        assert registry.has("nonexistent") is False

    def test_unregister_indicator(self):
        """Test unregistering an indicator."""
        registry = IndicatorRegistry()
        spec = IndicatorSpec(
            name="test_ind",
            category=IndicatorCategory.OTHER,
            source=IndicatorSource.CUSTOM,
        )
        registry.register(spec)
        assert registry.has("test_ind")
        registry.unregister("test_ind")
        assert not registry.has("test_ind")

    def test_list_indicators(self):
        """Test listing all indicators."""
        registry = IndicatorRegistry()
        for i in range(3):
            spec = IndicatorSpec(
                name=f"ind_{i}",
                category=IndicatorCategory.OTHER,
                source=IndicatorSource.CUSTOM,
            )
            registry.register(spec)
        names = registry.list_indicators()
        assert len(names) == 3
        assert "ind_0" in names

    def test_list_indicators_by_category(self):
        """Test filtering indicators by category."""
        registry = IndicatorRegistry()
        registry.register(IndicatorSpec(
            name="rsi",
            category=IndicatorCategory.MOMENTUM,
            source=IndicatorSource.TRANSFORMS,
        ))
        registry.register(IndicatorSpec(
            name="atr",
            category=IndicatorCategory.VOLATILITY,
            source=IndicatorSource.TRANSFORMS,
        ))
        momentum = registry.list_indicators(category=IndicatorCategory.MOMENTUM)
        assert "rsi" in momentum
        assert "atr" not in momentum

    def test_list_aliases(self):
        """Test listing all aliases."""
        registry = IndicatorRegistry()
        spec = IndicatorSpec(
            name="ma",
            category=IndicatorCategory.OVERLAP,
            source=IndicatorSource.TRANSFORMS,
            aliases=["sma", "avg"],
        )
        registry.register(spec)
        aliases = registry.list_aliases()
        assert aliases["sma"] == "ma"
        assert aliases["avg"] == "ma"


class TestIndicatorParsing:
    """Tests for indicator string parsing."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Set up registry with test indicators."""
        IndicatorRegistry.reset()
        registry = IndicatorRegistry()

        # Register test indicators
        registry.register(IndicatorSpec(
            name="rsi",
            category=IndicatorCategory.MOMENTUM,
            source=IndicatorSource.TRANSFORMS,
            params=[PERIOD_PARAM],
        ))
        registry.register(IndicatorSpec(
            name="ma",
            category=IndicatorCategory.OVERLAP,
            source=IndicatorSource.TRANSFORMS,
            params=[COLUMN_PARAM, PERIOD_PARAM],
        ))
        registry.register(IndicatorSpec(
            name="bbands",
            category=IndicatorCategory.VOLATILITY,
            source=IndicatorSource.TRANSFORMS,
            params=[
                COLUMN_PARAM,
                PERIOD_PARAM,
                IndicatorParam(name="sd", param_type=float, default=2.0),
            ],
        ))
        yield

    def test_parse_simple_indicator(self):
        """Test parsing simple indicator string like 'rsi_14'."""
        registry = IndicatorRegistry()
        spec, params = registry.parse("rsi_14")
        assert spec is not None
        assert spec.name == "rsi"
        assert params.get("p") == 14

    def test_parse_indicator_with_column(self):
        """Test parsing indicator with column like 'ma_close_20'."""
        registry = IndicatorRegistry()
        spec, params = registry.parse("ma_close_20")
        assert spec is not None
        assert spec.name == "ma"
        assert params.get("c") == "close"
        assert params.get("p") == 20

    def test_parse_indicator_with_float(self):
        """Test parsing indicator with float like 'bbands_high_10_2.0'."""
        registry = IndicatorRegistry()
        spec, params = registry.parse("bbands_high_10_2.0")
        assert spec is not None
        assert spec.name == "bbands"
        assert params.get("c") == "high"
        assert params.get("p") == 10
        assert params.get("sd") == 2.0

    def test_parse_unknown_indicator(self):
        """Test parsing unknown indicator returns None."""
        registry = IndicatorRegistry()
        spec, params = registry.parse("unknown_indicator")
        assert spec is None
        assert params == {}


class TestTransformDiscovery:
    """Tests for transforms.py auto-discovery."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        IndicatorRegistry.reset()
        yield

    def test_discover_transforms(self):
        """Test discovering transforms.py functions."""
        registry = IndicatorRegistry()
        count = registry.discover_transforms()
        # Should discover many indicators from transforms.py
        assert count > 50  # transforms.py has 76 functions

    def test_discovered_indicator_has_compute_func(self):
        """Test discovered indicators have compute functions."""
        registry = IndicatorRegistry()
        registry.discover_transforms()
        spec = registry.get("rsi")
        assert spec is not None
        assert spec.compute_func is not None

    def test_discovered_indicator_categories(self):
        """Test discovered indicators have correct categories."""
        registry = IndicatorRegistry()
        registry.discover_transforms()

        # Check known indicators have correct categories
        rsi = registry.get("rsi")
        if rsi:
            assert rsi.category == IndicatorCategory.MOMENTUM

        ma = registry.get("ma")
        if ma:
            assert ma.category == IndicatorCategory.OVERLAP

    def test_discovered_indicator_params(self):
        """Test discovered indicators have parameters extracted."""
        registry = IndicatorRegistry()
        registry.discover_transforms()

        ma = registry.get("ma")
        if ma:
            param_names = [p.name for p in ma.params]
            assert "c" in param_names  # column parameter
            assert "p" in param_names  # period parameter


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        IndicatorRegistry.reset()
        yield

    def test_get_registry(self):
        """Test get_registry returns singleton."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_discover_all(self):
        """Test discover_all discovers transforms."""
        count = discover_all()
        assert count > 50

    def test_get_indicator(self):
        """Test get_indicator convenience function."""
        discover_all()
        spec = get_indicator("rsi")
        assert spec is not None
        assert spec.name == "rsi"

    def test_parse_indicator(self):
        """Test parse_indicator convenience function."""
        discover_all()
        spec, params = parse_indicator("rsi_14")
        assert spec is not None
        assert params.get("p") == 14


class TestConfigLoading:
    """Tests for configuration file loading."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        IndicatorRegistry.reset()
        yield

    def test_load_config_file(self, tmp_path):
        """Test loading indicators.yml configuration."""
        # Create a test config file
        config_content = """
external_libraries:
  pandas_ta:
    enabled: false
    module: pandas_ta

aliases:
  sma: ma
  avg: ma

custom_indicators:
  test_indicator:
    source: custom
    category: momentum
    description: A test indicator
    params:
      period: 14
"""
        config_file = tmp_path / "indicators.yml"
        config_file.write_text(config_content)

        registry = IndicatorRegistry()
        count = registry.load_from_config(config_file)

        # Should have loaded the custom indicator
        assert count == 1
        assert registry.has("test_indicator")

        # Should have loaded aliases
        aliases = registry.list_aliases()
        assert aliases.get("sma") == "ma"

    def test_load_nonexistent_config(self):
        """Test loading nonexistent config returns 0."""
        registry = IndicatorRegistry()
        count = registry.load_from_config("/nonexistent/path/indicators.yml")
        assert count == 0


class TestIndicatorComputation:
    """Tests for indicator computation via registry."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Set up registry with discovered transforms."""
        IndicatorRegistry.reset()
        registry = IndicatorRegistry()
        registry.discover_transforms()
        yield

    @pytest.fixture
    def pandas_df(self, sample_ohlcv_df):
        """Convert Polars to Pandas, skip if pyarrow not available."""
        try:
            return sample_ohlcv_df.to_pandas()
        except ModuleNotFoundError:
            pytest.skip("pyarrow not available for Polars->Pandas conversion")

    def test_compute_rsi(self, pandas_df):
        """Test computing RSI via registry."""
        spec = get_indicator("rsi")
        if spec and spec.compute_func:
            result = spec.compute_func(pandas_df, p=14)
            assert result is not None
            assert len(result) == len(pandas_df)

    def test_compute_ma(self, pandas_df):
        """Test computing MA via registry."""
        spec = get_indicator("ma")
        if spec and spec.compute_func:
            result = spec.compute_func(pandas_df, c="close", p=20)
            assert result is not None
            assert len(result) == len(pandas_df)

    def test_compute_via_spec(self, pandas_df):
        """Test computing via IndicatorSpec.compute method."""
        spec = get_indicator("ma")
        if spec and spec.compute_func:
            # Note: spec.compute expects Polars but transforms.py uses Pandas
            # This test validates the interface exists
            assert callable(spec.compute)
