"""
Test basic package imports to ensure no import errors.
"""
import pytest


def test_alphapy_import():
    """Test that main package imports without error."""
    import alphapy
    assert alphapy is not None


class TestCoreModules:
    """Test that core modules can be imported."""
    
    def test_model_import(self):
        """Test model module import."""
        from alphapy import model
        assert model is not None
    
    def test_data_import(self):
        """Test data module import."""
        from alphapy import data
        assert data is not None
    
    def test_frame_import(self):
        """Test frame module import."""
        from alphapy import frame
        assert frame is not None
    
    def test_features_import(self):
        """Test features module import."""
        from alphapy import features
        assert features is not None


class TestOptionalModules:
    """Test optional modules that might have external dependencies."""
    
    def test_market_flow_import(self):
        """Test market flow module import."""
        try:
            from alphapy import market_flow
            assert market_flow is not None
        except ImportError as e:
            pytest.skip(f"Market flow module not available: {e}")
    
    def test_sport_flow_import(self):
        """Test sport flow module import."""
        try:
            from alphapy import sport_flow
            assert sport_flow is not None
        except ImportError as e:
            pytest.skip(f"Sport flow module not available: {e}")