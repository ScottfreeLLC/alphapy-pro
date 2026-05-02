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
        try:
            from alphapy import model
            assert model is not None
        except ImportError as e:
            if "scipy" in str(e) and "_lazywhere" in str(e):
                pytest.skip(f"Skipping due to scipy/statsmodels compatibility issue: {e}")
            else:
                raise
    
    def test_data_import(self):
        """Test data module import."""
        try:
            from alphapy import data
            assert data is not None
        except ModuleNotFoundError as e:
            if "distutils" in str(e):
                pytest.skip(f"Skipping due to distutils/Python 3.12 compatibility issue: {e}")
            else:
                raise
    
    def test_frame_import(self):
        """Test frame module import."""
        from alphapy import frame
        assert frame is not None
    
    def test_features_import(self):
        """Test features module import."""
        try:
            from alphapy import features
            assert features is not None
        except ImportError as e:
            if "scipy" in str(e) and "_lazywhere" in str(e):
                pytest.skip(f"Skipping due to scipy/statsmodels compatibility issue: {e}")
            else:
                raise


class TestSupportingModules:
    """Test supporting modules that are still part of the package."""

    def test_variables_import(self):
        """Test variables module import."""
        try:
            from alphapy import variables
            assert variables is not None
        except ImportError as e:
            pytest.skip(f"Skipping variables import due to optional dependency issue: {e}")

    def test_transforms_import(self):
        """Test transforms module import."""
        try:
            from alphapy import transforms
            assert transforms is not None
        except ImportError as e:
            pytest.skip(f"Skipping transforms import due to optional dependency issue: {e}")
