"""
Test version information and package imports.
"""
import alphapy


def test_version_exists():
    """Test that version information exists."""
    assert hasattr(alphapy, '__version__')
    assert alphapy.__version__ is not None
    assert isinstance(alphapy.__version__, str)


def test_version_format():
    """Test that version follows semantic versioning format."""
    version = alphapy.__version__
    parts = version.split('.')
    assert len(parts) == 3, f"Version {version} should have 3 parts (major.minor.patch)"
    
    # Check that all parts are numeric
    for part in parts:
        assert part.isdigit(), f"Version part '{part}' should be numeric"


def test_current_version():
    """Test that current version is 3.0.0."""
    assert alphapy.__version__ == "3.0.0"