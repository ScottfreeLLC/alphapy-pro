"""
Test utilities module functions.
"""
import pytest
import os
import tempfile
from datetime import datetime, timedelta
from alphapy.utilities import (
    datetime_stamp,
    ensure_dir,
    remove_list_items,
    valid_date,
    valid_name,
    subtract_days
)


class TestDateTimeFunctions:
    """Test date and time utility functions."""
    
    def test_datetime_stamp(self):
        """Test datetime_stamp returns a string timestamp."""
        stamp = datetime_stamp()
        assert isinstance(stamp, str)
        assert len(stamp) > 0
        # Should be in format YYYYMMDD_HHMMSS
        assert len(stamp) == 15  # 8 + 1 + 6
        assert stamp[8] == '_'
    
    def test_valid_date_with_valid_dates(self):
        """Test valid_date with valid date strings."""
        valid_dates = [
            "2023-01-01",
            "2023-12-31",
            "2024-02-29",  # leap year
        ]
        for date_str in valid_dates:
            assert valid_date(date_str) == date_str
    
    def test_valid_date_with_invalid_dates(self):
        """Test valid_date with invalid date strings."""
        invalid_dates = [
            "2023-13-01",  # invalid month
            "2023-01-32",  # invalid day
            "not-a-date",
            "",
            "2023/01/01",  # wrong format
        ]
        for date_str in invalid_dates:
            with pytest.raises(Exception):  # Could be ArgumentTypeError or ValueError
                valid_date(date_str)
    
    def test_subtract_days(self):
        """Test subtract_days function."""
        result = subtract_days("2023-01-10", 5)
        assert result == "2023-01-05"
        
        result = subtract_days("2023-01-01", 1)
        assert result == "2022-12-31"


class TestFileSystemFunctions:
    """Test file system utility functions."""
    
    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test ensure_dir creates a directory."""
        test_dir = os.path.join(temp_dir, "test_directory")
        assert not os.path.exists(test_dir)
        
        ensure_dir(test_dir)
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
    
    def test_ensure_dir_existing_directory(self, temp_dir):
        """Test ensure_dir with existing directory."""
        # Should not raise an error
        ensure_dir(temp_dir)
        assert os.path.exists(temp_dir)


class TestListUtilities:
    """Test list manipulation utilities."""
    
    def test_remove_list_items(self):
        """Test remove_list_items function."""
        original_list = [1, 2, 3, 4, 5]
        items_to_remove = [2, 4]
        result = remove_list_items(items_to_remove, original_list)
        assert result == [1, 3, 5]
    
    def test_remove_list_items_empty_removal(self):
        """Test remove_list_items with empty removal list."""
        original_list = [1, 2, 3]
        result = remove_list_items([], original_list)
        assert result == [1, 2, 3]
    
    def test_remove_list_items_nonexistent_items(self):
        """Test remove_list_items with items not in list."""
        original_list = [1, 2, 3]
        items_to_remove = [4, 5]
        result = remove_list_items(items_to_remove, original_list)
        assert result == [1, 2, 3]


class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_valid_name_with_valid_names(self):
        """Test valid_name with valid names."""
        valid_names = [
            "test",
            "test_name",
            "TestName",
            "test123",
            "_private",
        ]
        for name in valid_names:
            assert valid_name(name) == True
    
    def test_valid_name_with_invalid_names(self):
        """Test valid_name with invalid names."""
        invalid_names = [
            "",
            "123test",  # starts with number
            "test-name",  # contains hyphen
            "test name",  # contains space
            "test.name",  # contains dot
        ]
        for name in invalid_names:
            assert valid_name(name) == False