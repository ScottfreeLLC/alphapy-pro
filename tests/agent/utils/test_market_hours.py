"""Tests for agent.utils.market_hours module.

These are pure logic tests - no external API calls.
"""

from datetime import datetime, time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from freezegun import freeze_time

from agent.utils.market_hours import MarketHours


class TestMarketHoursConstants:
    """Tests for market hours constants."""

    def test_stock_open_time(self):
        """Test stock market open time."""
        assert MarketHours.STOCK_OPEN == time(9, 30)

    def test_stock_close_time(self):
        """Test stock market close time."""
        assert MarketHours.STOCK_CLOSE == time(16, 0)

    def test_premarket_open(self):
        """Test premarket open time."""
        assert MarketHours.PREMARKET_OPEN == time(4, 0)

    def test_afterhours_close(self):
        """Test after hours close time."""
        assert MarketHours.AFTERHOURS_CLOSE == time(20, 0)

    def test_timezone(self):
        """Test Eastern timezone is set."""
        assert MarketHours.ET == ZoneInfo("America/New_York")


class TestIsStockMarketOpen:
    """Tests for is_stock_market_open method."""

    @freeze_time("2024-01-16 10:30:00", tz_offset=-5)  # Tuesday, 10:30 AM ET
    def test_open_during_market_hours(self):
        """Test market is open during regular hours."""
        dt = datetime(2024, 1, 16, 10, 30, tzinfo=MarketHours.ET)
        assert MarketHours.is_stock_market_open(dt) is True

    @freeze_time("2024-01-16 09:00:00", tz_offset=-5)  # Before open
    def test_closed_before_market_open(self):
        """Test market is closed before 9:30 AM."""
        dt = datetime(2024, 1, 16, 9, 0, tzinfo=MarketHours.ET)
        assert MarketHours.is_stock_market_open(dt) is False

    @freeze_time("2024-01-16 16:30:00", tz_offset=-5)  # After close
    def test_closed_after_market_close(self):
        """Test market is closed after 4:00 PM."""
        dt = datetime(2024, 1, 16, 16, 30, tzinfo=MarketHours.ET)
        assert MarketHours.is_stock_market_open(dt) is False

    def test_closed_on_saturday(self):
        """Test market is closed on Saturday."""
        dt = datetime(2024, 1, 20, 12, 0, tzinfo=MarketHours.ET)  # Saturday
        assert MarketHours.is_stock_market_open(dt) is False

    def test_closed_on_sunday(self):
        """Test market is closed on Sunday."""
        dt = datetime(2024, 1, 21, 12, 0, tzinfo=MarketHours.ET)  # Sunday
        assert MarketHours.is_stock_market_open(dt) is False

    def test_closed_on_holiday(self):
        """Test market is closed on holiday."""
        dt = datetime(2024, 12, 25, 12, 0, tzinfo=MarketHours.ET)  # Christmas
        assert MarketHours.is_stock_market_open(dt) is False

    def test_open_at_exactly_open_time(self):
        """Test market is open at exactly 9:30 AM."""
        dt = datetime(2024, 1, 16, 9, 30, tzinfo=MarketHours.ET)  # Tuesday
        assert MarketHours.is_stock_market_open(dt) is True

    def test_closed_at_exactly_close_time(self):
        """Test market is closed at exactly 4:00 PM."""
        dt = datetime(2024, 1, 16, 16, 0, tzinfo=MarketHours.ET)  # Tuesday
        assert MarketHours.is_stock_market_open(dt) is False


class TestIsCryptoMarketOpen:
    """Tests for is_crypto_market_open method."""

    def test_crypto_always_open(self):
        """Test crypto market is always open."""
        assert MarketHours.is_crypto_market_open() is True


class TestIsMarketOpen:
    """Tests for is_market_open method."""

    def test_stock_symbol(self):
        """Test stock symbol uses stock market hours."""
        dt = datetime(2024, 1, 16, 12, 0, tzinfo=MarketHours.ET)  # Tuesday noon
        assert MarketHours.is_market_open("AAPL", dt) is True

    def test_crypto_symbol(self):
        """Test crypto symbol is always open."""
        dt = datetime(2024, 1, 20, 2, 0, tzinfo=MarketHours.ET)  # Saturday 2 AM
        assert MarketHours.is_market_open("BTC/USD", dt) is True


class TestMinutesUntilClose:
    """Tests for minutes_until_close method."""

    def test_minutes_until_close(self):
        """Test calculating minutes until close."""
        dt = datetime(2024, 1, 16, 15, 30, tzinfo=MarketHours.ET)  # 30 min before close
        result = MarketHours.minutes_until_close(dt)
        assert result == 30

    def test_minutes_at_open(self):
        """Test minutes at market open."""
        dt = datetime(2024, 1, 16, 9, 30, tzinfo=MarketHours.ET)
        result = MarketHours.minutes_until_close(dt)
        assert result == 390  # 6.5 hours

    def test_none_when_closed(self):
        """Test returns None when market is closed."""
        dt = datetime(2024, 1, 16, 17, 0, tzinfo=MarketHours.ET)  # After close
        result = MarketHours.minutes_until_close(dt)
        assert result is None


class TestMinutesUntilOpen:
    """Tests for minutes_until_open method."""

    def test_minutes_until_open_before_market(self):
        """Test calculating minutes before open."""
        dt = datetime(2024, 1, 16, 8, 30, tzinfo=MarketHours.ET)  # 1 hour before
        result = MarketHours.minutes_until_open(dt)
        assert result == 60

    def test_none_when_open(self):
        """Test returns None when market is open."""
        dt = datetime(2024, 1, 16, 12, 0, tzinfo=MarketHours.ET)  # During market
        result = MarketHours.minutes_until_open(dt)
        assert result is None

    def test_none_on_weekend(self):
        """Test returns None on weekend (can't calculate next open easily)."""
        dt = datetime(2024, 1, 20, 12, 0, tzinfo=MarketHours.ET)  # Saturday
        result = MarketHours.minutes_until_open(dt)
        assert result is None


class TestShouldTrade:
    """Tests for should_trade method."""

    def test_should_trade_during_market(self):
        """Test should trade during market hours."""
        dt = datetime(2024, 1, 16, 12, 0, tzinfo=MarketHours.ET)
        can_trade, reason = MarketHours.should_trade("AAPL", dt=dt)
        assert can_trade is True
        assert "open" in reason.lower()

    def test_should_not_trade_when_closed(self):
        """Test should not trade when market closed."""
        dt = datetime(2024, 1, 16, 17, 0, tzinfo=MarketHours.ET)
        can_trade, reason = MarketHours.should_trade("AAPL", dt=dt)
        assert can_trade is False
        assert "closed" in reason.lower()

    def test_should_not_trade_near_close(self):
        """Test should not trade near market close."""
        dt = datetime(2024, 1, 16, 15, 50, tzinfo=MarketHours.ET)  # 10 min before
        can_trade, reason = MarketHours.should_trade(
            "AAPL", min_minutes_before_close=15, dt=dt
        )
        assert can_trade is False
        assert "minutes" in reason.lower()

    def test_crypto_should_trade_always(self):
        """Test crypto should always trade."""
        dt = datetime(2024, 1, 20, 2, 0, tzinfo=MarketHours.ET)  # Weekend
        can_trade, reason = MarketHours.should_trade("BTC/USD", dt=dt)
        assert can_trade is True
        assert "24/7" in reason


class TestGetMarketStatus:
    """Tests for get_market_status method."""

    @freeze_time("2024-01-16 12:00:00", tz_offset=-5)
    def test_market_status_structure(self):
        """Test market status returns correct structure."""
        result = MarketHours.get_market_status(["AAPL", "TSLA", "BTC/USD"])

        assert "current_time_et" in result
        assert "stock_market" in result
        assert "crypto_market" in result

    @freeze_time("2024-01-16 12:00:00", tz_offset=-5)
    def test_market_status_counts(self):
        """Test market status counts symbols correctly."""
        result = MarketHours.get_market_status(["AAPL", "TSLA", "BTC/USD", "ETH/USD"])

        assert result["stock_market"]["symbols_count"] == 2
        assert result["crypto_market"]["symbols_count"] == 2

    def test_market_status_open_state(self):
        """Test market status shows correct open state during market hours."""
        # Use explicit datetime instead of freeze_time for clarity
        dt = datetime(2024, 1, 16, 12, 0, tzinfo=MarketHours.ET)  # Tuesday noon

        # Mock the now_et method to return our test time
        with patch.object(MarketHours, 'now_et', return_value=dt):
            result = MarketHours.get_market_status(["AAPL"])

            assert result["stock_market"]["is_open"] is True
            assert result["crypto_market"]["is_open"] is True


class TestHolidays:
    """Tests for holiday handling."""

    def test_holidays_include_major_days(self):
        """Test that major holidays are included."""
        holidays = MarketHours.HOLIDAYS

        # Check some 2024 holidays
        assert "2024-12-25" in holidays  # Christmas
        assert "2024-11-28" in holidays  # Thanksgiving
        assert "2024-07-04" in holidays  # Independence Day

        # Check 2025 holidays
        assert "2025-01-01" in holidays  # New Year's

    def test_holiday_detection(self):
        """Test market is closed on specific holidays."""
        # MLK Day 2024
        dt = datetime(2024, 1, 15, 12, 0, tzinfo=MarketHours.ET)
        assert MarketHours.is_stock_market_open(dt) is False
