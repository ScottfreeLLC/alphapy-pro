"""Market hours utilities for trading scheduling."""

import logging
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class MarketHours:
    """Utility for checking market hours and trading windows.

    Handles both stock market hours (NYSE/NASDAQ) and crypto (24/7).
    """

    # US Eastern timezone
    ET = ZoneInfo("America/New_York")

    # NYSE/NASDAQ regular trading hours
    STOCK_OPEN = time(9, 30)  # 9:30 AM ET
    STOCK_CLOSE = time(16, 0)  # 4:00 PM ET

    # Pre-market and after-hours (for reference)
    PREMARKET_OPEN = time(4, 0)  # 4:00 AM ET
    AFTERHOURS_CLOSE = time(20, 0)  # 8:00 PM ET

    # US market holidays (2024-2025)
    # Update this list annually
    HOLIDAYS = {
        # 2024
        "2024-01-01",  # New Year's Day
        "2024-01-15",  # MLK Day
        "2024-02-19",  # Presidents Day
        "2024-03-29",  # Good Friday
        "2024-05-27",  # Memorial Day
        "2024-06-19",  # Juneteenth
        "2024-07-04",  # Independence Day
        "2024-09-02",  # Labor Day
        "2024-11-28",  # Thanksgiving
        "2024-12-25",  # Christmas
        # 2025
        "2025-01-01",  # New Year's Day
        "2025-01-20",  # MLK Day
        "2025-02-17",  # Presidents Day
        "2025-04-18",  # Good Friday
        "2025-05-26",  # Memorial Day
        "2025-06-19",  # Juneteenth
        "2025-07-04",  # Independence Day
        "2025-09-01",  # Labor Day
        "2025-11-27",  # Thanksgiving
        "2025-12-25",  # Christmas
        # 2026
        "2026-01-01",  # New Year's Day
        "2026-01-19",  # MLK Day
        "2026-02-16",  # Presidents Day
        "2026-04-03",  # Good Friday
        "2026-05-25",  # Memorial Day
        "2026-06-19",  # Juneteenth
        "2026-07-03",  # Independence Day (observed)
        "2026-09-07",  # Labor Day
        "2026-11-26",  # Thanksgiving
        "2026-12-25",  # Christmas
    }

    @classmethod
    def now_et(cls) -> datetime:
        """Get current time in Eastern Time."""
        return datetime.now(cls.ET)

    @classmethod
    def is_stock_market_open(cls, dt: Optional[datetime] = None) -> bool:
        """Check if the stock market is currently open.

        Args:
            dt: Datetime to check (defaults to now)

        Returns:
            True if market is open, False otherwise.
        """
        if dt is None:
            dt = cls.now_et()
        else:
            # Ensure timezone is ET
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=cls.ET)
            else:
                dt = dt.astimezone(cls.ET)

        # Check if it's a weekday
        if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if it's a holiday
        if dt.strftime("%Y-%m-%d") in cls.HOLIDAYS:
            return False

        # Check if within trading hours
        current_time = dt.time()
        return cls.STOCK_OPEN <= current_time < cls.STOCK_CLOSE

    @classmethod
    def is_crypto_market_open(cls) -> bool:
        """Check if crypto markets are open (always True - 24/7)."""
        return True

    @classmethod
    def is_market_open(cls, symbol: str, dt: Optional[datetime] = None) -> bool:
        """Check if market is open for a given symbol.

        Args:
            symbol: Stock or crypto symbol
            dt: Datetime to check (defaults to now)

        Returns:
            True if market is open for this symbol.
        """
        if "/" in symbol:  # Crypto (e.g., BTC/USD)
            return cls.is_crypto_market_open()
        else:
            return cls.is_stock_market_open(dt)

    @classmethod
    def minutes_until_close(cls, dt: Optional[datetime] = None) -> Optional[int]:
        """Get minutes until stock market close.

        Args:
            dt: Datetime to check (defaults to now)

        Returns:
            Minutes until close, or None if market is closed.
        """
        if dt is None:
            dt = cls.now_et()
        else:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=cls.ET)
            else:
                dt = dt.astimezone(cls.ET)

        if not cls.is_stock_market_open(dt):
            return None

        close_dt = dt.replace(
            hour=cls.STOCK_CLOSE.hour,
            minute=cls.STOCK_CLOSE.minute,
            second=0,
            microsecond=0,
        )
        delta = close_dt - dt
        return int(delta.total_seconds() / 60)

    @classmethod
    def minutes_until_open(cls, dt: Optional[datetime] = None) -> Optional[int]:
        """Get minutes until stock market opens.

        Args:
            dt: Datetime to check (defaults to now)

        Returns:
            Minutes until open, or None if market is already open.
        """
        if dt is None:
            dt = cls.now_et()
        else:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=cls.ET)
            else:
                dt = dt.astimezone(cls.ET)

        if cls.is_stock_market_open(dt):
            return None

        # If before market open today
        current_time = dt.time()
        if current_time < cls.STOCK_OPEN and dt.weekday() < 5:
            date_str = dt.strftime("%Y-%m-%d")
            if date_str not in cls.HOLIDAYS:
                open_dt = dt.replace(
                    hour=cls.STOCK_OPEN.hour,
                    minute=cls.STOCK_OPEN.minute,
                    second=0,
                    microsecond=0,
                )
                delta = open_dt - dt
                return int(delta.total_seconds() / 60)

        # Otherwise, need to find next trading day
        # This is a simplified version - doesn't account for all holidays
        return None

    @classmethod
    def should_trade(
        cls,
        symbol: str,
        min_minutes_before_close: int = 15,
        dt: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        """Check if we should execute trades for this symbol.

        Args:
            symbol: Stock or crypto symbol
            min_minutes_before_close: Don't enter new positions this close to close
            dt: Datetime to check (defaults to now)

        Returns:
            Tuple of (should_trade: bool, reason: str)
        """
        is_crypto = "/" in symbol

        if is_crypto:
            return True, "Crypto markets are 24/7"

        if not cls.is_stock_market_open(dt):
            return False, "Stock market is closed"

        minutes_left = cls.minutes_until_close(dt)
        if minutes_left is not None and minutes_left < min_minutes_before_close:
            return False, f"Only {minutes_left} minutes until close"

        return True, "Market is open"

    @classmethod
    def get_market_status(cls, symbols: list[str]) -> dict:
        """Get market status for a list of symbols.

        Args:
            symbols: List of symbols to check

        Returns:
            Dictionary with market status for each symbol type.
        """
        dt = cls.now_et()
        stock_open = cls.is_stock_market_open(dt)
        minutes_to_close = cls.minutes_until_close(dt)

        stocks = [s for s in symbols if "/" not in s]
        crypto = [s for s in symbols if "/" in s]

        return {
            "current_time_et": dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "stock_market": {
                "is_open": stock_open,
                "minutes_until_close": minutes_to_close,
                "symbols_count": len(stocks),
            },
            "crypto_market": {
                "is_open": True,
                "symbols_count": len(crypto),
            },
        }
