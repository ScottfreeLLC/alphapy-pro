"""Aggregate raw trades into OHLCV candlestick bars."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single OHLCV candlestick bar."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    trade_count: int = 0
    complete: bool = False

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "trade_count": self.trade_count,
            "complete": self.complete,
        }


# Map timeframe strings to seconds
TIMEFRAME_SECONDS = {
    "1min": 60,
    "5min": 300,
    "15min": 900,
    "1h": 3600,
    "4h": 14400,
}


class BarAggregator:
    """Aggregates incoming trades into OHLCV bars at configurable intervals."""

    def __init__(
        self,
        timeframes: List[str] = None,
        on_bar_complete: Optional[Callable[[Bar], None]] = None,
    ):
        self.timeframes = timeframes or ["1min", "5min"]
        self.on_bar_complete = on_bar_complete

        # Active (in-progress) bars: {(symbol, timeframe): Bar}
        self._active_bars: Dict[tuple, Bar] = {}
        # Completed bars buffer: {symbol: [Bar]}
        self._completed_bars: Dict[str, List[Bar]] = defaultdict(list)

    def process_trade(self, symbol: str, price: float, size: int, timestamp_ms: int):
        """
        Process an incoming trade tick.

        Args:
            symbol: Ticker symbol
            price: Trade price
            size: Trade size (shares/units)
            timestamp_ms: Trade timestamp in milliseconds
        """
        trade_time = datetime.fromtimestamp(timestamp_ms / 1000)

        for tf in self.timeframes:
            interval = TIMEFRAME_SECONDS.get(tf, 60)
            bar_key = (symbol, tf)

            # Calculate bar boundary
            epoch = int(trade_time.timestamp())
            bar_start_epoch = (epoch // interval) * interval
            bar_start = datetime.fromtimestamp(bar_start_epoch)

            active = self._active_bars.get(bar_key)

            # If there's an active bar from a different period, complete it
            if active and active.timestamp != bar_start:
                self._complete_bar(active)
                active = None

            if active is None:
                # Start a new bar
                active = Bar(
                    symbol=symbol,
                    timeframe=tf,
                    timestamp=bar_start,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=size,
                    trade_count=1,
                )
                self._active_bars[bar_key] = active
            else:
                # Update existing bar
                active.high = max(active.high, price)
                active.low = min(active.low, price)
                active.close = price
                active.volume += size
                active.trade_count += 1

    def _complete_bar(self, bar: Bar):
        """Mark a bar as complete and notify listeners."""
        bar.complete = True
        self._completed_bars[bar.symbol].append(bar)

        # Keep only last 500 bars per symbol
        if len(self._completed_bars[bar.symbol]) > 500:
            self._completed_bars[bar.symbol] = self._completed_bars[bar.symbol][-500:]

        if self.on_bar_complete:
            self.on_bar_complete(bar)

        logger.debug(
            f"Bar complete: {bar.symbol} {bar.timeframe} "
            f"O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume}"
        )

    def flush(self):
        """Force-complete all active bars (e.g. at market close)."""
        for bar in list(self._active_bars.values()):
            self._complete_bar(bar)
        self._active_bars.clear()

    def get_completed_bars(self, symbol: str, timeframe: str = None, limit: int = 100) -> List[Dict]:
        """Get completed bars for a symbol, optionally filtered by timeframe."""
        bars = self._completed_bars.get(symbol, [])
        if timeframe:
            bars = [b for b in bars if b.timeframe == timeframe]
        return [b.to_dict() for b in bars[-limit:]]

    def get_active_bar(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get the current in-progress bar."""
        bar = self._active_bars.get((symbol, timeframe))
        return bar.to_dict() if bar else None
