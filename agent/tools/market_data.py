"""Market Data tool for fetching real-time bars from Polygon."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .base import Tool
from alphapy.data_sources import PolygonDataSource

logger = logging.getLogger(__name__)


@dataclass
class MarketDataTool(Tool):
    """Tool for fetching market data from Polygon.

    Uses the consolidated PolygonDataSource from AlphaPy.
    Returns Polars DataFrames.
    """

    name: str = "get_market_data"
    description: str = """
Retrieves market data (OHLCV bars) for specified symbols from Polygon.
Supports stocks (e.g., AAPL, TSLA) and crypto (e.g., BTC/USD, ETH/USD).
Returns bar data with open, high, low, close, volume columns.
Use this to get current market conditions before generating signals.
"""
    input_schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of symbols to fetch (e.g., ['AAPL', 'BTC/USD'])",
            },
            "timeframe": {
                "type": "string",
                "enum": ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"],
                "default": "5Min",
                "description": "Bar timeframe",
            },
            "lookback_bars": {
                "type": "integer",
                "default": 100,
                "minimum": 10,
                "maximum": 1000,
                "description": "Number of historical bars to fetch",
            },
        },
        "required": ["symbols"],
    })

    _client: PolygonDataSource = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the Polygon data source."""
        if self._client is None:
            self._client = PolygonDataSource()

    async def execute(
        self,
        symbols: list[str],
        timeframe: str = "5Min",
        lookback_bars: int = 100,
    ) -> str:
        """Fetch market data for the specified symbols.

        Returns:
            JSON string with bar data for each symbol.
        """
        logger.info(f"Fetching {timeframe} bars for {len(symbols)} symbols")

        try:
            bars = self._client.get_bars(
                symbols=symbols,
                timeframe=timeframe,
                lookback=lookback_bars,
            )

            # Convert Polars DataFrames to JSON-serializable format
            result = {}
            for symbol, df in bars.items():
                if not df.is_empty():
                    result[symbol] = {
                        "bars": df.to_dicts(),
                        "count": len(df),
                        "latest_close": float(df["close"][-1]),
                        "latest_time": str(df["datetime"][-1]),
                    }
                else:
                    result[symbol] = {
                        "bars": [],
                        "count": 0,
                        "latest_close": None,
                        "latest_time": None,
                    }

            # Add summary
            result["_summary"] = {
                "symbols_requested": len(symbols),
                "symbols_returned": len(result) - 1,
                "timeframe": timeframe,
                "lookback_bars": lookback_bars,
            }

            return json.dumps(result, default=str)

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return json.dumps({
                "error": str(e),
                "symbols": symbols,
            })
