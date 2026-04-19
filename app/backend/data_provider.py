"""Unified data interface merging cache + REST backfill + real-time aggregation."""

import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from bar_aggregator import BarAggregator
from data_cache import DataCache
from data_fetcher import MassiveDataFetcher

logger = logging.getLogger(__name__)

# Massive REST API multiplier/span for each timeframe
TIMEFRAME_PARAMS = {
    "1min":  (1, "minute"),
    "5min":  (5, "minute"),
    "15min": (15, "minute"),
    "1h":    (1, "hour"),
    "4h":    (4, "hour"),
    "1d":    (1, "day"),
}


class DataProvider:
    """
    Unified data interface for the Alfi agent.

    Merges:
    - In-memory stock_data (already loaded by the backend at startup)
    - SQLite cache (fast, local)
    - Massive REST API (backfill)
    - Real-time bar aggregator (live trades -> bars)
    - Live prices from WebSocket trades
    - FeatureEngine for computed technical indicators
    """

    def __init__(
        self,
        fetcher: MassiveDataFetcher,
        cache: Optional[DataCache] = None,
        aggregator: Optional[BarAggregator] = None,
        stock_data: Optional[Dict[str, pd.DataFrame]] = None,
        live_prices: Optional[Dict[str, float]] = None,
        trade_lock: Optional[threading.Lock] = None,
        feature_engine: Optional[Any] = None,
    ):
        self.fetcher = fetcher
        self.cache = cache or DataCache()
        self.aggregator = aggregator
        self.stock_data = stock_data or {}
        self.live_prices = live_prices or {}
        self.trade_lock = trade_lock
        self.feature_engine = feature_engine

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        days_back: int = 100,
        use_cache: bool = True,
    ) -> List[Dict]:
        """
        Get historical bars, using cache when possible.

        Falls back to Massive REST if cache is empty/stale.
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Try cache first
        if use_cache:
            cached = self.cache.get_bars(symbol, timeframe, start_date, end_date)
            if cached:
                logger.debug(f"Cache hit: {symbol}/{timeframe} ({len(cached)} bars)")
                return cached

        # Fetch from Massive
        bars = self._fetch_from_massive(symbol, timeframe, start_date, end_date)

        # Store in cache
        if bars and use_cache:
            self.cache.store_bars(symbol, timeframe, bars)

        return bars

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = None,
        days_back: int = 30,
    ) -> Dict[str, List[Dict]]:
        """Get bars across multiple timeframes for a symbol."""
        if timeframes is None:
            timeframes = ["1d", "1h", "5min"]

        result = {}
        for tf in timeframes:
            bars = self.get_bars(symbol, tf, days_back)
            if bars:
                result[tf] = bars

        return result

    async def get_current_snapshots(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        lookback_bars: int = 100,
    ) -> Dict[str, Dict]:
        """
        Get current market snapshots for a list of symbols.

        Args:
            symbols: List of ticker symbols.
            timeframe: Bar timeframe ("1d", "5min", etc.).
            lookback_bars: Number of historical bars to include.

        Uses in-memory stock_data for daily bars, BarAggregator for intraday,
        then falls back to cache/REST. Overlays live prices from WebSocket.
        """
        snapshots = {}
        bars_key = f"bars_{timeframe}"

        for symbol in symbols:
            try:
                bars_list = None

                if timeframe == "1d":
                    # Daily bars: use in-memory stock_data first
                    if symbol in self.stock_data:
                        df = self.stock_data[symbol]
                        bars_list = self._df_to_bar_dicts(df, symbol)
                    if not bars_list:
                        bars_list = self.get_bars(symbol, "1d", days_back=lookback_bars)
                else:
                    # Intraday bars: try BarAggregator's completed bars first
                    if self.aggregator:
                        completed = self.aggregator.get_completed_bars(symbol, timeframe)
                        if completed and len(completed) >= 10:
                            bars_list = completed

                    # Fall back to cache/REST for intraday
                    if not bars_list:
                        # For 5-min bars, fetch ~5 trading days to get enough bars
                        days_needed = max(5, lookback_bars // 78 + 1)
                        bars_list = self.get_bars(symbol, timeframe, days_back=days_needed)

                if not bars_list:
                    continue

                # Trim to lookback_bars
                bars_list = bars_list[-lookback_bars:]

                latest = bars_list[-1]
                current_price = latest.get("close", 0)

                # Overlay live price from WebSocket if available
                if self.live_prices and self.trade_lock:
                    with self.trade_lock:
                        live = self.live_prices.get(symbol)
                        if live:
                            current_price = live

                snapshot = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "current_price": current_price,
                    "open": latest.get("open", 0),
                    "high": latest.get("high", 0),
                    "low": latest.get("low", 0),
                    "volume": latest.get("volume", 0),
                    bars_key: bars_list,
                }

                # Add real-time bar if available
                if self.aggregator:
                    rt_tf = timeframe if timeframe != "1d" else "1min"
                    active = self.aggregator.get_active_bar(symbol, rt_tf)
                    if active:
                        snapshot["real_time_bar"] = active

                snapshots[symbol] = snapshot

            except Exception as e:
                logger.error(f"Failed to build snapshot for {symbol}: {e}")

        # Enrich snapshots with computed indicators
        if self.feature_engine and snapshots:
            snapshots = self.feature_engine.compute_features_batch(snapshots)

        return snapshots

    @staticmethod
    def _df_to_bar_dicts(df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Convert a pandas DataFrame to list of bar dicts."""
        bars = []
        for _, row in df.iterrows():
            bars.append({
                "symbol": symbol,
                "timeframe": "1d",
                "timestamp": str(row.get("date", "")),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            })
        return bars

    def _fetch_from_massive(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> List[Dict]:
        """Fetch bars from Massive REST API with timeframe support."""
        params = TIMEFRAME_PARAMS.get(timeframe)
        if not params:
            logger.error(f"Unknown timeframe: {timeframe}")
            return []

        multiplier, span = params

        # For crypto, use the Massive crypto endpoint format
        ticker = symbol  # e.g. "AAPL" or "X:BTCUSD"

        try:
            import requests
            from config import BASE_URL, API_KEY

            url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{span}/{start_date}/{end_date}"
            response = requests.get(url, params={"apikey": API_KEY, "adjusted": "true", "limit": 5000})

            if response.status_code != 200:
                logger.warning(f"Massive API error for {symbol}/{timeframe}: HTTP {response.status_code}")
                return []

            results = response.json().get("results", [])
            bars = []
            for r in results:
                bars.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.fromtimestamp(r["t"] / 1000).isoformat(),
                    "open": r.get("o", 0),
                    "high": r.get("h", 0),
                    "low": r.get("l", 0),
                    "close": r.get("c", 0),
                    "volume": r.get("v", 0),
                    "vwap": r.get("vw", 0),
                    "n_trades": r.get("n", 0),
                })

            logger.info(f"Fetched {len(bars)} bars for {symbol}/{timeframe}")
            return bars

        except Exception as e:
            logger.error(f"Error fetching {symbol}/{timeframe} from Massive: {e}")
            return []
