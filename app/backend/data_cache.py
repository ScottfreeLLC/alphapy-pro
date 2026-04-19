"""SQLite-backed cache for historical bar data."""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "cache", "bars.db"
)


class DataCache:
    """SQLite cache to avoid re-fetching historical bars from Massive."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bars (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    vwap REAL,
                    n_trades INTEGER,
                    fetched_at TEXT,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bars_symbol_tf
                ON bars (symbol, timeframe, timestamp)
            """)

    def store_bars(self, symbol: str, timeframe: str, bars: List[Dict]):
        """Store bar data in the cache."""
        if not bars:
            return

        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO bars
                   (symbol, timeframe, timestamp, open, high, low, close, volume, vwap, n_trades, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        symbol,
                        timeframe,
                        bar.get("timestamp", bar.get("date", "")),
                        bar.get("open", 0),
                        bar.get("high", 0),
                        bar.get("low", 0),
                        bar.get("close", 0),
                        bar.get("volume", 0),
                        bar.get("vwap", 0),
                        bar.get("n_trades", 0),
                        now,
                    )
                    for bar in bars
                ],
            )
        logger.debug(f"Cached {len(bars)} bars for {symbol}/{timeframe}")

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """Retrieve cached bars."""
        query = "SELECT * FROM bars WHERE symbol = ? AND timeframe = ?"
        params: list = [symbol, timeframe]

        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "symbol": r["symbol"],
                "timeframe": r["timeframe"],
                "timestamp": r["timestamp"],
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
                "vwap": r["vwap"],
                "n_trades": r["n_trades"],
            }
            for r in rows
        ]

    def has_data(self, symbol: str, timeframe: str, start: str, end: str) -> bool:
        """Check if we have cached data for a given range."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT COUNT(*) FROM bars
                   WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ?""",
                (symbol, timeframe, start, end),
            ).fetchone()
        return row[0] > 0

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        """Get the most recent cached timestamp for a symbol/timeframe."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(timestamp) FROM bars WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            ).fetchone()
        return row[0] if row and row[0] else None

    def clear(self, symbol: str = None, timeframe: str = None):
        """Clear cached data, optionally filtered."""
        query = "DELETE FROM bars WHERE 1=1"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, params)
