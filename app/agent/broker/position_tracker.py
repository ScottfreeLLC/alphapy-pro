"""Position tracking and P&L calculation, synced with Alpaca."""

import logging
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional

from .alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "trades", "positions.db",
)


class PositionTracker:
    """
    Tracks positions and P&L.

    Alpaca is the source of truth for current positions.
    SQLite stores trade history for performance analysis.
    """

    def __init__(self, alpaca: AlpacaClient, db_path: str = DEFAULT_DB_PATH):
        self.alpaca = alpaca
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    skill_name TEXT,
                    signal_id TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    duration_hours REAL
                )
            """)

    def get_positions(self) -> List[Dict]:
        """Get current positions from Alpaca."""
        return self.alpaca.get_positions()

    def get_portfolio_summary(self) -> Dict:
        """Get account + positions summary."""
        account = self.alpaca.get_account() or {}
        positions = self.get_positions()

        total_unrealized_pl = sum(p.get("unrealized_pl", 0) for p in positions)
        total_market_value = sum(p.get("market_value", 0) for p in positions)
        equity = account.get("equity", 0)

        exposure_pct = (total_market_value / equity * 100) if equity > 0 else 0

        return {
            "equity": equity,
            "cash": account.get("cash", 0),
            "buying_power": account.get("buying_power", 0),
            "portfolio_value": account.get("portfolio_value", 0),
            "positions_count": len(positions),
            "total_unrealized_pl": round(total_unrealized_pl, 2),
            "total_market_value": round(total_market_value, 2),
            "exposure_pct": round(exposure_pct, 2),
            "positions": positions,
        }

    def record_closed_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        skill_name: str = "",
        signal_id: str = "",
        entry_time: str = "",
        exit_time: str = "",
    ):
        """Record a completed trade for performance tracking."""
        pnl = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        if side == "short":
            pnl_pct = -pnl_pct

        duration = 0.0
        if entry_time and exit_time:
            try:
                t1 = datetime.fromisoformat(entry_time)
                t2 = datetime.fromisoformat(exit_time)
                duration = (t2 - t1).total_seconds() / 3600
            except ValueError:
                pass

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trade_history
                   (symbol, side, qty, entry_price, exit_price, pnl, pnl_pct,
                    skill_name, signal_id, entry_time, exit_time, duration_hours)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, side, qty, entry_price, exit_price,
                 round(pnl, 2), round(pnl_pct, 4),
                 skill_name, signal_id, entry_time, exit_time, round(duration, 2)),
            )
        logger.info(f"Recorded trade: {side} {symbol} P&L={pnl:.2f} ({pnl_pct:.2f}%)")

    def get_trade_history(self, limit: int = 100, skill_name: str = None) -> List[Dict]:
        """Get trade history, optionally filtered by skill."""
        query = "SELECT * FROM trade_history"
        params: list = []
        if skill_name:
            query += " WHERE skill_name = ?"
            params.append(skill_name)
        query += " ORDER BY exit_time DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_trade_count(self) -> int:
        """Get total number of recorded trades."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM trade_history").fetchone()
        return row[0] if row else 0
