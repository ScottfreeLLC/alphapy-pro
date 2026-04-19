"""Performance tracking: win rate, Sharpe, drawdown, per-skill metrics."""

import logging
import math
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "performance", "metrics.db",
)


class PerformanceTracker:
    """
    Tracks trade outcomes and computes key performance metrics.

    Metrics: win rate, Sharpe ratio, profit factor, max drawdown, expectancy.
    All data persisted to SQLite.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    skill_name TEXT,
                    signal_id TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    qty REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date TEXT PRIMARY KEY,
                    pnl REAL,
                    cumulative_pnl REAL,
                    equity REAL,
                    trade_count INTEGER
                )
            """)

    def record_trade(
        self,
        symbol: str,
        direction: str,
        skill_name: str,
        signal_id: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        entry_time: str = "",
        exit_time: str = "",
    ):
        """Record a completed trade."""
        if direction == "long":
            pnl = (exit_price - entry_price) * qty
            pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price else 0
        else:
            pnl = (entry_price - exit_price) * qty
            pnl_pct = (entry_price - exit_price) / entry_price * 100 if entry_price else 0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trades
                   (symbol, direction, skill_name, signal_id, entry_price, exit_price,
                    qty, pnl, pnl_pct, entry_time, exit_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, direction, skill_name, signal_id, entry_price, exit_price,
                 qty, round(pnl, 2), round(pnl_pct, 4), entry_time, exit_time),
            )
        logger.info(f"Recorded trade: {direction} {symbol} P&L=${pnl:.2f} ({pnl_pct:.2f}%)")

    def get_metrics(self) -> Dict:
        """Calculate all performance metrics."""
        trades = self._get_all_trades()
        if not trades:
            return self._empty_metrics()

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total = len(pnls)
        win_count = len(wins)
        win_rate = win_count / total if total > 0 else 0

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0

        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Sharpe ratio (annualized, assuming daily returns)
        pnl_pcts = [t["pnl_pct"] for t in trades]
        sharpe = self._calculate_sharpe(pnl_pcts)

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(pnls)

        # Equity curve
        cumulative = []
        running = 0
        for p in pnls:
            running += p
            cumulative.append(round(running, 2))

        return {
            "total_trades": total,
            "win_count": win_count,
            "loss_count": len(losses),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "expectancy": round(expectancy, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_pnl": round(sum(pnls), 2),
            "equity_curve": cumulative[-20:],  # Last 20 data points
        }

    def get_skill_metrics(self) -> Dict[str, Dict]:
        """Get per-skill performance breakdown."""
        trades = self._get_all_trades()
        if not trades:
            return {}

        by_skill: Dict[str, List] = {}
        for t in trades:
            skill = t.get("skill_name", "unknown")
            by_skill.setdefault(skill, []).append(t)

        result = {}
        for skill_name, skill_trades in by_skill.items():
            pnls = [t["pnl"] for t in skill_trades]
            wins = [p for p in pnls if p > 0]
            total = len(pnls)

            result[skill_name] = {
                "total_trades": total,
                "win_rate": round(len(wins) / total, 4) if total > 0 else 0,
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(sum(pnls) / total, 2) if total > 0 else 0,
            }

        return result

    def get_trade_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
        return row[0] if row else 0

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _get_all_trades(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trades ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]

    def _calculate_sharpe(self, pnl_pcts: List[float], risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe ratio from trade return percentages."""
        if len(pnl_pcts) < 2:
            return 0.0

        mean_return = sum(pnl_pcts) / len(pnl_pcts)
        variance = sum((r - mean_return) ** 2 for r in pnl_pcts) / (len(pnl_pcts) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        if std_dev == 0:
            return 0.0

        # Annualize assuming ~252 trading days
        sharpe = (mean_return - risk_free_rate) / std_dev * math.sqrt(252)
        return sharpe

    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Max drawdown as a percentage of peak equity."""
        if not pnls:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            if peak > 0:
                dd_pct = drawdown / peak * 100
                max_dd = max(max_dd, dd_pct)

        return max_dd

    def _empty_metrics(self) -> Dict:
        return {
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
            "total_pnl": 0,
            "equity_curve": [],
        }
