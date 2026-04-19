"""Order lifecycle management — links orders to signals and tracks state."""

import logging
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional

from ..core.signal import TradeSignal, SignalDirection, SignalStatus
from .alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "trades", "orders.db",
)


class OrderManager:
    """
    Manages the lifecycle of orders:
      Signal → Order Submitted → Filled/Cancelled/Rejected

    Uses bracket orders for automatic stop-loss / take-profit.
    """

    def __init__(self, alpaca: AlpacaClient, db_path: str = DEFAULT_DB_PATH):
        self.alpaca = alpaca
        self.db_path = db_path
        # In-memory mapping: signal_id -> order_id
        self._signal_to_order: Dict[str, str] = {}
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    signal_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL,
                    order_type TEXT,
                    limit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT,
                    skill_name TEXT,
                    filled_price REAL,
                    filled_qty REAL,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

    def submit_order(self, signal: TradeSignal) -> Optional[Dict]:
        """
        Submit an order for a trade signal.

        Uses bracket orders when both stop_loss and take_profit are set.
        """
        if not self.alpaca.connected:
            logger.error("Alpaca client not connected")
            return None

        side = "buy" if signal.direction == SignalDirection.LONG else "sell"

        # Calculate quantity from account equity and position size
        account = self.alpaca.get_account()
        if not account:
            return None

        equity = account.get("equity", 0)
        position_value = equity * signal.position_size_pct
        qty = int(position_value / signal.entry_price) if signal.entry_price > 0 else 0

        if qty <= 0:
            logger.warning(f"Calculated qty is 0 for {signal.symbol} (equity={equity}, size_pct={signal.position_size_pct})")
            return None

        # Use bracket order if stop/target are set
        if signal.stop_loss > 0 and signal.take_profit > 0:
            order = self.alpaca.submit_bracket_order(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                limit_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
        else:
            order = self.alpaca.submit_limit_order(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                limit_price=signal.entry_price,
            )

        if order:
            self._signal_to_order[signal.id] = order["id"]
            self._persist_order(order, signal)
            signal.status = SignalStatus.EXECUTED
            logger.info(f"Order submitted: {order['id']} for signal {signal.id} ({signal.symbol})")
        else:
            signal.status = SignalStatus.CANCELLED
            logger.error(f"Failed to submit order for signal {signal.id}")

        return order

    def get_order_for_signal(self, signal_id: str) -> Optional[str]:
        """Get the order ID linked to a signal."""
        return self._signal_to_order.get(signal_id)

    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get order history from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _persist_order(self, order: Dict, signal: TradeSignal):
        """Save order to SQLite."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO orders
                   (order_id, signal_id, symbol, side, qty, order_type,
                    limit_price, stop_loss, take_profit, status, skill_name,
                    filled_price, filled_qty, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order.get("id"),
                    signal.id,
                    signal.symbol,
                    signal.direction.value,
                    order.get("qty"),
                    order.get("type"),
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    order.get("status"),
                    signal.skill_name,
                    order.get("filled_avg_price"),
                    order.get("filled_qty"),
                    order.get("created_at", now),
                    now,
                ),
            )
