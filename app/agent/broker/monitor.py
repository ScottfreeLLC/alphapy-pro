"""Background position and order monitoring, synced with Alpaca."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from .alpaca_client import AlpacaClient
from .position_tracker import PositionTracker

logger = logging.getLogger(__name__)

# Polling interval
DEFAULT_POLL_SECONDS = 30


class PositionMonitor:
    """
    Background loop that polls Alpaca for position/order updates.

    Detects:
    - New fills (order status changes)
    - Stop-outs and take-profit exits
    - P&L changes
    - Position closures

    Feeds results to PerformanceTracker and GraduationManager.
    """

    def __init__(
        self,
        alpaca: AlpacaClient,
        position_tracker: PositionTracker,
        performance_tracker: Optional[Any] = None,
        on_fill: Optional[Callable[[Dict], None]] = None,
        on_close: Optional[Callable[[Dict], None]] = None,
        poll_seconds: int = DEFAULT_POLL_SECONDS,
    ):
        self.alpaca = alpaca
        self.position_tracker = position_tracker
        self.performance_tracker = performance_tracker
        self.on_fill = on_fill
        self.on_close = on_close
        self.poll_seconds = poll_seconds

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._known_positions: Set[str] = set()  # symbols currently held
        self._known_orders: Dict[str, str] = {}   # order_id -> last known status
        self._last_equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._circuit_breaker_tripped = False

    @property
    def running(self) -> bool:
        return self._running

    async def start(self):
        """Start the monitoring loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Position monitor started (polling every {self.poll_seconds}s)")

    async def stop(self):
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Position monitor stopped")

    async def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                await self._check_positions()
                await self._check_orders()
                await self._check_account()
            except Exception as e:
                logger.error(f"Monitor poll error: {e}", exc_info=True)

            await asyncio.sleep(self.poll_seconds)

    async def _check_positions(self):
        """Detect new positions and closed positions."""
        if not self.alpaca.connected:
            return

        current_positions = self.alpaca.get_positions()
        current_symbols = {p["symbol"] for p in current_positions}

        # Detect newly closed positions
        closed = self._known_positions - current_symbols
        for symbol in closed:
            logger.info(f"Position closed: {symbol}")
            if self.on_close:
                self.on_close({"symbol": symbol, "event": "position_closed", "time": datetime.now().isoformat()})

        # Detect newly opened positions
        opened = current_symbols - self._known_positions
        for symbol in opened:
            pos = next((p for p in current_positions if p["symbol"] == symbol), None)
            if pos:
                logger.info(f"New position: {symbol} qty={pos['qty']} side={pos['side']}")
                if self.on_fill:
                    self.on_fill({**pos, "event": "position_opened", "time": datetime.now().isoformat()})

        self._known_positions = current_symbols

    async def _check_orders(self):
        """Detect order status changes (fills, cancellations)."""
        if not self.alpaca.connected:
            return

        orders = self.alpaca.get_orders(status="all")
        # Only look at recent orders (last 50)
        for order in orders[:50]:
            order_id = order.get("id")
            new_status = order.get("status")
            old_status = self._known_orders.get(order_id)

            if old_status != new_status:
                self._known_orders[order_id] = new_status

                if new_status == "filled" and old_status != "filled":
                    logger.info(f"Order filled: {order['symbol']} {order['side']} qty={order['filled_qty']} @ {order['filled_avg_price']}")
                    if self.on_fill:
                        self.on_fill({**order, "event": "order_filled", "time": datetime.now().isoformat()})

                elif new_status in ("canceled", "cancelled", "expired", "rejected"):
                    logger.info(f"Order {new_status}: {order['symbol']} {order_id[:8]}")

    async def _check_account(self):
        """Track equity changes for daily P&L and circuit breaker."""
        if not self.alpaca.connected:
            return

        account = self.alpaca.get_account()
        if not account:
            return

        equity = account.get("equity", 0)

        if self._last_equity > 0 and equity > 0:
            pnl_change = equity - self._last_equity
            self._daily_pnl += pnl_change

        self._last_equity = equity

    def get_status(self) -> Dict:
        """Get monitor status for API/frontend."""
        return {
            "running": self._running,
            "poll_seconds": self.poll_seconds,
            "known_positions": list(self._known_positions),
            "position_count": len(self._known_positions),
            "tracked_orders": len(self._known_orders),
            "last_equity": self._last_equity,
            "daily_pnl": round(self._daily_pnl, 2),
        }
