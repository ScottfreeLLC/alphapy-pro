"""Order Execution tool for placing trades via Alpaca."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import Tool
from ..utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


@dataclass
class OrderExecutionTool(Tool):
    """Tool for executing trades via Alpaca.

    Submits market and limit orders for stocks and crypto.
    """

    name: str = "execute_order"
    description: str = """
Submits a trading order to Alpaca.
Supports market, limit, stop, and stop_limit order types.
Works for both stocks and crypto.
Returns order confirmation with fill details.
IMPORTANT: Always check risk constraints before calling this tool.
"""
    input_schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol to trade (e.g., 'AAPL', 'BTC/USD')",
            },
            "side": {
                "type": "string",
                "enum": ["buy", "sell"],
                "description": "Order side",
            },
            "quantity": {
                "type": "number",
                "minimum": 0.0001,
                "description": "Number of shares/units to trade",
            },
            "order_type": {
                "type": "string",
                "enum": ["market", "limit", "stop", "stop_limit"],
                "default": "market",
                "description": "Order type",
            },
            "limit_price": {
                "type": "number",
                "description": "Limit price (required for limit/stop_limit orders)",
            },
            "stop_price": {
                "type": "number",
                "description": "Stop price (required for stop/stop_limit orders)",
            },
            "time_in_force": {
                "type": "string",
                "enum": ["day", "gtc", "ioc", "fok"],
                "default": "day",
                "description": "Time in force",
            },
        },
        "required": ["symbol", "side", "quantity"],
    })

    _client: AlpacaClient = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the Alpaca client."""
        if self._client is None:
            self._client = AlpacaClient()

    async def execute(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> str:
        """Execute a trade.

        Returns:
            JSON string with order confirmation.
        """
        try:
            logger.info(
                f"Executing order: {side.upper()} {quantity} {symbol} @ {order_type}"
            )

            result = self._client.submit_order(
                symbol=symbol,
                side=side,
                qty=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
            )

            return json.dumps({
                "success": True,
                "order": result,
                "message": f"Order submitted: {side.upper()} {quantity} {symbol}",
            })

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            })

    async def cancel(self, order_id: str) -> str:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            JSON string with cancellation status.
        """
        try:
            result = self._client.cancel_order(order_id)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "order_id": order_id,
            })

    async def close_position(self, symbol: str) -> str:
        """Close all shares of a position.

        Args:
            symbol: Symbol to close

        Returns:
            JSON string with closing order confirmation.
        """
        try:
            result = self._client.close_position(symbol)
            return json.dumps({
                "success": True,
                "order": result,
                "message": f"Position closed: {symbol}",
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "symbol": symbol,
            })
