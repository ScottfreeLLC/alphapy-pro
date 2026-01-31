"""Alpaca API client for order execution."""

import os
import logging
from typing import Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Client for Alpaca Trading API.

    Used for order execution only (not market data).
    Supports both stocks and crypto via paper or live trading.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
    ):
        """Initialize Alpaca client.

        Args:
            api_key: Alpaca API key. Reads from ALPACA_API_KEY if not provided.
            api_secret: Alpaca API secret. Reads from ALPACA_API_SECRET if not provided.
            paper: If True, use paper trading (default). Set to False for live.
        """
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca credentials required. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET environment variables."
            )

        # Check paper mode from env if not explicitly set
        paper_env = os.environ.get("ALPACA_PAPER", "true").lower()
        self.paper = paper if paper is not None else (paper_env == "true")

        # Lazy import to avoid dependency issues
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
            StopOrderRequest,
            StopLimitOrderRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

        self._TradingClient = TradingClient
        self._MarketOrderRequest = MarketOrderRequest
        self._LimitOrderRequest = LimitOrderRequest
        self._StopOrderRequest = StopOrderRequest
        self._StopLimitOrderRequest = StopLimitOrderRequest
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self._OrderType = OrderType

        self.client = TradingClient(
            self.api_key,
            self.api_secret,
            paper=self.paper,
        )

        mode = "PAPER" if self.paper else "LIVE"
        logger.info(f"Alpaca client initialized in {mode} mode")

    def get_account(self) -> dict:
        """Get account information.

        Returns:
            Dictionary with equity, buying power, etc.
        """
        account = self.client.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "last_equity": float(account.last_equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "initial_margin": float(account.initial_margin),
            "maintenance_margin": float(account.maintenance_margin),
            "daytrade_count": account.daytrade_count,
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions.

        Returns:
            List of position dictionaries.
        """
        positions = self.client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value if hasattr(p.side, 'value') else str(p.side),
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price),
                "change_today": float(p.change_today) if p.change_today else 0,
            }
            for p in positions
        ]

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol.

        Args:
            symbol: Stock or crypto symbol

        Returns:
            Position dictionary or None if no position.
        """
        try:
            p = self.client.get_open_position(symbol.upper())
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value if hasattr(p.side, 'value') else str(p.side),
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price),
            }
        except Exception:
            return None

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> dict:
        """Submit an order.

        Args:
            symbol: Stock or crypto symbol
            side: "buy" or "sell"
            qty: Number of shares/units
            order_type: "market", "limit", "stop", or "stop_limit"
            limit_price: Limit price (required for limit/stop_limit orders)
            stop_price: Stop price (required for stop/stop_limit orders)
            time_in_force: "day", "gtc", "ioc", or "fok"

        Returns:
            Order confirmation dictionary.
        """
        symbol = symbol.upper()
        order_side = self._OrderSide.BUY if side.lower() == "buy" else self._OrderSide.SELL

        # Map time in force
        tif_map = {
            "day": self._TimeInForce.DAY,
            "gtc": self._TimeInForce.GTC,
            "ioc": self._TimeInForce.IOC,
            "fok": self._TimeInForce.FOK,
        }
        tif = tif_map.get(time_in_force.lower(), self._TimeInForce.DAY)

        # Build order request based on type
        if order_type.lower() == "market":
            request = self._MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
        elif order_type.lower() == "limit":
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            request = self._LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
            )
        elif order_type.lower() == "stop":
            if stop_price is None:
                raise ValueError("stop_price required for stop orders")
            request = self._StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                stop_price=stop_price,
            )
        elif order_type.lower() == "stop_limit":
            if limit_price is None or stop_price is None:
                raise ValueError("Both limit_price and stop_price required for stop_limit orders")
            request = self._StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
                stop_price=stop_price,
            )
        else:
            raise ValueError(f"Invalid order type: {order_type}")

        # Submit order
        order = self.client.submit_order(request)

        logger.info(
            f"Order submitted: {side.upper()} {qty} {symbol} @ {order_type} "
            f"(ID: {order.id})"
        )

        return {
            "id": str(order.id),
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
            "qty": float(order.qty) if order.qty else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "type": order.type.value if hasattr(order.type, 'value') else str(order.type),
            "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "created_at": str(order.created_at),
        }

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation status.
        """
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return {"status": "cancelled", "order_id": order_id}
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {"status": "error", "order_id": order_id, "error": str(e)}

    def get_orders(self, status: str = "open") -> list[dict]:
        """Get orders by status.

        Args:
            status: "open", "closed", or "all"

        Returns:
            List of order dictionaries.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }

        request = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.OPEN))
        orders = self.client.get_orders(request)

        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": o.side.value if hasattr(o.side, 'value') else str(o.side),
                "qty": float(o.qty) if o.qty else None,
                "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                "type": o.type.value if hasattr(o.type, 'value') else str(o.type),
                "status": o.status.value if hasattr(o.status, 'value') else str(o.status),
                "created_at": str(o.created_at),
            }
            for o in orders
        ]

    def close_position(self, symbol: str) -> dict:
        """Close all shares of a position.

        Args:
            symbol: Symbol to close

        Returns:
            Order confirmation for the closing trade.
        """
        try:
            order = self.client.close_position(symbol.upper())
            logger.info(f"Position closed: {symbol}")
            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                "qty": float(order.qty) if order.qty else None,
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
            }
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    def close_all_positions(self) -> list[dict]:
        """Close all open positions.

        Returns:
            List of closing order confirmations.
        """
        try:
            orders = self.client.close_all_positions(cancel_orders=True)
            logger.info(f"All positions closed: {len(orders)} orders")
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "status": o.status.value if hasattr(o.status, 'value') else str(o.status),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return [{"status": "error", "error": str(e)}]
