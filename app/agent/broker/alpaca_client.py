"""Alpaca SDK wrapper for paper/live trading."""

import logging
from typing import Dict, List, Optional

from ..config import AgentConfig

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        OrderStatus,
        OrderType,
        QueryOrderStatus,
        TimeInForce,
    )
    from alpaca.common.exceptions import APIError as AlpacaAPIError

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. Broker features disabled.")


class AlpacaClient:
    """
    Wraps alpaca-py SDK for order submission, position queries, and account info.

    Starts in paper mode by default.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._client: Optional["TradingClient"] = None

        if not ALPACA_AVAILABLE:
            logger.error("alpaca-py is not installed. Run: pip install alpaca-py")
            return

        if not config.alpaca_api_key or not config.alpaca_secret_key:
            logger.warning("Alpaca API keys not configured. Broker will not connect.")
            return

        self._client = TradingClient(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            paper=config.alpaca_paper,
        )
        mode = "PAPER" if config.alpaca_paper else "LIVE"
        logger.info(f"Alpaca client initialized ({mode} mode)")

    @property
    def connected(self) -> bool:
        return self._client is not None

    def get_account(self) -> Optional[Dict]:
        """Get account information."""
        if not self._client:
            return None
        try:
            account = self._client.get_account()
            return {
                "id": str(account.id),
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "status": str(account.status),
                "currency": str(account.currency),
                "pattern_day_trader": account.pattern_day_trader,
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> Optional[Dict]:
        """Submit a market order."""
        if not self._client:
            logger.error("Alpaca client not connected")
            return None

        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
            order = self._client.submit_order(request)
            return self._order_to_dict(order)

        except Exception as e:
            logger.error(f"Failed to submit market order for {symbol}: {e}")
            return None

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Optional[Dict]:
        """Submit a limit order."""
        if not self._client:
            return None

        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC

            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
            )
            order = self._client.submit_order(request)
            return self._order_to_dict(order)

        except Exception as e:
            logger.error(f"Failed to submit limit order for {symbol}: {e}")
            return None

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[Dict]:
        """Submit a bracket order (entry + stop loss + take profit)."""
        if not self._client:
            return None

        try:
            from alpaca.trading.requests import (
                LimitOrderRequest,
                StopLossRequest,
                TakeProfitRequest,
            )

            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price,
                order_class="bracket",
                take_profit=TakeProfitRequest(limit_price=take_profit),
                stop_loss=StopLossRequest(stop_price=stop_loss),
            )
            order = self._client.submit_order(request)
            return self._order_to_dict(order)

        except Exception as e:
            logger.error(f"Failed to submit bracket order for {symbol}: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        if not self._client:
            return []

        try:
            positions = self._client.get_all_positions()
            return [
                {
                    "symbol": str(p.symbol),
                    "qty": float(p.qty),
                    "side": str(p.side),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "change_today": float(p.change_today),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_orders(self, status: str = "open") -> List[Dict]:
        """Get orders by status."""
        if not self._client:
            return []

        try:
            query_status = QueryOrderStatus.OPEN if status == "open" else QueryOrderStatus.ALL
            request = GetOrdersRequest(status=query_status)
            orders = self._client.get_orders(request)
            return [self._order_to_dict(o) for o in orders]
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        if not self._client:
            return False

        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """Close a position by symbol."""
        if not self._client:
            return False

        try:
            self._client.close_position(symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return False

    def _order_to_dict(self, order) -> Dict:
        """Convert an Alpaca order object to a dict."""
        return {
            "id": str(order.id),
            "symbol": str(order.symbol),
            "side": str(order.side),
            "qty": str(order.qty),
            "type": str(order.type),
            "status": str(order.status),
            "limit_price": str(order.limit_price) if order.limit_price else None,
            "stop_price": str(order.stop_price) if order.stop_price else None,
            "filled_qty": str(order.filled_qty) if order.filled_qty else "0",
            "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
            "created_at": str(order.created_at),
            "submitted_at": str(order.submitted_at),
        }
