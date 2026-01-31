"""Portfolio State tool for querying account and positions."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .base import Tool
from ..utils.alpaca_client import AlpacaClient
from alphapy.portfolio import LivePortfolio

logger = logging.getLogger(__name__)


@dataclass
class PortfolioStateTool(Tool):
    """Tool for querying portfolio state from Alpaca.

    Returns current account equity, positions, and daily P&L.
    """

    name: str = "get_portfolio_state"
    description: str = """
Returns current portfolio state including:
- Account equity, buying power, and cash
- Open positions with P&L
- Today's orders and fills
- Daily P&L and drawdown
Use this to understand current exposure before placing new trades.
"""
    input_schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "include_orders": {
                "type": "boolean",
                "default": False,
                "description": "Include open orders in response",
            },
            "include_history": {
                "type": "boolean",
                "default": False,
                "description": "Include today's filled orders",
            },
        },
    })

    _client: AlpacaClient = field(default=None, repr=False)
    _portfolio: LivePortfolio = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the Alpaca client and LivePortfolio."""
        if self._client is None:
            self._client = AlpacaClient()
        if self._portfolio is None:
            self._portfolio = LivePortfolio(broker=self._client)

    async def execute(
        self,
        include_orders: bool = False,
        include_history: bool = False,
    ) -> str:
        """Query portfolio state.

        Returns:
            JSON string with account and position data.
        """
        try:
            # Sync portfolio with broker
            self._portfolio.sync()

            # Get account info for additional metrics
            account = self._client.get_account()

            # Daily P&L (from account)
            daily_pl = account["equity"] - account["last_equity"]
            daily_pl_pct = (daily_pl / account["last_equity"]) * 100 if account["last_equity"] > 0 else 0

            # Get positions as list of dicts from LivePortfolio
            position_details = [
                {
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "side": pos.side,
                    "market_value": round(pos.market_value, 2),
                    "unrealized_pl": round(pos.unrealized_pl, 2),
                    "unrealized_plpc": round(pos.unrealized_plpc * 100, 2),
                    "current_price": round(pos.current_price, 2),
                    "avg_entry_price": round(pos.avg_entry_price, 2),
                }
                for pos in self._portfolio.positions.values()
            ]

            # Get exposure metrics from LivePortfolio
            exposure = self._portfolio.exposure()

            result = {
                "account": self._portfolio.summary(),
                "daily_pnl": {
                    "value": round(daily_pl, 2),
                    "percent": round(daily_pl_pct, 2),
                    "last_equity": round(account["last_equity"], 2),
                },
                "positions": {
                    "count": len(self._portfolio.positions),
                    "total_value": round(self._portfolio.long_market_value + abs(self._portfolio.short_market_value), 2),
                    "total_unrealized_pl": round(self._portfolio.unrealized_pl, 2),
                    "details": position_details,
                },
                "risk_metrics": {
                    "position_count": len(self._portfolio.positions),
                    "long_exposure": exposure["long_exposure"],
                    "short_exposure": exposure["short_exposure"],
                    "net_exposure": exposure["net_exposure"],
                    "gross_exposure": exposure["gross_exposure"],
                    "cash_pct": round((self._portfolio.cash / self._portfolio.equity) * 100, 2) if self._portfolio.equity > 0 else 0,
                },
                "status": {
                    "trading_blocked": account["trading_blocked"],
                    "account_blocked": account["account_blocked"],
                    "pattern_day_trader": account["pattern_day_trader"],
                    "daytrade_count": account["daytrade_count"],
                },
            }

            # Include open orders if requested
            if include_orders:
                orders = self._client.get_orders(status="open")
                result["open_orders"] = {
                    "count": len(orders),
                    "orders": orders,
                }

            # Include filled orders if requested
            if include_history:
                filled = self._client.get_orders(status="closed")
                # Filter to today's fills only
                result["todays_fills"] = {
                    "count": len(filled),
                    "orders": filled[:20],  # Limit to most recent 20
                }

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            return json.dumps({
                "error": str(e),
                "account": None,
                "positions": None,
            })
