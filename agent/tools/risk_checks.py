"""Risk Check tool for validating trades against risk constraints."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import Tool
from ..utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckTool(Tool):
    """Tool for validating trades against risk management rules.

    Checks position size limits, portfolio exposure, and daily loss limits
    before allowing trades to execute.
    """

    name: str = "check_risk"
    description: str = """
Validates a proposed trade against risk management rules:
- Position size limits (max $ per position)
- Portfolio exposure limits
- Daily loss limit (max drawdown)
- Concentration limits (max % in one symbol)
Returns approval/rejection with detailed reasons.
ALWAYS call this before executing any trade.
"""
    input_schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol to trade",
            },
            "side": {
                "type": "string",
                "enum": ["buy", "sell"],
                "description": "Order side",
            },
            "quantity": {
                "type": "number",
                "description": "Number of shares/units",
            },
            "price": {
                "type": "number",
                "description": "Current price per share",
            },
        },
        "required": ["symbol", "side", "quantity", "price"],
    })

    _client: AlpacaClient = field(default=None, repr=False)

    # Default risk limits
    max_position_value: float = 5000.0
    max_portfolio_exposure: float = 25000.0
    max_positions: int = 5
    max_symbol_pct: float = 0.25
    daily_loss_limit: float = 0.02
    position_stop_loss: float = 0.01
    min_order_value: float = 100.0
    no_new_trades_before_close: int = 15  # minutes

    def __post_init__(self):
        """Initialize the Alpaca client."""
        if self._client is None:
            self._client = AlpacaClient()

    def configure(
        self,
        max_position_value: Optional[float] = None,
        max_portfolio_exposure: Optional[float] = None,
        max_positions: Optional[int] = None,
        max_symbol_pct: Optional[float] = None,
        daily_loss_limit: Optional[float] = None,
        position_stop_loss: Optional[float] = None,
        min_order_value: Optional[float] = None,
        no_new_trades_before_close: Optional[int] = None,
    ) -> None:
        """Configure risk limits.

        Args:
            max_position_value: Maximum $ value per position
            max_portfolio_exposure: Maximum total $ in positions
            max_positions: Maximum number of concurrent positions
            max_symbol_pct: Maximum % of portfolio in one symbol
            daily_loss_limit: Stop trading if daily loss exceeds this %
            position_stop_loss: Exit position if loss exceeds this %
            min_order_value: Minimum order value in $
            no_new_trades_before_close: Minutes before close to stop new trades
        """
        if max_position_value is not None:
            self.max_position_value = max_position_value
        if max_portfolio_exposure is not None:
            self.max_portfolio_exposure = max_portfolio_exposure
        if max_positions is not None:
            self.max_positions = max_positions
        if max_symbol_pct is not None:
            self.max_symbol_pct = max_symbol_pct
        if daily_loss_limit is not None:
            self.daily_loss_limit = daily_loss_limit
        if position_stop_loss is not None:
            self.position_stop_loss = position_stop_loss
        if min_order_value is not None:
            self.min_order_value = min_order_value
        if no_new_trades_before_close is not None:
            self.no_new_trades_before_close = no_new_trades_before_close

    async def execute(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> str:
        """Validate a proposed trade against risk rules.

        Returns:
            JSON string with approval status and detailed checks.
        """
        try:
            # Get current account and positions
            account = self._client.get_account()
            positions = self._client.get_positions()

            order_value = quantity * price
            checks = []
            approved = True

            # Check 1: Minimum order value
            if order_value < self.min_order_value:
                checks.append({
                    "rule": "min_order_value",
                    "passed": False,
                    "message": f"Order value ${order_value:.2f} below minimum ${self.min_order_value}",
                })
                approved = False
            else:
                checks.append({
                    "rule": "min_order_value",
                    "passed": True,
                    "message": f"Order value ${order_value:.2f} meets minimum",
                })

            # Check 2: Position size limit
            if order_value > self.max_position_value:
                checks.append({
                    "rule": "max_position_value",
                    "passed": False,
                    "message": f"Order value ${order_value:.2f} exceeds max ${self.max_position_value}",
                })
                approved = False
            else:
                checks.append({
                    "rule": "max_position_value",
                    "passed": True,
                    "message": f"Position size within limits",
                })

            # Check 3: Portfolio exposure
            current_exposure = sum(float(p["market_value"]) for p in positions)
            new_exposure = current_exposure + order_value if side == "buy" else current_exposure

            if new_exposure > self.max_portfolio_exposure:
                checks.append({
                    "rule": "max_portfolio_exposure",
                    "passed": False,
                    "message": f"Total exposure ${new_exposure:.2f} would exceed max ${self.max_portfolio_exposure}",
                })
                approved = False
            else:
                checks.append({
                    "rule": "max_portfolio_exposure",
                    "passed": True,
                    "message": f"Portfolio exposure ${new_exposure:.2f} within limits",
                })

            # Check 4: Max positions (for new positions only)
            existing_position = any(p["symbol"] == symbol.upper() for p in positions)
            if not existing_position and side == "buy":
                if len(positions) >= self.max_positions:
                    checks.append({
                        "rule": "max_positions",
                        "passed": False,
                        "message": f"Already at max {self.max_positions} positions",
                    })
                    approved = False
                else:
                    checks.append({
                        "rule": "max_positions",
                        "passed": True,
                        "message": f"Position count {len(positions)} within limit",
                    })

            # Check 5: Concentration limit
            equity = account["equity"]
            if equity > 0:
                concentration = order_value / equity
                if concentration > self.max_symbol_pct:
                    checks.append({
                        "rule": "max_symbol_pct",
                        "passed": False,
                        "message": f"Position {concentration*100:.1f}% exceeds max {self.max_symbol_pct*100:.1f}%",
                    })
                    approved = False
                else:
                    checks.append({
                        "rule": "max_symbol_pct",
                        "passed": True,
                        "message": f"Concentration {concentration*100:.1f}% within limits",
                    })

            # Check 6: Daily loss limit
            daily_pnl_pct = (account["equity"] - account["last_equity"]) / account["last_equity"]
            if daily_pnl_pct < -self.daily_loss_limit:
                checks.append({
                    "rule": "daily_loss_limit",
                    "passed": False,
                    "message": f"Daily loss {daily_pnl_pct*100:.2f}% exceeds limit {self.daily_loss_limit*100:.1f}%",
                })
                approved = False
            else:
                checks.append({
                    "rule": "daily_loss_limit",
                    "passed": True,
                    "message": f"Daily P&L {daily_pnl_pct*100:.2f}% within limits",
                })

            # Check 7: Trading blocked
            if account["trading_blocked"]:
                checks.append({
                    "rule": "trading_blocked",
                    "passed": False,
                    "message": "Trading is blocked on this account",
                })
                approved = False

            # Adjust quantity if partially approved (for position sizing)
            adjusted_quantity = quantity
            if approved:
                # Calculate max allowed quantity based on limits
                max_by_position = self.max_position_value / price
                max_by_exposure = (self.max_portfolio_exposure - current_exposure) / price
                max_by_concentration = (self.max_symbol_pct * equity) / price

                max_allowed = min(max_by_position, max_by_exposure, max_by_concentration)
                if quantity > max_allowed:
                    adjusted_quantity = max_allowed
                    checks.append({
                        "rule": "quantity_adjustment",
                        "passed": True,
                        "message": f"Quantity adjusted from {quantity} to {adjusted_quantity:.4f}",
                    })

            return json.dumps({
                "approved": approved,
                "symbol": symbol,
                "side": side,
                "requested_quantity": quantity,
                "approved_quantity": adjusted_quantity if approved else 0,
                "order_value": round(order_value, 2),
                "checks": checks,
                "risk_limits": {
                    "max_position_value": self.max_position_value,
                    "max_portfolio_exposure": self.max_portfolio_exposure,
                    "max_positions": self.max_positions,
                    "max_symbol_pct": self.max_symbol_pct,
                    "daily_loss_limit": self.daily_loss_limit,
                },
            })

        except Exception as e:
            logger.error(f"Error checking risk: {e}")
            return json.dumps({
                "approved": False,
                "error": str(e),
                "symbol": symbol,
                "checks": [],
            })
