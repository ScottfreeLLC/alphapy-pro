"""Live portfolio management with Alpaca broker integration.

This module bridges AlphaPy's backtest Portfolio concepts with live trading,
providing real-time position tracking and P&L calculations.

All data operations use Polars DataFrames.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """A live position synchronized with the broker.

    Attributes:
        symbol: Trading symbol
        qty: Current quantity (positive=long, negative=short)
        side: "long" or "short"
        avg_entry_price: Average entry price
        current_price: Current market price
        market_value: Current market value
        cost_basis: Total cost basis
        unrealized_pl: Unrealized profit/loss
        unrealized_plpc: Unrealized P&L percentage
        change_today: Change today (from broker)
        last_updated: Timestamp of last sync
    """

    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    change_today: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_alpaca(cls, position_dict: dict) -> "LivePosition":
        """Create LivePosition from Alpaca position dictionary.

        Args:
            position_dict: Dictionary from AlpacaClient.get_positions()

        Returns:
            LivePosition instance
        """
        return cls(
            symbol=position_dict["symbol"],
            qty=position_dict["qty"],
            side=position_dict["side"],
            avg_entry_price=position_dict["avg_entry_price"],
            current_price=position_dict["current_price"],
            market_value=position_dict["market_value"],
            cost_basis=position_dict["cost_basis"],
            unrealized_pl=position_dict["unrealized_pl"],
            unrealized_plpc=position_dict["unrealized_plpc"],
            change_today=position_dict.get("change_today", 0.0),
            last_updated=datetime.now(),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pl": self.unrealized_pl,
            "unrealized_plpc": self.unrealized_plpc,
            "change_today": self.change_today,
            "last_updated": self.last_updated.isoformat(),
        }


class LivePortfolio:
    """Live portfolio management with broker synchronization.

    Bridges AlphaPy's backtest Portfolio concepts with live Alpaca trading.
    Provides real-time position tracking and unified P&L calculations.

    All DataFrames are Polars.

    Attributes:
        positions: Dictionary of symbol -> LivePosition
        cash: Available cash
        equity: Total portfolio equity
        buying_power: Available buying power
        portfolio_value: Total portfolio value
        unrealized_pl: Total unrealized P&L
        realized_pl: Total realized P&L (tracked internally)
        history: Polars DataFrame of portfolio snapshots
    """

    def __init__(self, broker=None):
        """Initialize live portfolio.

        Args:
            broker: AlpacaClient instance (optional, can set later)
        """
        self._broker = broker
        self.positions: dict[str, LivePosition] = {}

        # Account metrics
        self.cash: float = 0.0
        self.equity: float = 0.0
        self.buying_power: float = 0.0
        self.portfolio_value: float = 0.0
        self.long_market_value: float = 0.0
        self.short_market_value: float = 0.0

        # P&L tracking
        self.unrealized_pl: float = 0.0
        self.realized_pl: float = 0.0
        self._starting_equity: Optional[float] = None

        # History tracking
        self._history_records: list[dict] = []

        # Sync state
        self._last_sync: Optional[datetime] = None

    def set_broker(self, broker) -> None:
        """Set the broker client.

        Args:
            broker: AlpacaClient instance
        """
        self._broker = broker
        logger.info("Broker client set for LivePortfolio")

    def sync(self) -> None:
        """Synchronize portfolio state with broker.

        Fetches latest account info and positions from Alpaca,
        updating all internal state.

        Raises:
            RuntimeError: If broker is not set
        """
        if self._broker is None:
            raise RuntimeError("Broker not set. Call set_broker() first.")

        try:
            # Sync account info
            account = self._broker.get_account()
            self.cash = account["cash"]
            self.equity = account["equity"]
            self.buying_power = account["buying_power"]
            self.portfolio_value = account["portfolio_value"]
            self.long_market_value = account["long_market_value"]
            self.short_market_value = account["short_market_value"]

            # Track starting equity for return calculations
            if self._starting_equity is None:
                self._starting_equity = self.equity

            # Sync positions
            positions = self._broker.get_positions()
            self.positions = {}
            self.unrealized_pl = 0.0

            for pos_dict in positions:
                pos = LivePosition.from_alpaca(pos_dict)
                self.positions[pos.symbol] = pos
                self.unrealized_pl += pos.unrealized_pl

            self._last_sync = datetime.now()

            # Record history snapshot
            self._record_snapshot()

            logger.debug(
                f"Portfolio synced: ${self.equity:.2f} equity, "
                f"{len(self.positions)} positions"
            )

        except Exception as e:
            logger.error(f"Error syncing portfolio: {e}")
            raise

    def _record_snapshot(self) -> None:
        """Record a portfolio snapshot for history tracking."""
        snapshot = {
            "timestamp": datetime.now(),
            "equity": self.equity,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "unrealized_pl": self.unrealized_pl,
            "realized_pl": self.realized_pl,
            "num_positions": len(self.positions),
            "long_value": self.long_market_value,
            "short_value": self.short_market_value,
        }
        self._history_records.append(snapshot)

    def get_position(self, symbol: str) -> Optional[LivePosition]:
        """Get position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            LivePosition or None if no position
        """
        return self.positions.get(symbol.upper())

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if position exists
        """
        return symbol.upper() in self.positions

    def get_quantity(self, symbol: str) -> float:
        """Get current quantity for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position quantity (0 if no position)
        """
        pos = self.get_position(symbol)
        return pos.qty if pos else 0.0

    def positions_df(self) -> pl.DataFrame:
        """Get all positions as a Polars DataFrame.

        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pl.DataFrame(schema={
                "symbol": pl.Utf8,
                "qty": pl.Float64,
                "side": pl.Utf8,
                "avg_entry_price": pl.Float64,
                "current_price": pl.Float64,
                "market_value": pl.Float64,
                "cost_basis": pl.Float64,
                "unrealized_pl": pl.Float64,
                "unrealized_plpc": pl.Float64,
                "change_today": pl.Float64,
            })

        records = [pos.to_dict() for pos in self.positions.values()]
        return pl.DataFrame(records).drop("last_updated")

    def history_df(self) -> pl.DataFrame:
        """Get portfolio history as a Polars DataFrame.

        Returns:
            DataFrame with portfolio snapshots over time
        """
        if not self._history_records:
            return pl.DataFrame(schema={
                "timestamp": pl.Datetime,
                "equity": pl.Float64,
                "cash": pl.Float64,
                "portfolio_value": pl.Float64,
                "unrealized_pl": pl.Float64,
                "realized_pl": pl.Float64,
                "num_positions": pl.Int64,
                "long_value": pl.Float64,
                "short_value": pl.Float64,
            })

        return pl.DataFrame(self._history_records)

    def total_return(self) -> float:
        """Calculate total return since tracking started.

        Returns:
            Return as decimal (0.10 = 10%)
        """
        if self._starting_equity is None or self._starting_equity == 0:
            return 0.0
        return (self.equity - self._starting_equity) / self._starting_equity

    def total_return_pct(self) -> float:
        """Calculate total return percentage.

        Returns:
            Return as percentage (10.0 = 10%)
        """
        return self.total_return() * 100

    def summary(self) -> dict:
        """Get portfolio summary.

        Returns:
            Dictionary with key portfolio metrics
        """
        return {
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "portfolio_value": self.portfolio_value,
            "num_positions": len(self.positions),
            "long_market_value": self.long_market_value,
            "short_market_value": self.short_market_value,
            "unrealized_pl": self.unrealized_pl,
            "realized_pl": self.realized_pl,
            "total_return_pct": round(self.total_return_pct(), 2),
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
        }

    def can_buy(self, symbol: str, qty: float, price: float) -> bool:
        """Check if we can buy the specified quantity.

        Args:
            symbol: Trading symbol
            qty: Quantity to buy
            price: Estimated price

        Returns:
            True if buying power is sufficient
        """
        cost = qty * price
        return cost <= self.buying_power

    def position_size(
        self,
        price: float,
        risk_pct: float = 0.02,
        stop_loss_pct: float = 0.05,
    ) -> int:
        """Calculate position size based on risk parameters.

        Uses fixed-fractional position sizing:
        - risk_pct: Maximum portfolio risk per trade
        - stop_loss_pct: Expected stop loss distance

        Args:
            price: Entry price
            risk_pct: Portfolio risk per trade (default 2%)
            stop_loss_pct: Stop loss distance (default 5%)

        Returns:
            Number of shares/units to trade
        """
        if price <= 0 or stop_loss_pct <= 0:
            return 0

        risk_amount = self.equity * risk_pct
        risk_per_share = price * stop_loss_pct
        shares = int(risk_amount / risk_per_share)

        # Ensure we don't exceed buying power
        max_shares = int(self.buying_power / price)
        return min(shares, max_shares)

    def record_realized_pl(self, amount: float) -> None:
        """Record realized P&L from a closed trade.

        Call this after closing a position to track realized gains/losses.

        Args:
            amount: Realized P&L amount (positive=profit, negative=loss)
        """
        self.realized_pl += amount
        logger.info(f"Recorded realized P&L: ${amount:+.2f} (total: ${self.realized_pl:.2f})")

    def exposure(self) -> dict:
        """Calculate portfolio exposure metrics.

        Returns:
            Dictionary with exposure breakdown
        """
        long_exposure = self.long_market_value / self.equity if self.equity > 0 else 0
        short_exposure = abs(self.short_market_value) / self.equity if self.equity > 0 else 0
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure

        return {
            "long_exposure": round(long_exposure, 4),
            "short_exposure": round(short_exposure, 4),
            "net_exposure": round(net_exposure, 4),
            "gross_exposure": round(gross_exposure, 4),
        }

    def sector_allocation(self) -> pl.DataFrame:
        """Get sector allocation (if sector info available).

        Note: Requires position metadata - returns empty if not available.

        Returns:
            DataFrame with sector allocations
        """
        # Placeholder - would need sector data from market data source
        return pl.DataFrame(schema={
            "sector": pl.Utf8,
            "market_value": pl.Float64,
            "weight": pl.Float64,
        })

    def __repr__(self) -> str:
        return (
            f"LivePortfolio(equity=${self.equity:.2f}, "
            f"positions={len(self.positions)}, "
            f"return={self.total_return_pct():+.2f}%)"
        )
