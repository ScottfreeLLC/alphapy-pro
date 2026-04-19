"""Cross-agent shared risk management for dual-agent coordination."""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional

from ..config import AgentConfig
from ..core.signal import TradeSignal
from .rules import CircuitBreaker

logger = logging.getLogger(__name__)

# Cross-agent limits
MAX_COMBINED_EXPOSURE_PCT = 0.80   # 80% total across both agents
MAX_COMBINED_POSITIONS = 15        # 15 total positions
MAX_DAY_TRADES_PER_WEEK = 3       # PDT avoidance on margin accounts
COMBINED_DAILY_LOSS_PCT = 0.03    # 3% combined daily loss circuit breaker


class SharedRiskManager:
    """
    Cross-agent risk manager enforcing combined portfolio limits.

    Sits above per-agent RiskManagers and enforces:
    - Total exposure across both agents <= 80%
    - Combined max 15 positions
    - Day trading round-trip counter (PDT avoidance)
    - Combined daily loss circuit breaker
    """

    def __init__(self, circuit_breaker: Optional[CircuitBreaker] = None):
        self.circuit_breaker = circuit_breaker
        # Per-agent position tracking: {agent_type: {symbol: direction}}
        self._positions: Dict[str, Dict[str, str]] = {"swing": {}, "day": {}}
        # Day trade counter: {date_str: count}
        self._day_trades: Dict[str, int] = {}
        # Combined daily P&L
        self._daily_pnl: float = 0.0
        self._pnl_date: Optional[date] = None

    @property
    def total_positions(self) -> int:
        return sum(len(p) for p in self._positions.values())

    def can_open_position(self, agent_type: str, symbol: str, direction: str) -> List[str]:
        """
        Check if a new position is allowed under shared constraints.

        Returns list of rejection reasons (empty = allowed).
        """
        reasons = []

        # Combined position limit
        if self.total_positions >= MAX_COMBINED_POSITIONS:
            reasons.append(
                f"Combined positions ({self.total_positions}) at max ({MAX_COMBINED_POSITIONS})"
            )

        # Cross-agent duplicate check — different agents can't hold conflicting
        # positions in the same symbol
        for other_agent, positions in self._positions.items():
            if other_agent == agent_type:
                continue
            if symbol in positions and positions[symbol] != direction:
                reasons.append(
                    f"{other_agent} agent has opposite {positions[symbol]} position in {symbol}"
                )

        # PDT check for day trading agent
        if agent_type == "day":
            today_str = date.today().isoformat()
            day_trades_today = self._day_trades.get(today_str, 0)
            if day_trades_today >= MAX_DAY_TRADES_PER_WEEK:
                reasons.append(
                    f"Day trade limit reached ({day_trades_today}/{MAX_DAY_TRADES_PER_WEEK} today)"
                )

        # Combined daily loss
        self._ensure_daily_reset()
        if self._daily_pnl < 0 and abs(self._daily_pnl) >= COMBINED_DAILY_LOSS_PCT:
            reasons.append(
                f"Combined daily loss {abs(self._daily_pnl)*100:.1f}% exceeds {COMBINED_DAILY_LOSS_PCT*100:.0f}% limit"
            )

        # Circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_tripped():
            reasons.append("Shared circuit breaker tripped")

        return reasons

    def filter_signals(self, agent_type: str, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Filter signals through shared risk constraints."""
        approved = []
        for signal in signals:
            reasons = self.can_open_position(agent_type, signal.symbol, signal.direction.value)
            if reasons:
                logger.info(f"Shared risk rejected {signal.symbol}: {', '.join(reasons)}")
                signal.metadata["shared_risk_rejected"] = reasons
                continue
            approved.append(signal)
        return approved

    def register_position(self, agent_type: str, symbol: str, direction: str):
        """Register a new position for an agent."""
        if agent_type not in self._positions:
            self._positions[agent_type] = {}
        self._positions[agent_type][symbol] = direction

    def close_position(self, agent_type: str, symbol: str):
        """Close a position and track day trades."""
        if agent_type in self._positions:
            self._positions[agent_type].pop(symbol, None)

        # Count as a day trade round-trip for day agent
        if agent_type == "day":
            today_str = date.today().isoformat()
            self._day_trades[today_str] = self._day_trades.get(today_str, 0) + 1

    def update_daily_pnl(self, pnl_delta: float):
        """Update the combined daily P&L."""
        self._ensure_daily_reset()
        self._daily_pnl += pnl_delta

    def get_status(self) -> Dict:
        """Get combined risk status."""
        return {
            "total_positions": self.total_positions,
            "max_combined_positions": MAX_COMBINED_POSITIONS,
            "positions_by_agent": {
                agent: dict(positions)
                for agent, positions in self._positions.items()
            },
            "combined_daily_pnl": round(self._daily_pnl, 4),
            "combined_daily_loss_limit": COMBINED_DAILY_LOSS_PCT,
            "day_trades_today": self._day_trades.get(date.today().isoformat(), 0),
            "max_day_trades": MAX_DAY_TRADES_PER_WEEK,
            "circuit_breaker_tripped": (
                self.circuit_breaker.is_tripped() if self.circuit_breaker else False
            ),
        }

    def _ensure_daily_reset(self):
        """Reset daily P&L at the start of each new trading day."""
        today = date.today()
        if self._pnl_date != today:
            self._daily_pnl = 0.0
            self._pnl_date = today
