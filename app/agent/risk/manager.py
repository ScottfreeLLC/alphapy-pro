"""Risk manager: position sizing, exposure limits, signal filtering."""

import logging
from typing import Dict, List, Optional

from ..config import AgentConfig
from ..core.signal import TradeSignal
from .rules import CircuitBreaker

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforces risk management rules on trade signals.

    Rules:
    - Max 10% per position
    - Max 60% total exposure
    - Max 10 concurrent positions
    - Min 1.5:1 risk/reward ratio
    - No duplicate positions (same symbol, same direction)
    - 2% daily loss limit
    """

    def __init__(
        self,
        config: AgentConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.config = config
        self.circuit_breaker = circuit_breaker or CircuitBreaker(config)
        self._current_positions: Dict[str, str] = {}  # symbol -> direction
        self._daily_pnl: float = 0.0

    def filter_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Filter signals through all risk rules. Returns only acceptable signals."""
        if self.circuit_breaker.is_tripped():
            logger.warning("Circuit breaker tripped — rejecting all signals")
            return []

        approved = []
        for signal in signals:
            reasons = self._check_signal(signal)
            if reasons:
                logger.info(
                    f"Risk rejected {signal.symbol}: {', '.join(reasons)}"
                )
                signal.metadata["risk_rejected"] = reasons
                continue
            approved.append(signal)

        return approved

    def _check_signal(self, signal: TradeSignal) -> List[str]:
        """Check a single signal against all risk rules. Returns list of rejection reasons."""
        reasons = []

        # 1. Risk/reward ratio
        if signal.risk_reward_ratio < self.config.min_risk_reward:
            reasons.append(
                f"R:R {signal.risk_reward_ratio:.1f} < min {self.config.min_risk_reward}"
            )

        # 2. Position size limit
        if signal.position_size_pct > self.config.max_position_pct:
            reasons.append(
                f"Position size {signal.position_size_pct*100:.0f}% > max {self.config.max_position_pct*100:.0f}%"
            )

        # 3. Duplicate position check
        if signal.symbol in self._current_positions:
            existing_dir = self._current_positions[signal.symbol]
            if existing_dir == signal.direction.value:
                reasons.append(f"Already have {existing_dir} position in {signal.symbol}")

        # 4. Max positions
        if len(self._current_positions) >= self.config.max_positions:
            reasons.append(
                f"At max positions ({self.config.max_positions})"
            )

        # 5. Daily loss limit
        if self._daily_pnl < 0 and abs(self._daily_pnl) >= self.config.max_daily_loss_pct:
            reasons.append("Daily loss limit reached")

        return reasons

    def register_position(self, symbol: str, direction: str):
        """Register a new open position."""
        self._current_positions[symbol] = direction

    def close_position(self, symbol: str):
        """Remove a closed position."""
        self._current_positions.pop(symbol, None)

    def update_daily_pnl(self, pnl: float):
        """Update the running daily P&L."""
        self._daily_pnl += pnl

    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        self._daily_pnl = 0.0

    def get_exposure(self) -> Dict:
        """Get current exposure summary."""
        return {
            "open_positions": len(self._current_positions),
            "max_positions": self.config.max_positions,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_loss_limit": self.config.max_daily_loss_pct,
            "circuit_breaker_tripped": self.circuit_breaker.is_tripped(),
            "positions": dict(self._current_positions),
        }
