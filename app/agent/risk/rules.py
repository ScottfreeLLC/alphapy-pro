"""Circuit breaker rules — hard stops to prevent catastrophic losses."""

import logging
from datetime import datetime
from typing import Optional

from ..config import AgentConfig

logger = logging.getLogger(__name__)

# Circuit breaker thresholds
DAILY_LOSS_MULTIPLIER = 2.0    # Trip at 2x the daily loss limit
MAX_CONSECUTIVE_LOSSES = 8     # Trip after 8 consecutive losses


class CircuitBreaker:
    """
    Emergency stop mechanism.

    Trips when:
    - Daily losses exceed 2x the daily loss limit (4% of equity)
    - 8 consecutive losing trades

    When tripped, ALL trading is halted until manually reset.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._tripped = False
        self._tripped_at: Optional[datetime] = None
        self._trip_reason: Optional[str] = None
        self._consecutive_losses = 0
        self._daily_loss = 0.0

    def is_tripped(self) -> bool:
        return self._tripped

    def check_and_trip(self, daily_loss: float, consecutive_losses: int) -> bool:
        """
        Check if circuit breaker should trip.

        Args:
            daily_loss: Current daily loss as a fraction of equity (e.g. 0.03 = 3%)
            consecutive_losses: Number of consecutive losing trades

        Returns:
            True if circuit breaker just tripped
        """
        self._daily_loss = daily_loss
        self._consecutive_losses = consecutive_losses

        if self._tripped:
            return False  # Already tripped

        # Check daily loss threshold
        loss_limit = self.config.max_daily_loss_pct * DAILY_LOSS_MULTIPLIER
        if abs(daily_loss) >= loss_limit:
            self._trip(f"Daily loss {abs(daily_loss)*100:.1f}% exceeds {loss_limit*100:.1f}% limit")
            return True

        # Check consecutive losses
        if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self._trip(f"{consecutive_losses} consecutive losses (limit: {MAX_CONSECUTIVE_LOSSES})")
            return True

        return False

    def reset(self):
        """Manually reset the circuit breaker. Requires user action."""
        if self._tripped:
            logger.info(
                f"Circuit breaker reset. Was tripped at {self._tripped_at} "
                f"for: {self._trip_reason}"
            )
        self._tripped = False
        self._tripped_at = None
        self._trip_reason = None
        self._consecutive_losses = 0
        self._daily_loss = 0.0

    def get_status(self) -> dict:
        return {
            "tripped": self._tripped,
            "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
            "reason": self._trip_reason,
            "consecutive_losses": self._consecutive_losses,
            "daily_loss": round(self._daily_loss, 4),
        }

    def _trip(self, reason: str):
        """Activate the circuit breaker."""
        self._tripped = True
        self._tripped_at = datetime.now()
        self._trip_reason = reason
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")
